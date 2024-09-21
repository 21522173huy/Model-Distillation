from torch import nn, optim
from tqdm import tqdm
from dataset_ import create_dataloaders
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
import os
import yaml
import sys 
import argparse
import torch

# Add the parent directory to the system path to allow imports from the teacher subfolder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from teacher.teacher_model import TeacherModel

def distill(teacher_model, student_model, dataloader,
            optimizer, 
            ce_loss=nn.CrossEntropyLoss(),
            temperature=2.0, 
            soft_target_loss_weight=0.5,
            ce_loss_weight=0.5,
            epochs=10, 
            max_grad_norm=1.0,
            save_dir='student/student_checkpoints'):
    
    # Define the learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    teacher_model.to('cuda')  # Move teacher model to GPU

    for epoch in tqdm(range(epochs)):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        teacher_model.eval()  # Set teacher model to evaluation mode
        student_model.train()  # Set student model to training mode
        student_model.to('cuda')  # Move student model to GPU

        for batch in dataloader:  # Iterate over the data
            optimizer.zero_grad()  # Zero the gradients
            input_ids, attention_mask = batch['input_ids'].to('cuda'), batch['attention_mask'].to('cuda')
            labels = batch['labels'].to('cuda')

            # Predictions from the teacher model
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids, attention_mask=attention_mask).logits

            # Predictions from the student model
            student_logits = student_model(input_ids, attention_mask=attention_mask).logits

            # Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / temperature, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (temperature**2)

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_grad_norm)

            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

            # Calculate accuracy
            predictions = torch.argmax(student_logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

        # Step the scheduler
        scheduler.step()

        # Calculate average loss and accuracy for the epoch
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions

        # Log the loss and accuracy for the epoch
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

        # Save the student model checkpoint
        checkpoint_dir = os.path.join(save_dir, f'epoch-{epoch + 1}')
        student_model.save_pretrained(checkpoint_dir)
        print(f"Saved checkpoint to {checkpoint_dir}")

def main():
  
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_checkpoint', type=str)
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()
    print(f'Teacher Checkpoint: {args.teacher_checkpoint}')
    # Load training configurations from a YAML file
    with open("teacher/training-config.yaml", encoding="utf8") as conf:
        config = yaml.safe_load(conf)
    
    model_name = config["model_name"]
    training_config = config["training_config"]
    training_params = config["training"]

    # Ensure numeric parameters are correctly parsed as floats
    training_params["learning_rate"] = float(training_params["learning_rate"])
    training_params["weight_decay"] = float(training_params["weight_decay"])
    training_params["eps"] = float(training_params["eps"])

    # Initialize the TeacherModel
    teacher_model = TeacherModel(model_name, training_config)

    # Load Pretrained Teacher
    pretrained_teacher_model = PeftModel.from_pretrained(teacher_model.teacher_model.base_model.model, args.teacher_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_checkpoint)

    # Load Dataloader
    train_dataloader, test_dataloader, val_dataloader = create_dataloaders(tokenizer, batch_size=8, subset_ratio=0.15)

    # Load PLMs 
    student_model = AutoModelForSequenceClassification.from_pretrained(
        os.getenv('Student_PATH'),
        num_labels=3
    )

    optimizer = torch.optim.AdamW(student_model.parameters())
    distill(pretrained_teacher_model, student_model, train_dataloader, optimizer, epochs = args.epochs)

if __name__ == "__main__":
    main()
