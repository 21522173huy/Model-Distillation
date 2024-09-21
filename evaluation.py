%%writefile evaluation.py

from torch import nn, optim
from tqdm import tqdm
from student.dataset_ import create_dataloaders
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
import os
import yaml
import sys
import argparse
import torch
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from teacher.teacher_model import TeacherModel


def evaluate_models(teacher_model, student_model, val_dataloader, save_path='evaluation_results.json'):
    teacher_model.eval()
    student_model.eval(), student_model.to('cuda')
    
    teacher_predictions = []
    student_predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            input_ids, attention_mask = batch['input_ids'].to('cuda'), batch['attention_mask'].to('cuda')
            labels = batch['labels'].to('cuda')

            # Predictions from the teacher model
            teacher_outputs = teacher_model(input_ids, attention_mask=attention_mask)
            teacher_preds = torch.argmax(teacher_outputs.logits, dim=-1)
            teacher_predictions.extend(teacher_preds.cpu().numpy())

            # Predictions from the student model
            student_outputs = student_model(input_ids, attention_mask=attention_mask)
            student_preds = torch.argmax(student_outputs.logits, dim=-1)
            student_predictions.extend(student_preds.cpu().numpy())

            # True labels
            true_labels.extend(labels.cpu().numpy())

    # Calculate metrics for teacher model
    teacher_accuracy = accuracy_score(true_labels, teacher_predictions)
    teacher_precision = precision_score(true_labels, teacher_predictions, average='weighted')
    teacher_recall = recall_score(true_labels, teacher_predictions, average='weighted')
    teacher_f1 = f1_score(true_labels, teacher_predictions, average='weighted')

    # Calculate metrics for student model
    student_accuracy = accuracy_score(true_labels, student_predictions)
    student_precision = precision_score(true_labels, student_predictions, average='weighted')
    student_recall = recall_score(true_labels, student_predictions, average='weighted')
    student_f1 = f1_score(true_labels, student_predictions, average='weighted')

    # Save results to a JSON file
    results = {
        "Teacher": {
            "accuracy_score": teacher_accuracy,
            "precision_score": teacher_precision,
            "recall_score": teacher_recall,
            "f1_score": teacher_f1
        },
        "Student": {
            "accuracy_score": student_accuracy,
            "precision_score": student_precision,
            "recall_score": student_recall,
            "f1_score": student_f1
        }
    }

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation results saved to {save_path}")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_checkpoint', type=str)
    parser.add_argument('--student_checkpoint', type=str)
    parser.add_argument('--save_path', type=str)

    args = parser.parse_args()

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

    # Load Pretrained PLMs
    student_model = AutoModelForSequenceClassification.from_pretrained(
        args.student_checkpoint,
        num_labels=3
    )

    evaluate_models(pretrained_teacher_model, student_model, val_dataloader, save_path = args.save_path)

if __name__ == "__main__":
    main()
