
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorWithPadding
from peft import get_peft_model, LoraConfig, TaskType
from typing import Callable, Any, Tuple

class TeacherModel:
    def __init__(self, model_name: str, training_config, is_quantized=False):
        print(f"Loading teacher model: {model_name}")
        self.model_name = model_name
        self.training_config = training_config
        
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        ) if is_quantized else None

        self = self._load_teacher_model()
        self.trainable = True  # Assuming the model is trainable
        self.load_in_nbits = 16  # Assuming 16-bit precision for fp16 training

        print("Teacher model loaded successfully")

    def _load_teacher_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,
            quantization_config=self.quantization_config,
        )
        lora_config = LoraConfig(**self.training_config)
        model = get_peft_model(model, lora_config)
        return model

    def __optimizer(self, _load_in_4bit: bool, _load_in_8bit: bool, **unused_kwargs) -> str:
        return "adamw_hf"

    def save_checkpoint(self, save_path: str):
        self.teacher_model.save_pretrained(save_path)

    def finetune(
        self,
        dataset: Dataset,
        batch_size: int,
        gradient_accumulation_steps: int = 1,
        data_collator: Callable[..., Any] = None,
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.98),
        weight_decay: float = 1e-3,
        eps: float = 1e-8,
        epochs: int = 3,
    ):
        if not self.trainable:
            raise ValueError("currently not supporting finetuning with this model's configurations")

        optimizer_name = self.__optimizer(_load_in_4bit=True, _load_in_8bit=False)  # Adjust as needed

        training_args = TrainingArguments(
            "./out",
            per_device_train_batch_size=batch_size,
            optim=optimizer_name,
            fp16=(self.load_in_nbits <= 16),
            learning_rate=lr,
            adam_beta1=betas[0],
            adam_beta2=betas[1],
            adam_epsilon=eps,
            weight_decay=weight_decay,
            num_train_epochs=epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=1.0,
        )

        if data_collator is None:
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.teacher_model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()

    def evaluate(self, dataloader: DataLoader):
        self.teacher_model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input_ids'].to(self.teacher_model.device)
                labels = batch['labels'].to(self.teacher_model.device)

                outputs = self.teacher_model(inputs, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                predictions = torch.argmax(outputs.logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        print(f"Evaluation - Loss: {avg_loss}, Accuracy: {accuracy}")
