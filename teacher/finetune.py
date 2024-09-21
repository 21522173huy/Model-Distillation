import yaml
import time
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from teacher_model import TeacherModel

def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], padding=True, truncation=True, max_length = 256, return_tensors = 'pt')

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

# Load the dataset
dataset = load_dataset('tweet_eval', 'sentiment')
tokenizer = teacher_model.tokenizer

# Tokenize the dataset
tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

# Remove columns other than input_ids, attention_mask, and labels
tokenized_datasets = tokenized_datasets.remove_columns(['text'])
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

train_dataset = tokenized_datasets['train']
num_samples = int(0.15 * len(train_dataset))
train_subset = train_dataset.select(range(num_samples))

# Create DataCollator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Fine-tune the teacher model
teacher_model.finetune(
    dataset=train_subset,
    batch_size=training_params["batch_size"],
    gradient_accumulation_steps=training_params["gradient_accumulation_steps"],
    data_collator=data_collator,
    lr=training_params["learning_rate"],
    betas=training_params["betas"],
    weight_decay=training_params["weight_decay"],
    eps=training_params["eps"],
    epochs=training_params["epochs"]
)

teacher_model.save_checkpoint('./checkpoints')
