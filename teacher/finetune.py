import yaml
import argparse
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description="Fine-tune a teacher model.")
parser.add_argument("--config", type=str, default='teacher/training-config.yaml', help="Config Path")
parser.add_argument("--model_name", type=str, required=True, help="The name of the model to fine-tune.")
parser.add_argument("--dataset", type=str, required=True, help="The name of the dataset to use for fine-tuning.")
parser.add_argument("--batch_size", type=int, default=16, help="Batch Size")
args = parser.parse_args()

# Model Name
TEACHER_PATH = {
    'Qwen': 'Qwen/Qwen2.5-7B',
    'mT5': 'google/mt5-xxl',
    'Bloom': 'bigscience/bloomz-7b1',
}

STUDENT_PATH = {
    'Qwen': 'Qwen/Qwen2.5-0.5B',
    'mT5': 'google/mt5-small',
    'Bloom': 'bigscience/bloomz-560m',
}
# Loading Model
tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH[args.model])

# Loading Dataset, DataLoader
if args.dataset == 'imdb':
    from dataset.imdb import create_dataloaders
elif args.dataset == 'uit-vsfc':
    from dataset.uit_vsfc import create_dataloaders

# Check dataloader
train_dataloader, val_dataloader, test_dataloader = create_dataloaders(tokenizer, args.batch_size)
sample = next(iter(train_dataloader))
print(tokenizer.batch_decode(sample.input_ids, skip_special_tokens=True))

# Fine-tuning
with open(args.config, encoding="utf8") as conf:
    config = yaml.safe_load(conf)

training_config = config["training_config"]
training_params = config["training"]

# Ensure numeric parameters are correctly parsed as floats
training_params["learning_rate"] = float(training_params["learning_rate"])
training_params["weight_decay"] = float(training_params["weight_decay"])
training_params["eps"] = float(training_params["eps"])

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


def main(model_name, dataset_name):
    # Load training configurations from a YAML file
    with open("teacher/training-config.yaml", encoding="utf8") as conf:
        config = yaml.safe_load(conf)

    training_config = config["training_config"]
    training_params = config["training"]

    # Ensure numeric parameters are correctly parsed as floats
    training_params["learning_rate"] = float(training_params["learning_rate"])
    training_params["weight_decay"] = float(training_params["weight_decay"])
    training_params["eps"] = float(training_params["eps"])

    # Initialize the TeacherModel
    teacher_model = TeacherModel(model_name, training_config)

    # Load the dataset
    dataset = load_dataset(dataset_name, 'sentiment')
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