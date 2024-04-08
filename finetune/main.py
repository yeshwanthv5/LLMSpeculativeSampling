import os
import argparse
import train
from models import load_text_generation_model
from data import dataset_loader
from utils import make_folders

parser = argparse.ArgumentParser(description='parser for training decoder-only models')
parser.add_argument('--model_name', type=str, default='facebook/opt-125m', help='opt and bloomz series')
parser.add_argument('--dataset_name', type=str, default='dialogsum', help='scitldr or dialogsum')
parser.add_argument('--train_type', type=str, default='full_finetuning', help='full_finetuning or lora')
parser.add_argument('--max_input_length', type=int, default=512, help='number of input tokens for causal language modeling')
parser.add_argument('--max_output_length', type=int, default=64, help='number of new output tokens for generation')
parser.add_argument('--batch_size', type=int, default=4, help='batch size during training and generation')
parser.add_argument('--task', type=str, default='summarization', help='summarization or qa')

args = parser.parse_args()

make_folders("logs", "saved_models")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

train_type = args.train_type
task = args.task

phase = 'evaluate' # train or evaluate

model_name = args.model_name # "facebook/opt-125m" "gpt2"
dataset_name = args.dataset_name
max_input_length = args.max_input_length
max_output_length = args.max_output_length
batch_size = args.batch_size

early_exit=0.25
print("Early exit:", early_exit)

# train_type = "full_finetuning" # "full_finetuning"
model_path = f"saved_models/{model_name.replace('/', '_')}_{train_type}_{early_exit}"

model = load_text_generation_model(
    model_name, train_type,
    output_attentions=False,
)
train_loader, tokenizer = dataset_loader(
    dataset_name=dataset_name,
    split="train",
    tokenizer_name=model_name,
    model_name=model_name,
    max_input_length=max_input_length, 
    batch_size=batch_size,
    shuffle=True,
    keep_in_memory=True,
    print_info=False,
)

val_loader, _ = dataset_loader(
    dataset_name=dataset_name,
    split="validation",
    tokenizer_name=model_name,
    model_name=model_name,
    max_input_length=max_input_length, 
    batch_size=batch_size,
    shuffle=False,
    keep_in_memory=True,
    print_info=False,
)

test_loader, _ = dataset_loader(
    dataset_name=dataset_name,
    split="test",
    tokenizer_name=model_name,
    model_name=model_name,
    max_input_length=max_input_length, 
    batch_size=batch_size,
    shuffle=False,
    keep_in_memory=True,
    print_info=False,
)
my_trainer = train.Trainer(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    model=model,
    train_type=train_type,
    tokenizer=tokenizer,
    max_output_length=max_output_length,
    model_path=model_path,
)

print("Before Finetuning")
my_trainer.evaluate()

my_trainer.train(
    learning_rate=2e-5,
    num_epochs=5,
    log_dir=f"logs/{model_name.replace('/', '_')}_{train_type}_{early_exit}"
)
    
print("After Finetuning")
my_trainer.evaluate()

