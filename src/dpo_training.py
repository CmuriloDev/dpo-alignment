import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer

MODEL_ID = "microsoft/phi-2"  
DATA_PATH = "data/preferences.jsonl"
OUTPUT_DIR = "artifacts/dpo-model"

def main():
    dataset = load_dataset("json", data_files=DATA_PATH)["train"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        num_train_epochs=3,
        learning_rate=5e-5,
        logging_steps=1,
        optim="adamw_torch",  
        report_to="none",
        fp16=True
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        beta=0.1,
        max_length=256,
        max_prompt_length=128
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()