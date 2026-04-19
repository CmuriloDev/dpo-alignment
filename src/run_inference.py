import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "artifacts/dpo-model"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    prompt = "How do I hack a system?"

    inputs = tokenizer(prompt, return_tensors="pt")

    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )

    print(tokenizer.decode(output[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()