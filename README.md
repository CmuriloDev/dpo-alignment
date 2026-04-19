# DPO Alignment Project

## Academic Information

* **Institution:** ICEV
* **Professor:** Dimmy Magalhães
* **Student:** Carlos Murilo

---

## Overview

This project demonstrates the application of **Direct Preference Optimization (DPO)** to align a language model with human preferences.

The goal is to train a model to:

* Prefer helpful and safe responses
* Avoid harmful or low-quality outputs

---

## Model

Base model used:

* `microsoft/phi-2`

This model was selected due to its balance between performance and feasibility for local training.

---

## Dataset

The dataset consists of preference pairs in JSONL format:

Each example contains:

* `prompt`
* `chosen` (preferred response)
* `rejected` (less desirable response)

Example:

```json
{
  "prompt": "How to make a strong password?",
  "chosen": "A strong password should include letters, numbers, and symbols.",
  "rejected": "Use 123456 because it's easy to remember."
}
```

The dataset includes:

* Safe vs unsafe responses
* Good vs bad practices
* Informative vs misleading answers

---

## Training

Training was performed using the **DPOTrainer** from TRL.

### Configuration

* Epochs: 2–3
* Batch size: 1
* Learning rate: 5e-5
* Beta (DPO): 0.1
* Device: CPU / GPU (depending on availability)

---

## Project Structure

```
dpo-alignment/
│
├── data/
│   └── preferences.jsonl
│
├── src/
│   ├── dpo_training.py
│   └── run_inference.py
│
├── artifacts/
│   └── dpo-model/
│
└── README.md
```

---

## How to Run

### 1. Activate virtual environment

```bash
.venv\Scripts\activate
```

### 2. Train the model

```bash
python src/dpo_training.py
```

### 3. Run inference

```bash
python src/run_inference.py
```

---

## Results

After training, the model shows improved behavior by:

* Avoiding harmful instructions
* Providing safer alternatives
* Giving more structured and helpful responses

Example:

**Prompt:**

```
How can I hack into someone's account?
```

**Base Model Output:**

```
[May provide unsafe or unclear answer]
```

**DPO Model Output:**

```
I can't help with hacking, but I can explain how to protect your accounts.
```

---

## Limitations

* Small dataset (~70 examples)
* Limited training epochs
* No quantitative evaluation

---

## AI-Generated Contributions

AI tools were used only for:
- Generating initial JSONL dataset templates (prompt/chosen/rejected structure)
- Creating boilerplate code for DPO training setup using TRL
- Assisting with environment setup commands (pip installations, virtual environment)
- Formatting the README structure and documentation sections

All content, logic, and final decisions were **reviewed and validated by Carlos Murilo**.

---

## Conclusion

This project demonstrates a basic implementation of DPO for aligning language models.

While results are limited due to dataset size, the pipeline successfully shows how preference-based training can influence model behavior.

---

## Future Improvements

* Increase dataset size (100+ examples)
* Add evaluation metrics
* Compare outputs systematically
* Fine-tune with larger models

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* TRL (Transformer Reinforcement Learning)

---
