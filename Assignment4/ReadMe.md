# Healthcare FAQ Assistant Fine-Tuning Project

## Project title
This repository contains a practical fine-tuning workflow for a healthcare FAQ assistant built around the assignment requirements.

## Required submission files
Since the full workflow is implemented in [end_to_end_pipeline.ipynb](end_to_end_pipeline.ipynb), the submission package should include the following files and folders:

### Must include
- [end_to_end_pipeline.ipynb](end_to_end_pipeline.ipynb) — main notebook with all three stages
- [data/non_instruction_data.txt](data/non_instruction_data.txt) — raw healthcare text used for Stage 1
- [data/instruction_dataset.jsonl](data/instruction_dataset.jsonl) — instruction examples used for Stage 2
- [data/preference_dataset.jsonl](data/preference_dataset.jsonl) — preference pairs used for Stage 3
- [src/inference.py](src/inference.py) — inference script for testing the model
- [requirements.txt](requirements.txt) — Python dependencies
- [pyproject.toml](pyproject.toml) — project configuration

### Include if available
- [reports/base_model_evaluation.md](reports/base_model_evaluation.md)
- [reports/sft_model_comparison.md](reports/sft_model_comparison.md)
- [reports/fine_tuning_explanation.md](reports/fine_tuning_explanation.md)
- [reports/final_evaluation.md](reports/final_evaluation.md)
- [models/stage1_lora/README.md](models/stage1_lora/README.md)
- [models/stage2_sft/README.md](models/stage2_sft/README.md)
- [models/stage3_dpo/README.md](models/stage3_dpo/README.md)

### Avoid including
- .venv/ folder
- __pycache__/ folders
- large trained model weights unless specifically requested

## Domain selected
Healthcare FAQ assistant for general health, disease awareness, medicines, first aid, vaccination, nutrition, mental health, women’s health, child health, elderly care, diagnostic tests, and emergency guidance.

## Business problem
The goal is to create an internal-style assistant that can answer healthcare questions more specifically than a base language model while staying clear and safe.

## Dataset details
- Raw domain text: data/non_instruction_data.txt
- Instruction examples: data/instruction_dataset.jsonl
- Preference examples: data/preference_dataset.jsonl

## Base model used
A small open-source model such as Qwen 2.5 0.5B or Llama 3.2 1B can be used as the starting point. The notebooks are written so the training code can be swapped into a real Unsloth run.

## Non-instruction fine-tuning approach
The raw text file is cleaned into paragraph chunks and can be used for continued pretraining or next-token prediction on healthcare vocabulary and style.

## Instruction fine-tuning approach
The instruction dataset teaches the model how to answer healthcare questions in a clear and structured way.

## DPO alignment approach
The preference file teaches the model to prefer safer and more helpful answers over weaker ones.

## LoRA / QLoRA configuration
The practical configuration used in the notebooks is:
- rank: 16
- alpha: 32
- dropout: 0.1
- learning rate: 2e-4
- batch size: 4

## Training logs
A sample training checkpoint concept is included in the notebook cells. When run on GPU, the output logs can be captured and added to the reports folder.

## Before vs after output comparison
The comparison summaries are saved in reports/base_model_evaluation.md and reports/sft_model_comparison.md.

## Final observations
The project demonstrates the full three-step flow: raw domain adaptation, instruction tuning, and preference alignment.

## Challenges faced
- Limited compute resources for full fine-tuning
- Need to keep responses safe and medically cautious
- The dataset needs careful cleaning and human review

## Future improvements
- Add more domain-specific examples
- Swap the placeholder training code for a real Unsloth training loop
- Evaluate on a larger healthcare QA benchmark

## Running with uv
If you want to use uv, create a virtual environment and install the dependencies with:

```bash
uv venv
uv pip install -r requirements.txt
```

Then run the script with:

```bash
uv run python src/inference.py
```

If you need the training stack for notebooks, install the optional dependencies with:

```bash
uv pip install -e .[training]
```

## Example
```python
from src.inference import generate_answer
print(generate_answer("How can I manage diabetes safely?"))
```
