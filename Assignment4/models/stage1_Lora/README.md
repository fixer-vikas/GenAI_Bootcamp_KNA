---
base_model: HuggingFaceTB/SmolLM2-135M
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:HuggingFaceTB/SmolLM2-135M
- lora
- transformers
---

# Stage 1 LoRA Adapter

This adapter is the first stage of a healthcare FAQ assistant fine-tuning pipeline. It was created using LoRA on a small base model and adapted to healthcare-domain text so the model becomes more familiar with medical terminology and question style.

## Model Details

- **Base model:** HuggingFaceTB/SmolLM2-135M
- **Adapter type:** LoRA
- **Purpose:** domain adaptation for healthcare-related text generation and question answering
- **Framework:** PEFT and Transformers

## Intended Use

This adapter is intended for research and educational use in a healthcare FAQ assistant workflow. It can be used as a starting point for further instruction tuning or for generating draft responses that should be reviewed carefully.

## Limitations

- It is not a medical expert and should not be treated as a substitute for professional advice.
- It may produce incomplete, outdated, or incorrect information.
- Critical medical decisions should be verified with qualified healthcare professionals.

## Training Data

The adapter was trained on healthcare-related domain text and FAQ-style examples that were prepared for the assignment project.

## Training Setup

- Parameter-efficient fine-tuning with LoRA
- Low-rank adaptation applied to selected transformer layers
- No secrets, API keys, or sensitive credentials are included in this repository

## Example Use

This adapter can be combined with later instruction-tuning and preference-tuning stages to build a more helpful healthcare assistant.

### Framework versions

- PEFT 0.19.1
