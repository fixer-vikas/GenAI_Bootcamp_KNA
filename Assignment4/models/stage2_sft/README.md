---
base_model: HuggingFaceTB/SmolLM2-135M
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:HuggingFaceTB/SmolLM2-135M
- lora
- transformers
---

# Stage 2 SFT Adapter

This adapter is the second stage of the healthcare FAQ assistant pipeline. After domain adaptation, it was trained with supervised fine-tuning on instruction-response pairs so it can follow healthcare-related prompts more clearly.

## Model Details

- **Base model:** HuggingFaceTB/SmolLM2-135M
- **Adapter type:** SFT with LoRA
- **Purpose:** improve instruction-following for healthcare Q&A
- **Framework:** PEFT and Transformers

## Intended Use

This adapter is suitable for generating draft answers to healthcare questions in a research or educational setting. It is intended to support a chatbot-style workflow rather than replace expert advice.

## Limitations

- It may generate cautious or generic answers instead of precise medical guidance.
- It should not be used as a standalone medical diagnosis tool.
- Sensitive or urgent cases should be reviewed by professionals.

## Training Data

The adapter was trained on instruction-style examples built around healthcare FAQs, symptoms, safe guidance, and general wellness topics.

## Training Setup

- Supervised fine-tuning on paired instructions and responses
- Parameter-efficient adaptation with LoRA
- No secrets or credentials are stored in this project

## Example Use

This stage is useful for improving answer quality after the initial domain-adaptation step and before preference-based alignment.

### Framework versions

- PEFT 0.19.1
