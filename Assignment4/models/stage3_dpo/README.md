---
base_model: HuggingFaceTB/SmolLM2-135M
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:HuggingFaceTB/SmolLM2-135M
- lora
- transformers
---

# Stage 3 DPO Adapter

This adapter is the final alignment stage of the healthcare FAQ assistant pipeline. After domain adaptation and supervised instruction tuning, it was refined with preference-based training so it prefers safer and more helpful responses.

## Model Details

- **Base model:** HuggingFaceTB/SmolLM2-135M
- **Adapter type:** DPO-style preference alignment with LoRA
- **Purpose:** improve response quality and safety for healthcare-themed interactions
- **Framework:** PEFT and Transformers

## Intended Use

This adapter is intended for experimental use in a healthcare assistant workflow where preference-based alignment is being studied. It can help create more polished answer drafts, but it still needs careful human review.

## Limitations

- It may still prefer responses that are overly cautious or not fully accurate.
- It should not be used for medical diagnosis or emergency decision-making.
- All critical guidance should be confirmed with trusted sources or professionals.

## Training Data

The adapter was trained using preference pairs that compare a better healthcare response against a weaker one, helping the model learn a more helpful ranking.

## Training Setup

- Preference-based optimization using DPO-style training
- Parameter-efficient adaptation with LoRA
- No secrets, keys, or credentials are included in this repository

## Example Use

This final stage is useful when the goal is to improve the model’s relative preference for safer and more helpful healthcare answers.

### Framework versions

- PEFT 0.19.1
