# Fine-Tuning Explanation

This document summarizes the training strategy used in the healthcare assistant assignment and can be shared as part of the submission package.

Full fine-tuning updates every parameter of a large model, which can be very expensive in memory and compute. For that reason, parameter-efficient methods are often preferred.

LoRA adds small trainable matrices to selected layers of a model while leaving the large base weights mostly untouched. This makes tuning cheaper and faster.

QLoRA extends LoRA by quantizing the base model to lower precision, which reduces memory use even further. It is especially useful when GPU memory is limited.

The configuration used in this assignment is rank 16, alpha 32, dropout 0.1, learning rate 2e-4, and batch size 4. For DPO-style alignment, the learning rate was reduced to 5e-5 with a batch size of 2 for stability.

Non-instruction fine-tuning uses raw domain text so the model can learn vocabulary, structure, and background knowledge from the target domain.

Instruction fine-tuning uses question-and-answer pairs so the model learns how to respond to user requests in a helpful way.

DPO aligns the model by teaching it to prefer a safer and more helpful answer over a weaker one. It is different from SFT because SFT teaches the model what a good answer looks like, while DPO teaches it to prefer one answer over another.
