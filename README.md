# Domain-Specific-Whisper-Fine-tuning-for-Law-Health-Domain

This repository contains the implementation and experiments of our paper:  
**"Multilingual Domain Adaptation for Speech Recognition Using LLMs"**  
Accepted at TSD 2025.

## 📝 Abstract

We propose a scalable pipeline to fine-tune Whisper-large-v3 for domain-specific multilingual ASR tasks. By leveraging the multilingual LLM **Aya-23-8B**, we automatically classify **Common Voice 17.0** transcriptions into **Law**, **Healthcare**, and **Other** domains across 22 languages. These domain labels enable parameter-efficient fine-tuning (using **LoRA**) of Whisper, achieving consistent **up to 15% relative WER reductions** in in-domain settings.

## 🔍 Key Contributions

- 📚 **Automatic Domain Classification**  
  Using Aya-23-8B with few-shot prompting, we label transcriptions in 22 languages with high quality and low manual cost.

- 🎯 **LoRA-based Fine-Tuning**  
  We fine-tune Whisper-large-v3 on domain-specific data using Low-Rank Adaptation for efficiency.

- 📊 **Data Volume Analysis**  
  We identify a clear threshold: **≥800 in-domain utterances** are needed for consistent WER improvement.

- 🌍 **Multilingual & Monolingual Comparisons**  
  Evaluation shows multilingual training is generally effective, while monolingual fine-tuning helps in low-resource or complex languages.

## 🗂️ Dataset

We use the [Common Voice 17.0](https://commonvoice.mozilla.org) dataset. The classification covers 22 languages (excluding Korean due to data scarcity):

