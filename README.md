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

## 🧠 Models

- **ASR Model:** [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)  
- **LLM Classifier:** [CohereForAI/aya-23-8B](https://huggingface.co/CohereForAI/aya-23-8B)

## ⚙️ Training Details

- **Framework:** PyTorch with Huggingface Transformers  
- **Fine-tuning Method:** LoRA (8-bit quantized)  
- **Hardware:** Single NVIDIA RTX 3060  
- **Training Time:** ~7-8 hours per domain

### LoRA Config

- Learning rate: `1e-5`  
- Batch size: `16`  
- Warm-up steps: `50`  
- Scheduler: Linear

## 📉 Sample Evaluation Results (WER Reductions)

To demonstrate the effectiveness of domain-specific fine-tuning, we highlight two examples from different domains:

| Language     | Domain     | Baseline WER | Fine-Tuned WER | Fine-Tuning     | Absolute ΔWER | Relative Δ (%) 
|--------------|------------|--------------|----------------|------------------|---------------|----------------|
| Vietnamese   | Law        | 21.43        | 14.29          | Multilingual     | 7.14          | 33.3%          |
| Japanese     | Healthcare | 89.47        | 84.21          | Monolingual      | 5.26          | 5.9%           |

> These results show that even modest domain supervision via LLM-labeled data can yield substantial ASR performance improvements—especially when in-domain sample count exceeds ~800.


## 📚 Citation

```bibtex
@inproceedings{ulu2025multilingual,
  title={Multilingual Domain Adaptation for Speech Recognition Using LLMs},
  author={Ulu, Elif Nehir and Derya, Ece and Tümer, Duygu and Demirel, Berkan and Karamanlıoğlu, Alper},
  booktitle={Text, Speech and Dialogue (TSD)},
  year={2025}
}


