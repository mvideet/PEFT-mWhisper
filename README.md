# Parameter-Efficient Fine-Tuning for **mWhisper-Flamingo**
*Multilingual Audio-Visual Speech Recognition — minus the billion-parameter price-tag.*

---

## ✨ Project Snapshot

|                           | **Before (full FT)** | **After (best PEFT)** |
|---------------------------|----------------------|-----------------------|
| **Trainable parameters**  | ≈ 3 B               | **≈ 3.8 M** (LoRA-16 QV) |
| **Clean-set WER**         | 1.19 %               | **1.26 %** |
| **0 SNR WER**             | 17.65 %              | **21.35 %** |

<sub>LoRA rank-16, tuning only the **Query & Value** projections, recovers **> 85 %** of noisy-speech performance while training **≈ 700 ×** fewer weights.</sub>

---

## 🗺️ Why this project?

`mWhisper-Flamingo` fuses **Whisper** (audio) with **AV-HuBERT** (video) to achieve state-of-the-art multilingual AVSR, but fine-tuning the 3 B-parameter model is out of reach for many researchers.

We explore **Parameter-Efficient Fine-Tuning (PEFT)** strategies that

* **cut** GPU memory / time costs *dramatically*, and  
* **preserve—or even improve—** Word-Error-Rate (WER) in both clean **and** noisy conditions.

---

## 🔬 What we did

| PEFT technique | Key idea | Variants evaluated | Take-aways |
|----------------|----------|--------------------|------------|
| **Linear Adapters** | Bottleneck MLP blocks after each attention/FFN layer | 5 ranks (64 → 1024); with/without near-identity init; ± LayerNorm-tuning | Higher ranks help, but still trail LoRA in noise robustness |
| **LoRA** | Low-rank Δ*W* inserted **inside** Q/K/V/O projections | Ranks 4, 16, 32; full QKV vs. QV-only | **LoRA-16 QV** is the sweet-spot—minimal params, near-baseline WER |
| **Soft Prompting (+ visual adapter)** | Learnable prefix tokens on decoder time-axis | Prompt lengths 8 & 64 | Did *not* beat LoRA; extra prompts can add noise |

---

## 📊 Experimental setup

* **Datasets:** MuAViC (9 languages); clean & 0 SNR noisy splits  
* **Metric:** Word-Error-Rate (WER)  
* **Baselines:** Whisper-en-small (zero-shot & full FT)  
* **Hardware:** 4 × A100 80 GB̄ (but PEFT runs fit on a single 24 GB card)

---
