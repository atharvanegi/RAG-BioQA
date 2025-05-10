# RAG-BioQA: Retrieval-Augmented Generation for Long-Form Biomedical Question Answering

RAG-BioQA is a domain-specific, retrieval-augmented generation (RAG) framework designed to generate long-form, evidence-based answers to complex biomedical questions. It integrates dense BioBERT embeddings, efficient FAISS-based retrieval, multiple re-ranking strategies, and a fine-tuned T5 generative model to provide comprehensive responses tailored for biomedical research and clinical use cases.

## 📌 Features

- ✅ Uses BioBERT for domain-specific dense embeddings
- 🔍 Retrieves and re-ranks relevant QA contexts using FAISS, BM25, ColBERT, and MonoT5
- 🧠 Generates long-form answers via a fine-tuned FLAN-T5 model
- 🧪 Trained on PubMedQA, MedDialog, and MedQuAD datasets
- ⚙️ Lightweight fine-tuning using PEFT (LoRA)

## 📊 Results

| Model Variant           | BLEU-1 | ROUGE-1 | BERTScore | METEOR |
|------------------------|--------|---------|-----------|--------|
| Base T5 + FAISS        | 0.2065 | 0.2618  | 0.1132    | 0.1948 |
| Finetuned T5 + FAISS   | 0.2415 | 0.2918  | 0.2054    | 0.2264 |
| Finetuned T5 + BM25    | 0.2221 | 0.2714  | 0.1318    | 0.2054 |
| Finetuned T5 + ColBERT | 0.2218 | 0.2713  | 0.1364    | 0.2053 |
| Finetuned T5 + MonoT5  | 0.2172 | 0.2632  | 0.1277    | 0.2023 |

## 🏗️ System Architecture

### Diagram 1:
<img width="856" alt="Screenshot 2025-05-10 at 4 04 24 PM" src="https://github.com/user-attachments/assets/51f9af3d-4b2b-49fb-b6de-038a3c975fbc" />


### Diagram 2:
<img width="856" alt="Screenshot 2025-05-10 at 4 04 40 PM" src="https://github.com/user-attachments/assets/f7fc82f4-adfb-4ae9-a195-0bd90fd15c71" />



## 🧪 Experimental Setup

- **Dataset**: PubMedQA, MedDialog, MedQuAD (181k QA pairs)
- **Embeddings**: BioBERT-base-cased-v1.1 (768-dim)
- **Retriever**: FAISS (IndexFlatL2)
- **Re-rankers**: BM25, ColBERT, MonoT5
- **Generator**: FLAN-T5-base + LoRA (PEFT)
- **Hardware**: NVIDIA A100, 128GB GPU memory



## 👨‍💻 Authors

- Lovely Yeswanth Panchumarthi  
- Atharva Negi   
- Harsit Upadhya
- Sai Prasad Gudari  
*(All from Emory University)*

## Project Report
[RAG-BioQA Retrieval-Augmented Generation for Long-Form Biomedical Question Answering.pdf](https://github.com/user-attachments/files/20141214/RAG-BioQA.Retrieval-Augmented.Generation.for.Long-Form.Biomedical.Question.Answering.pdf)



