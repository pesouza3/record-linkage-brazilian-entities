# Record Linkage for Brazilian Corporate Data Integration: A Comparative Machine Learning Approach

> **MBA Thesis in Data Science & Analytics — USP/Esalq**  
> **Authors:** Pedro Soares Souza & Dr. Edilson José Rodrigues  

An automated, memory-efficient Python pipeline that merges and deduplicates noisy corporate datasets (Pessoas Jurídicas) in Brazilian tax ecosystems without shared primary keys (CNPJ).

---

## 📊 Dataset & Public Source
The primary corporate records used in this research were extracted from public Brazilian business registries:
* **Primary Source:** [Receita Federal do Brasil — Dados Públicos de CNPJ](https://dados.gov.br/)
* **Public Repository:** [Brasil.IO — Dataset Sócios das Empresas Brasileiras](https://brasil.io/dataset/socios-brasil/)

> **Note:** The dataset used in the pipeline contains public records including corporate names (`razao_social`), Brazilian federative units (`uf`), and tax IDs (`cnpj`).

## 📌 Executive Summary
In Mergers & Acquisitions (M&A) and enterprise data governance, integrating relational databases without universal primary keys leads to severe data fragmentation. Traditional static thresholds (e.g., Jaro-Winkler > 0.85) fail due to corporate suffix variations (`LTDA`, `S.A.`), syntactic inversions, and typographical errors, yielding high false-positive rates.

This research implements an end-to-end Machine Learning pipeline across **100,000 Brazilian enterprise records**, combining custom text normalization, 4-character + UF memory blocking, string similarity feature extraction, and supervised classification.

### 🏆 Key Results
| Classifier Model | Precision | Recall (Sensibilidade) | F₁-Score |
| :--- | :---: | :---: | :---: |
| **Baseline (Jaro-Winkler > 0.85)** | 16.39% | **99.99%** | 0.2816 |
| **Logistic Regression** | 58.33% | 98.71% | 0.7333 |
| **XGBoost** | 54.84% | **99.94%** | 0.7082 |
| 🥇 **Random Forest (Best Model)** | **62.61%** | **97.99%** | **0.7641** |

* **Key Finding:** Random Forest achieved the most resilient decision boundary, overcoming class imbalance and reducing false positives while maintaining a **97.99% recall rate**.
* **Feature Importance:** Jaro-Winkler similarity carries higher predictive weight than Levenshtein distance due to positional prefix resilience in Brazilian company naming structures.

---

## ⚙️ Technical Architecture & Pipeline

```text
1. Data Extraction (100k Brazilian Public Corporate Records)
   └── Normalization via Regex & Unidecode (Removing legal suffixes & accents)
2. Synthetic Noise Injection (Gold Standard Control)
   └── 70% Intact | 10% Typos | 10% Tail Omissions | 10% Syntactic Inversions
3. Memory-Safe Blocking (RecordLinkage)
   └── 4-Character Prefix + UF (Prevents O(N²) product space explosion)
4. Feature Engineering
   └── Pairwise computation of Jaro-Winkler & Levenshtein metrics
5. Streaming & SSD Chunking
   └── PyArrow Parquet incremental dumping (Anti-RAM crash architecture)
6. Supervised ML Training & Evaluation
   └── Random Forest vs. XGBoost vs. Logistic Regression vs. Static Baseline
