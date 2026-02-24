# Predatory-Clause-Detector
## 🚀 Phase 1: Data Collection & Research Strategy

**Context & Reference Paper:**
This project's methodology—specifically the need for expert-annotated data to detect legal anomalies—is directly inspired by foundational research in Legal NLP.
> **📄 Key Reference:** *CLAUDETTE: An automated detector of potentially unfair clauses in online terms of service* — Marco Lippi, Przemysław Pałka, et al. (2019), Artificial Intelligence and Law (Springer). 
> **🔗 Link:** [Read the Paper on arXiv](https://arxiv.org/abs/1805.01217) | [Springer Official](https://link.springer.com/article/10.1007/s10506-019-09243-2)

**The Challenge (As per the Research):**
The referenced research highlights a critical bottleneck in Legal AI: standard web scraping is highly ineffective for ToS analysis. Determining whether a legal clause is "Fair", "Potentially Unfair", or "Predatory" (violating consumer rights or privacy laws like GDPR/CCPA) requires strict evaluation by legal experts. Training a model on raw, unverified internet data leads to inaccurate and biased legal predictions.

**Our Implementation & Dataset Choice:**
To strictly adhere to the paper's requirement for **"expert-annotated ground truth,"** we avoided raw data scraping and instead integrated a scientifically validated dataset:

* **SOTA Dataset:** We are utilizing the `joelniklaus/online_terms_of_service` dataset sourced via Hugging Face.
* **Alignment with the Paper:** Exactly as the methodology demands, this dataset consists of real-world ToS contracts that have been segmented into individual clauses and **manually labeled by legal professionals**. It classifies clauses into specific unfairness categories (e.g., unilateral changes, forced arbitration, limitation of liability).
* **Reproducibility:** The data is dynamically fetched and converted to a local CSV using our `data_generator.py` script. To maintain Git best practices and keep the repository lightweight, the generated `.csv` file is strictly ignored via `.gitignore`.