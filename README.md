# Web Crawling Strategy for High-Quality Training Data

*A ranking-based approach for prioritising web pages in large-scale crawling pipelines, designed for AI training data collection.*

---

## 🚀 Overview

Modern AI systems depend on high-quality web data, but crawling everything is inefficient and noisy.  
This project implements a **feature-based ranking model** that selects the most valuable pages to crawl first.

### Given:
- A directed web graph  
- Precomputed PageRank scores  
- Crawl permissions  

### The system:
Ranks pages and returns the **top-k highest-value URLs**.

---

## 🧠 Key Idea

Instead of relying solely on PageRank, this project combines multiple signals:

- **PageRank** (global authority)  
- **Inlink authority** (quality of incoming links)  
- **Content-quality prior**  
- **Domain diversity**  
- **Out-degree** (exploration potential)  
- **Low-quality page detection**  
- **Low-trust domain penalties**  

Feature weights are **learned using Ridge regression** (weak supervision), rather than manually tuned.

A **diversity-aware reranking step** ensures the final results are not dominated by a single domain.

---

## 📊 Results

Compared to a PageRank-only baseline, the model achieves:

- ↑ Higher content quality in top-k results  
- ↑ Increased domain diversity  
- ↓ Fewer low-quality pages  

### Additional comparisons:
- Uniform-weight baseline  
- Random-weight baseline  
- Ablation study (feature importance)  

---

## 🧪 Project Structure

```text
.
├── cloud.py              # Main ranking + learning pipeline
├── visualize.py          # Plots and evaluation visuals
├── web_graph_data.py     # Synthetic dataset + graph definition
└── report.pdf            # Full report
```

---

## ⚙️ Setup

### Install dependencies:
```bash
pip install networkx matplotlib pandas scikit-learn
```

### Run the pipeline:
```bash
python cloud.py
python visualize.py
```

---

## 📈 Outputs

Running the code generates:

- Ranked top-k URLs  
- Baseline vs learned comparison plots  
- Category distribution charts  
- Rank movement visualisation  
- Learned feature weights  
- CSV export of results  

---

## ⚠️ Limitations

- Uses a synthetic dataset (100 nodes)  
- Relies on weak supervision (pseudo-labels)  
- No downstream evaluation on model training  

---

## 🔮 Future Work

- Scale to larger or real-world datasets (e.g. Common Crawl)  
- Use learning-to-rank methods  
- Incorporate semantic content signals (NLP-based features)  
