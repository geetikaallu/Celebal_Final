# 📚 Topic Modeling and Document Clustering using KMeans and LDA

This project performs topic modeling and document clustering using the **20 Newsgroups dataset**. We apply **K-Means** to group documents and **Latent Dirichlet Allocation (LDA)** to uncover hidden topics. The project also includes visualization and evaluation.

---

## 🧠 Objective

- Group similar documents using **KMeans**
- Extract hidden topics using **LDA**
- Visualize clusters using **t-SNE**
- Evaluate with Homogeneity, Completeness, and NMI

---

## 📦 Dataset

- **Name**: 20 Newsgroups  
- **Source**: [UCI ML Repository](http://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups)  
- **Content**: ~18,000 documents across 20 categories

---

## 🛠️ Techniques Used

- `TfidfVectorizer` for KMeans
- `CountVectorizer` for LDA
- `KMeans` Clustering
- `LatentDirichletAllocation`
- `TSNE` for 2D projection
- Clustering Metrics: Homogeneity, Completeness, NMI

---

## ▶️ How to Run

### 1. Install Required Libraries

```bash
pip install numpy matplotlib scikit-learn nltk
```

Then run this once in Python:

```python
import nltk
nltk.download('stopwords')
```

### 2. Run the Script

```bash
python lda_kmeans_clustering.py
```

## 📊 Sample Output

### 🔹 KMeans Top Terms per Cluster

```
Cluster 0: ['god', 'jesus', 'bible', 'faith', 'church', 'christians', 'lord', 'heaven', 'believe', 'christ']
Cluster 1: ['car', 'engine', 'oil', 'dealer', 'bmw', 'ford', 'drive', 'cars', 'speed', 'miles']
Cluster 2: ['game', 'team', 'season', 'hockey', 'play', 'games', 'coach', 'win', 'players', 'league']
...
```

### 🔹 LDA Top Terms per Topic

```
Topic 0: ['windows', 'file', 'files', 'program', 'software', 'directory', 'run', 'system', 'data', 'disk']
Topic 1: ['god', 'jesus', 'bible', 'faith', 'christians', 'religion', 'heaven', 'lord', 'christ', 'believe']
Topic 2: ['car', 'engine', 'drive', 'ford', 'oil', 'miles', 'bmw', 'dealer', 'car', 'speed']
...
```

### 🔹 Evaluation Metrics

```
Homogeneity Score:     0.31
Completeness Score:    0.29
Normalized Mutual Info:0.30
```

---
