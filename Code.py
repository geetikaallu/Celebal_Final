# Install necessary resources
import nltk
nltk.download('stopwords')

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import homogeneity_score,completeness_score,normalized_mutual_info_score
from sklearn.manifold import TSNE

# Step 1: Load Dataset
print("Loading dataset...")
newsgroups=fetch_20newsgroups(remove=('headers','footers','quotes'))
documents=newsgroups.data
true_labels=newsgroups.target

# Step 2: TF-IDF Vectorization for K-Means
print("Vectorizing documents using TF-IDF...")
tfidf_vectorizer=TfidfVectorizer(stop_words='english',max_df=0.5,min_df=2)
tfidf=tfidf_vectorizer.fit_transform(documents)

# Step 3: Apply K-Means Clustering
print("Applying K-Means clustering...")
num_clusters=20
kmeans_model=KMeans(n_clusters=num_clusters,random_state=42)
kmeans_model.fit(tfidf)

# Step 4: Display Top Terms per KMeans Cluster
print("\nTop terms per KMeans cluster:")
terms=tfidf_vectorizer.get_feature_names_out()
for i in range(num_clusters):
    top_indices=kmeans_model.cluster_centers_[i].argsort()[-10:][::-1]
    top_terms=[terms[ind] for ind in top_indices]
    print(f"Cluster {i}:{top_terms}")

# Step 5: Evaluate Clustering
print("\nEvaluation Metrics for K-Means:")
print("Homogeneity Score:",homogeneity_score(true_labels,kmeans_model.labels_))
print("Completeness Score:",completeness_score(true_labels,kmeans_model.labels_))
print("NMI Score:",normalized_mutual_info_score(true_labels,kmeans_model.labels_))

# Step 6: Count Vectorization for LDA
print("\nVectorizing documents using CountVectorizer for LDA...")
count_vectorizer=CountVectorizer(stop_words='english',max_df=0.9,min_df=2)
doc_term_matrix=count_vectorizer.fit_transform(documents)

# Step 7: Apply Latent Dirichlet Allocation (LDA)
print("Applying Latent Dirichlet Allocation (LDA)...")
lda_model=LatentDirichletAllocation(n_components=20,random_state=42)
lda_model.fit(doc_term_matrix)

# Step 8: Display Top Terms per LDA Topic
print("\nTop terms per LDA topic:")
terms=count_vectorizer.get_feature_names_out()
for idx,topic in enumerate(lda_model.components_):
    top_terms=[terms[i] for i in topic.argsort()[-10:]]
    print(f"Topic {idx}:{top_terms}")

# Step 9: Visualize KMeans Clusters with t-SNE
print("\nVisualizing KMeans clusters using t-SNE (this may take a few minutes)...")
tsne_model=TSNE(n_components=2,perplexity=50,n_iter=300,random_state=42)
tsne_result=tsne_model.fit_transform(tfidf.toarray())

# Plot the t-SNE results
plt.figure(figsize=(12,6))
scatter=plt.scatter(tsne_result[:,0],tsne_result[:,1],c=kmeans_model.labels_,cmap='tab20',s=10)
plt.title("KMeans Clustering (t-SNE Visualization)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.colorbar(scatter,label='Cluster ID')
plt.tight_layout()
plt.show()
