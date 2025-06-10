# Unsupervised Topic Modelling

This project provides a flexible Python pipeline for discovering and analyzing topics from a collection of text messages.

The core of the project is the TopicAnalyzer, a high-level class that orchestrates the entire analysis process, from initial text preprocessing to the final generation of topic summaries.

### How It Works
- **Keyphrase Extraction**: Salient phrases and keywords are extracted from each message. You can choose from various methods, from statistical ones like TF-IDF to transformer-based ones like KeyBERT.
- **Semantic Embedding**: Pre-processed messages are converted into high-dimensional numerical vectors that capture the semantic meaning of the text.
- **Clustering**: The embeddings are grouped into clusters using an unsupervised algorithm. Each cluster represents a potential topic.
- **Topic Summarization**: The most common key phrases associated with each cluster is determined to understand common themes within the cluster and to help assign human-readable topic names.

### Available Techniques
| Component | Available Methods |
|-----------|-------------------|
| Keyphrase Extraction | bert, tfidf, rake, yake, textrank, pos, keybert, hybrid_pos_tfidf, hybrid_pos_keybert |
| Embedding Generation | sentence_transformer, bert_cls, bert_mean, tfidf, combined |
| Clustering | dbscan, kmeans, agglomerative |

### Example Usage
```python
analyzer = TopicAnalyzer()

analysis_results = analyzer.analyze_messages(
    messages=sample_messages,
    keyphrase_method='keybert',
    embedding_method='sentence_transformer',
    clustering_method='kmeans',
    clustering_params={'adaptive': True, 'max_clusters': 10}
)

