import numpy as np
import re
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from lib.key_phrases import KeyPhrasesExtractor
from lib.embedding import EmbeddingGenerator
from lib.clustering import TopicClusterer

class TopicAnalyzer:
    """
    Main topic analysis module that combines key phrase extraction, embedding generation,
    and clustering to identify topics in user messages.
    
    This modular approach allows for flexible selection of methods at each step.
    """
    
    def __init__(self, 
                 keyphrase_model: str = 'bert-base-uncased',
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 keybert_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the TopicAnalyzer.
        
        Args:
            keyphrase_model: Model for BERT-based key phrase extraction
            embedding_model: Model for sentence embeddings
            keybert_model: Model for KeyBERT extraction
        """
        self.keyphrase_extractor = KeyPhrasesExtractor(
            bert_model=keyphrase_model,
            keybert_model=keybert_model
        )
        self.embedding_generator = EmbeddingGenerator(
            sentence_model=embedding_model,
            bert_model=keyphrase_model
        )
        self.clusterer = TopicClusterer()
        
        # Store last analysis results
        self.last_results = {}
    
    def preprocess_messages(self, messages: List[str]) -> List[str]:
        """
        Clean and preprocess messages.
        
        Args:
            messages: List of user messages
            
        Returns:
            List of cleaned messages
        """
        cleaned_messages = []
        for msg in messages:
            # Remove excessive whitespace and newlines
            cleaned = re.sub(r'\s+', ' ', msg.strip())
            # Remove very short messages (less than 3 words)
            if len(cleaned.split()) >= 3:
                cleaned_messages.append(cleaned)
        return cleaned_messages
    
    def generate_topic_summaries(self, 
                               messages: List[str], 
                               key_phrases: List[List[str]], 
                               cluster_labels: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """
        Generate topic summaries with associated key phrases.
        
        Args:
            messages: Original messages
            key_phrases: Key phrases for each message
            cluster_labels: Cluster assignments
            
        Returns:
            Dictionary mapping cluster IDs to topic information
        """
        topics = {}
        
        unique_labels = sorted(list(set(cluster_labels)))
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # Skip noise points
                continue
                
            # Get messages and phrases for this cluster
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_messages = [messages[i] for i in cluster_indices]
            cluster_phrases = [key_phrases[i] for i in cluster_indices]
            
            # Flatten and count phrase frequencies
            all_phrases = [phrase for phrase_list in cluster_phrases for phrase in phrase_list]
            phrase_counts = Counter(all_phrases)
            
            # Get top phrases for this topic
            top_phrases = [phrase for phrase, count in phrase_counts.most_common(10)]
            
            # Generate topic name from most common phrases
            topic_name = f"Topic_{cluster_id}: {', '.join(top_phrases[:3])}"
            
            topics[int(cluster_id)] = {
                'topic_name': topic_name,
                'key_phrases': top_phrases,
                'message_count': len(cluster_messages),
                'sample_messages': cluster_messages[:3],  # First 3 messages as examples
                'all_messages': cluster_messages,
                'phrase_frequencies': dict(phrase_counts.most_common(10)),
                'message_indices': cluster_indices.tolist()
            }
        
        return topics
    
    def analyze_messages(self, 
                        messages: List[str],
                        keyphrase_method: str = 'keybert',
                        embedding_method: str = 'sentence_transformer',
                        clustering_method: str = 'dbscan',
                        keyphrase_top_k: int = 5,
                        clustering_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main method to analyze messages and extract topics.
        
        Args:
            messages: List of user messages
            keyphrase_method: Key phrase extraction method
            embedding_method: Embedding generation method
            clustering_method: Clustering method
            keyphrase_top_k: Number of key phrases to extract per message
            clustering_params: Additional parameters for clustering
            
        Returns:
            Dictionary containing topics and analysis results
        """
        if not messages:
            return {"error": "No messages provided"}
        
        print(f"Analyzing {len(messages)} messages...")
        print(f"Using: {keyphrase_method} + {embedding_method} + {clustering_method}")
        
        # Step 1: Preprocess messages
        cleaned_messages = self.preprocess_messages(messages)
        if len(messages) != len(cleaned_messages):
             print(f"Removed {len(messages) - len(cleaned_messages)} short messages.")
        
        if len(cleaned_messages) < 2:
            return {"error": "Not enough valid messages for analysis (need at least 2)"}
        
        # Step 2: Extract key phrases
        try:
            key_phrases = self.keyphrase_extractor.extract(
                cleaned_messages, 
                method=keyphrase_method, 
                top_k=keyphrase_top_k
            )
        except Exception as e:
            print(f"Error in key phrase extraction: {e}")
            return {"error": f"Key phrase extraction failed: {str(e)}"}
        
        # Step 3: Generate embeddings
        try:
            embeddings = self.embedding_generator.generate_embeddings(
                cleaned_messages, 
                method=embedding_method
            )
        except Exception as e:
            print(f"Error in embedding generation: {e}")
            return {"error": f"Embedding generation failed: {str(e)}"}
        
        # Step 4: Perform clustering
        try:
            clustering_params = clustering_params or {}
            cluster_labels, clustering_info = self.clusterer.cluster(
                embeddings, 
                method=clustering_method, 
                **clustering_params
            )
        except Exception as e:
            print(f"Error in clustering: {e}")
            return {"error": f"Clustering failed: {str(e)}"}
        
        # Step 5: Generate topic summaries
        topics = self.generate_topic_summaries(cleaned_messages, key_phrases, cluster_labels)
        
        # Step 6: Prepare results
        results = {
            'total_messages': len(messages),
            'processed_messages': len(cleaned_messages),
            'num_topics': len(topics),
            'topics': topics,
            'noise_messages': int(np.sum(cluster_labels == -1)),
            'methods_used': {
                'keyphrase_extraction': keyphrase_method,
                'embedding_generation': embedding_method,
                'clustering': clustering_method
            },
            'clustering_info': clustering_info,
            'embedding_info': self.embedding_generator.get_embedding_info(embeddings),
            'cluster_labels': cluster_labels.tolist(),
            'key_phrases_per_message': key_phrases,
            'cleaned_messages': cleaned_messages
        }
        
        # Store results for later use
        self.last_results = results
        
        return results
    
    def get_available_methods(self) -> Dict[str, List[str]]:
        """
        Get available methods for each component.
        
        Returns:
            Dictionary with available methods for each component
        """
        return {
            'keyphrase_methods': [
                'bert', 'tfidf', 'rake', 'yake', 'textrank', 'pos', 
                'keybert', 'hybrid_pos_tfidf', 'hybrid_pos_keybert'
            ],
            'embedding_methods': [
                'sentence_transformer', 'bert_cls', 'bert_mean', 'tfidf', 'combined'
            ],
            'clustering_methods': [
                'dbscan', 'kmeans', 'agglomerative'
            ]
        }


if __name__ == '__main__':
    # 1. Define sample messages with distinct topics
    sample_messages = [
        "How can I contact reo.dev suppport?",
        "Hi",
        "What documents do you have access to?",
        "What can you tell me about customer support chatbot features?",
        "What day is today?",
        "What does reo.dev do?",
        "How do they de-anonymize user data?"
    ]

    # 2. Initialize the TopicAnalyzer
    analyzer = TopicAnalyzer()

    # 3. Analyze the messages
    # Using robust defaults: KeyBERT + SentenceTransformer + DBSCAN
    print("🚀 Starting topic analysis...")
    print("-" * 40)
    try:
        # Use adaptive DBSCAN to automatically find the best parameters
        analysis_results = analyzer.analyze_messages(
            sample_messages,
            keyphrase_method='keybert',
            embedding_method='combined',
            clustering_method='dbscan',
            clustering_params={'adaptive': True, 'eps': 0.6, 'min_samples': 1}
            # clustering_params={}
        )

        if "error" in analysis_results:
            print(f"❌ An error occurred: {analysis_results['error']}")
        else:
            # 4. Display the results
            print("\n✅ Analysis Complete!")
            print(f"Total messages processed: {analysis_results['processed_messages']}")
            print(f"Number of topics found: {analysis_results['num_topics']}")
            print(f"Number of noise messages (unassigned): {analysis_results['noise_messages']}")
            print("-" * 40)

            # Display information about each detected topic
            print("\n📊 Detected Topics Summary 📊\n")
            if not analysis_results['topics']:
                print("No distinct topics were found. Try adjusting clustering parameters.")
            else:
                for topic_id, topic_info in analysis_results['topics'].items():
                    print(f"--- Topic {topic_id} ---")
                    print(f"  🏷️ Name: {topic_info['topic_name']}")
                    print(f"  #️⃣ Message Count: {topic_info['message_count']}")
                    print(f"  🔑 Key Phrases: {', '.join(topic_info['key_phrases'])}")
                    print(f"  ✉️ Sample Message: '{topic_info['sample_messages'][0]}'")
                    print()

            # Display the topic assigned to each message
            print("\n📄 Document Topic Assignments 📄\n")
            labels = analysis_results['cluster_labels']
            cleaned_messages = analysis_results['cleaned_messages']
            
            for i, message in enumerate(cleaned_messages):
                label = labels[i]
                topic_name = f"Topic {label}" if label != -1 else "Noise (Unassigned)"
                print(f"Message: \"{message}\"")
                print(f"--> Assigned Topic: {topic_name}\n")

    except Exception as e:
        import traceback
        print(f"A critical error occurred during analysis: {e}")
        traceback.print_exc()