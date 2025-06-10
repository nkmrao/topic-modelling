import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Union, Optional
import warnings
warnings.filterwarnings('ignore')


class EmbeddingGenerator:
    """
    A module for generating various types of embeddings from text messages.
    
    Supported embedding methods:
    1. Sentence Transformers (various models)
    2. TF-IDF vectors
    3. BERT embeddings (CLS token or mean pooling)
    4. Combined embeddings
    """
    
    def __init__(self, 
                 sentence_model: str = 'all-MiniLM-L6-v2',
                 bert_model: str = 'bert-base-uncased',
                 device: str = 'auto'):
        """
        Initialize the EmbeddingGenerator.
        
        Args:
            sentence_model: Sentence transformer model name
            bert_model: BERT model name
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.sentence_model_name = sentence_model
        self.bert_model_name = bert_model
        
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize models lazily
        self.sentence_transformer = None
        self.bert_tokenizer = None
        self.bert_model = None
        self.tfidf_vectorizer = None
    
    def _load_sentence_transformer(self):
        """Lazy loading of sentence transformer model."""
        if self.sentence_transformer is None:
            print(f"Loading Sentence Transformer model: {self.sentence_model_name}")
            self.sentence_transformer = SentenceTransformer(self.sentence_model_name, device=self.device)
    
    def _load_bert_model(self):
        """Lazy loading of BERT model."""
        if self.bert_tokenizer is None:
            print(f"Loading BERT model: {self.bert_model_name}")
            self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
            self.bert_model = AutoModel.from_pretrained(self.bert_model_name).to(self.device)
    
    def _load_tfidf_vectorizer(self, messages: List[str]):
        """Initialize and fit TF-IDF vectorizer."""
        if self.tfidf_vectorizer is None:
            print("Initializing TF-IDF vectorizer...")
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 3),
                max_df=0.95,
                min_df=1,
                sublinear_tf=True
            )
            self.tfidf_vectorizer.fit(messages)
    
    def generate_sentence_transformer_embeddings(self, 
                                               messages: List[str], 
                                               batch_size: int = 32,
                                               show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings using Sentence Transformers.
        
        Args:
            messages: List of messages
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings with shape (n_messages, embedding_dim)
        """
        self._load_sentence_transformer()
        
        print(f"Generating Sentence Transformer embeddings for {len(messages)} messages...")
        embeddings = self.sentence_transformer.encode(
            messages,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embeddings
    
    def generate_bert_embeddings(self, 
                               messages: List[str], 
                               pooling_strategy: str = 'cls',
                               max_length: int = 512) -> np.ndarray:
        """
        Generate embeddings using BERT.
        
        Args:
            messages: List of messages
            pooling_strategy: 'cls' for CLS token, 'mean' for mean pooling
            max_length: Maximum sequence length
            
        Returns:
            Numpy array of embeddings
        """
        self._load_bert_model()
        
        print(f"Generating BERT embeddings using {pooling_strategy} pooling...")
        embeddings = []
        
        for message in messages:
            try:
                # Tokenize
                inputs = self.bert_tokenizer(
                    message,
                    return_tensors='pt',
                    truncation=True,
                    max_length=max_length,
                    padding=True
                ).to(self.device)
                
                # Get BERT outputs
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    hidden_states = outputs.last_hidden_state
                    
                    if pooling_strategy == 'cls':
                        # Use CLS token embedding
                        embedding = hidden_states[0, 0, :].cpu().numpy()
                    elif pooling_strategy == 'mean':
                        # Use mean pooling
                        attention_mask = inputs['attention_mask']
                        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                        embedding = (sum_embeddings / sum_mask)[0].cpu().numpy()
                    else:
                        raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
                    
                    embeddings.append(embedding)
                    
            except Exception as e:
                print(f"Error processing message with BERT: {e}")
                # Add zero embedding as fallback
                if embeddings:
                    embeddings.append(np.zeros_like(embeddings[0]))
                else:
                    embeddings.append(np.zeros(768))  # Default BERT embedding size
        
        return np.array(embeddings)
    
    def generate_tfidf_embeddings(self, messages: List[str]) -> np.ndarray:
        """
        Generate TF-IDF embeddings.
        
        Args:
            messages: List of messages
            
        Returns:
            Sparse matrix of TF-IDF embeddings converted to dense numpy array
        """
        self._load_tfidf_vectorizer(messages)
        
        print("Generating TF-IDF embeddings...")
        tfidf_matrix = self.tfidf_vectorizer.transform(messages)
        
        # Convert to dense array
        embeddings = tfidf_matrix.toarray()
        
        return embeddings
    
    def generate_combined_embeddings(self, 
                                   messages: List[str],
                                   methods: List[str] = ['sentence_transformer', 'tfidf'],
                                   weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Generate combined embeddings from multiple methods.
        
        Args:
            messages: List of messages
            methods: List of embedding methods to combine
            weights: Weights for combining embeddings (if None, equal weights used)
            
        Returns:
            Combined embeddings
        """
        if weights is None:
            weights = [1.0] * len(methods)
        
        if len(weights) != len(methods):
            raise ValueError("Number of weights must match number of methods")
        
        print(f"Generating combined embeddings using: {', '.join(methods)}")
        
        all_embeddings = []
        
        for method in methods:
            if method == 'sentence_transformer':
                emb = self.generate_sentence_transformer_embeddings(messages, show_progress=False)
            elif method == 'bert_cls':
                emb = self.generate_bert_embeddings(messages, pooling_strategy='cls')
            elif method == 'bert_mean':
                emb = self.generate_bert_embeddings(messages, pooling_strategy='mean')
            elif method == 'tfidf':
                emb = self.generate_tfidf_embeddings(messages)
            else:
                raise ValueError(f"Unknown embedding method: {method}")
            
            # Normalize embeddings
            emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
            all_embeddings.append(emb_norm)
        
        # Combine embeddings with weights
        combined = np.zeros_like(all_embeddings[0])
        for emb, weight in zip(all_embeddings, weights):
            # Handle different dimensionalities by padding or truncating
            if emb.shape[1] != combined.shape[1]:
                if emb.shape[1] > combined.shape[1]:
                    # Truncate larger embedding
                    emb = emb[:, :combined.shape[1]]
                else:
                    # Pad smaller embedding
                    padding = np.zeros((emb.shape[0], combined.shape[1] - emb.shape[1]))
                    emb = np.concatenate([emb, padding], axis=1)
            
            combined += weight * emb
        
        # Normalize final combined embeddings
        combined = combined / (np.linalg.norm(combined, axis=1, keepdims=True) + 1e-8)
        
        return combined
    
    def generate_embeddings(self, 
                          messages: List[str], 
                          method: str = 'sentence_transformer',
                          **kwargs) -> np.ndarray:
        """
        Generate embeddings using the specified method.
        
        Args:
            messages: List of messages
            method: Embedding method ('sentence_transformer', 'bert_cls', 'bert_mean', 
                   'tfidf', 'combined')
            **kwargs: Additional arguments specific to each method
            
        Returns:
            Numpy array of embeddings
        """
        if not messages:
            raise ValueError("No messages provided")
        
        print(f"Generating embeddings using {method.upper()}...")
        
        if method == 'sentence_transformer':
            return self.generate_sentence_transformer_embeddings(messages, **kwargs)
        elif method == 'bert_cls':
            return self.generate_bert_embeddings(messages, pooling_strategy='cls', **kwargs)
        elif method == 'bert_mean':
            return self.generate_bert_embeddings(messages, pooling_strategy='mean', **kwargs)
        elif method == 'tfidf':
            return self.generate_tfidf_embeddings(messages)
        elif method == 'combined':
            return self.generate_combined_embeddings(messages, **kwargs)
        else:
            raise ValueError(f"Unknown embedding method: {method}. "
                           f"Available methods: sentence_transformer, bert_cls, bert_mean, tfidf, combined")
    
    def get_embedding_info(self, embeddings: np.ndarray) -> dict:
        """
        Get information about the generated embeddings.
        
        Args:
            embeddings: Generated embeddings
            
        Returns:
            Dictionary with embedding statistics
        """
        return {
            'shape': embeddings.shape,
            'dimensions': embeddings.shape[1],
            'num_messages': embeddings.shape[0],
            'mean_magnitude': np.mean(np.linalg.norm(embeddings, axis=1)),
            'std_magnitude': np.std(np.linalg.norm(embeddings, axis=1)),
            'dtype': str(embeddings.dtype)
        }