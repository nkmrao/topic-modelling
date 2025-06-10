import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
import networkx as nx
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any
import yake
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')


class KeyPhrasesExtractor:
    """
    A comprehensive key phrase extraction module supporting multiple approaches.
    
    Supported methods:
    1. BERT-based attention extraction
    2. TF-IDF
    3. RAKE (Rapid Automatic Keyword Extraction)
    4. YAKE (Yet Another Keyword Extractor)
    5. TextRank
    6. POS tagging
    7. KeyBERT
    8. Hybrid approaches
    """
    
    def __init__(self, 
                 bert_model: str = 'bert-base-uncased',
                 keybert_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the KeyPhrasesExtractor.
        
        Args:
            bert_model: BERT model name for attention-based extraction
            keybert_model: Model name for KeyBERT
        """
        self.bert_model_name = bert_model
        self.keybert_model_name = keybert_model
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize models lazily
        self.tokenizer = None
        self.bert_model = None
        self.ner_pipeline = None
        self.keybert_model = None
        self.tfidf_vectorizer = None
    
    def _load_bert_models(self):
        """Lazy loading of BERT models."""
        if self.tokenizer is None:
            print("Loading BERT models...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
            self.bert_model = AutoModel.from_pretrained(self.bert_model_name)
            self.ner_pipeline = pipeline("ner", 
                                       model="dbmdz/bert-large-cased-finetuned-conll03-english",
                                       aggregation_strategy="simple")
    
    def _load_keybert_model(self):
        """Lazy loading of KeyBERT model."""
        if self.keybert_model is None:
            print("Loading KeyBERT model...")
            self.keybert_model = KeyBERT(model=self.keybert_model_name)
    
    def extract_bert_attention(self, messages: List[str], top_k: int = 5) -> List[List[str]]:
        """
        Extract key phrases using BERT attention weights.
        
        Args:
            messages: List of messages
            top_k: Number of top phrases to extract per message
            
        Returns:
            List of key phrases for each message
        """
        self._load_bert_models()
        all_key_phrases = []
        
        for message in messages:
            try:
                inputs = self.tokenizer(message, return_tensors='pt', 
                                      truncation=True, max_length=512, 
                                      padding=True)
                
                with torch.no_grad():
                    outputs = self.bert_model(**inputs, output_attentions=True)
                
                attention = outputs.attentions[-1]
                cls_attention = attention[0, :, 0, :].mean(dim=0)
                
                tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                token_scores = cls_attention.numpy()
                
                phrases = self._extract_phrases_from_attention(tokens, token_scores, top_k)
                all_key_phrases.append(phrases)
                
            except Exception as e:
                print(f"Error processing message with BERT: {e}")
                all_key_phrases.append([])
        
        return all_key_phrases
    
    def _extract_phrases_from_attention(self, tokens: List[str], scores: np.ndarray, top_k: int) -> List[str]:
        """Extract phrases based on attention scores."""
        filtered_tokens_scores = []
        current_word = ""
        current_score = 0
        
        for token, score in zip(tokens, scores):
            if token.startswith('##'):
                current_word += token[2:]
                current_score = max(current_score, score)
            elif token not in ['[CLS]', '[SEP]', '[PAD]']:
                if current_word:
                    filtered_tokens_scores.append((current_word, current_score))
                current_word = token
                current_score = score
        
        if current_word:
            filtered_tokens_scores.append((current_word, current_score))
        
        filtered_tokens_scores.sort(key=lambda x: x[1], reverse=True)
        
        phrases = []
        for word, score in filtered_tokens_scores:
            if (word.lower() not in self.stop_words and 
                len(word) > 2 and 
                word.isalpha()):
                phrases.append(word)
                if len(phrases) >= top_k:
                    break
        
        return phrases[:top_k]
    
    def extract_tfidf(self, messages: List[str], top_k: int = 5) -> List[List[str]]:
        """
        Extract key phrases using TF-IDF.
        
        Args:
            messages: List of messages
            top_k: Number of top phrases to extract per message
            
        Returns:
            List of key phrases for each message
        """
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 3),
                max_df=0.8,
                min_df=1
            )
        
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(messages)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            all_key_phrases = []
            for i, message in enumerate(messages):
                # Get TF-IDF scores for this message
                tfidf_scores = tfidf_matrix[i].toarray()[0]
                
                # Get top features
                top_indices = np.argsort(tfidf_scores)[::-1][:top_k]
                phrases = [feature_names[idx] for idx in top_indices if tfidf_scores[idx] > 0]
                
                all_key_phrases.append(phrases)
            
            return all_key_phrases
            
        except Exception as e:
            print(f"Error with TF-IDF extraction: {e}")
            return [[] for _ in messages]
    
    def extract_rake(self, messages: List[str], top_k: int = 5) -> List[List[str]]:
        """
        Extract key phrases using RAKE algorithm.
        
        Args:
            messages: List of messages
            top_k: Number of top phrases to extract per message
            
        Returns:
            List of key phrases for each message
        """
        def _generate_candidate_keywords(sentences):
            phrase_list = []
            for sentence in sentences:
                words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
                phrase = []
                for word in words:
                    if word not in self.stop_words and len(word) > 2:
                        phrase.append(word)
                    else:
                        if phrase:
                            phrase_list.append(' '.join(phrase))
                            phrase = []
                if phrase:
                    phrase_list.append(' '.join(phrase))
            return phrase_list
        
        def _calculate_word_scores(phrase_list):
            word_freq = defaultdict(int)
            word_degree = defaultdict(int)
            
            for phrase in phrase_list:
                words = phrase.split()
                word_count = len(words)
                
                for word in words:
                    word_freq[word] += 1
                    word_degree[word] += word_count - 1
            
            word_scores = {}
            for word in word_freq:
                word_scores[word] = word_degree[word] / word_freq[word]
            
            return word_scores
        
        all_key_phrases = []
        
        for message in messages:
            try:
                sentences = sent_tokenize(message)
                phrase_list = _generate_candidate_keywords(sentences)
                
                if not phrase_list:
                    all_key_phrases.append([])
                    continue
                
                word_scores = _calculate_word_scores(phrase_list)
                
                phrase_scores = []
                for phrase in phrase_list:
                    words = phrase.split()
                    score = sum(word_scores.get(word, 0) for word in words)
                    phrase_scores.append((phrase, score))
                
                phrase_scores.sort(key=lambda x: x[1], reverse=True)
                phrases = [phrase for phrase, score in phrase_scores[:top_k]]
                
                all_key_phrases.append(phrases)
                
            except Exception as e:
                print(f"Error with RAKE extraction: {e}")
                all_key_phrases.append([])
        
        return all_key_phrases
    
    def extract_yake(self, messages: List[str], top_k: int = 5) -> List[List[str]]:
        """
        Extract key phrases using YAKE algorithm.
        
        Args:
            messages: List of messages
            top_k: Number of top phrases to extract per message
            
        Returns:
            List of key phrases for each message
        """
        all_key_phrases = []
        
        for message in messages:
            try:
                kw_extractor = yake.KeywordExtractor(
                    lan='en',
                    n=3,  # n-gram size
                    dedupLim=0.7,
                    top=top_k
                )
                
                keywords = kw_extractor.extract_keywords(message)
                phrases = [kw[0] for kw in keywords]
                
                all_key_phrases.append(phrases)
                
            except Exception as e:
                print(f"Error with YAKE extraction: {e}")
                all_key_phrases.append([])
        
        return all_key_phrases
    
    def extract_textrank(self, messages: List[str], top_k: int = 5) -> List[List[str]]:
        """
        Extract key phrases using TextRank algorithm.
        
        Args:
            messages: List of messages
            top_k: Number of top phrases to extract per message
            
        Returns:
            List of key phrases for each message
        """
        def _build_graph(words, window_size=2):
            graph = nx.Graph()
            
            for i, word in enumerate(words):
                if word not in self.stop_words and len(word) > 2:
                    graph.add_node(word)
                    
                    for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
                        if i != j and words[j] not in self.stop_words and len(words[j]) > 2:
                            graph.add_edge(word, words[j])
            
            return graph
        
        all_key_phrases = []
        
        for message in messages:
            try:
                words = word_tokenize(message.lower())
                words = [word for word in words if word.isalpha()]
                
                if len(words) < 3:
                    all_key_phrases.append([])
                    continue
                
                graph = _build_graph(words)
                
                if len(graph.nodes()) == 0:
                    all_key_phrases.append([])
                    continue
                
                scores = nx.pagerank(graph)
                sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                phrases = [word for word, score in sorted_words[:top_k]]
                
                all_key_phrases.append(phrases)
                
            except Exception as e:
                print(f"Error with TextRank extraction: {e}")
                all_key_phrases.append([])
        
        return all_key_phrases
    
    def extract_pos_tagging(self, messages: List[str], top_k: int = 5) -> List[List[str]]:
        """
        Extract key phrases using POS tagging (focus on nouns and adjectives).
        
        Args:
            messages: List of messages
            top_k: Number of top phrases to extract per message
            
        Returns:
            List of key phrases for each message
        """
        all_key_phrases = []
        
        for message in messages:
            try:
                words = word_tokenize(message)
                pos_tags = pos_tag(words)
                
                # Extract nouns, proper nouns, and adjectives
                key_words = []
                for word, pos in pos_tags:
                    if (pos in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'] and
                        word.lower() not in self.stop_words and
                        len(word) > 2 and
                        word.isalpha()):
                        key_words.append(word.lower())
                
                # Count frequency and get top words
                word_counts = Counter(key_words)
                phrases = [word for word, count in word_counts.most_common(top_k)]
                
                all_key_phrases.append(phrases)
                
            except Exception as e:
                print(f"Error with POS tagging extraction: {e}")
                all_key_phrases.append([])
        
        return all_key_phrases
    
    def extract_keybert(self, messages: List[str], top_k: int = 5) -> List[List[str]]:
        """
        Extract key phrases using KeyBERT.
        
        Args:
            messages: List of messages
            top_k: Number of top phrases to extract per message
            
        Returns:
            List of key phrases for each message
        """
        self._load_keybert_model()
        all_key_phrases = []
        
        for message in messages:
            try:
                keywords = self.keybert_model.extract_keywords(
                    message, 
                    keyphrase_ngram_range=(1, 3), 
                    stop_words='english',
                    top_n=top_k
                )
                phrases = [kw[0] for kw in keywords]
                all_key_phrases.append(phrases)
                
            except Exception as e:
                print(f"Error with KeyBERT extraction: {e}")
                all_key_phrases.append([])
        
        return all_key_phrases
    
    def extract_hybrid_pos_tfidf(self, messages: List[str], top_k: int = 5) -> List[List[str]]:
        """
        Hybrid approach: Use POS tagging to identify candidates, TF-IDF to score them.
        
        Args:
            messages: List of messages
            top_k: Number of top phrases to extract per message
            
        Returns:
            List of key phrases for each message
        """
        # Get POS-based candidates
        pos_candidates = self.extract_pos_tagging(messages, top_k * 2)
        
        # Create documents from POS candidates for TF-IDF
        candidate_docs = []
        for candidates in pos_candidates:
            candidate_docs.append(' '.join(candidates))
        
        # Score using TF-IDF
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_df=0.8,
                min_df=1
            )
        
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(candidate_docs)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            all_key_phrases = []
            for i, candidates in enumerate(pos_candidates):
                if not candidates:
                    all_key_phrases.append([])
                    continue
                
                tfidf_scores = tfidf_matrix[i].toarray()[0]
                
                # Score candidates based on TF-IDF
                candidate_scores = []
                for candidate in candidates:
                    if candidate in feature_names:
                        idx = np.where(feature_names == candidate)[0]
                        if len(idx) > 0:
                            score = tfidf_scores[idx[0]]
                            candidate_scores.append((candidate, score))
                
                candidate_scores.sort(key=lambda x: x[1], reverse=True)
                phrases = [phrase for phrase, score in candidate_scores[:top_k]]
                
                all_key_phrases.append(phrases)
            
            return all_key_phrases
            
        except Exception as e:
            print(f"Error with hybrid POS-TF-IDF extraction: {e}")
            return pos_candidates
    
    def extract_hybrid_pos_keybert(self, messages: List[str], top_k: int = 5) -> List[List[str]]:
        """
        Hybrid approach: Use POS tagging to identify candidates, KeyBERT to score them.
        
        Args:
            messages: List of messages
            top_k: Number of top phrases to extract per message
            
        Returns:
            List of key phrases for each message
        """
        self._load_keybert_model()
        
        # Get POS-based candidates
        pos_candidates = self.extract_pos_tagging(messages, top_k * 2)
        
        all_key_phrases = []
        
        for i, (message, candidates) in enumerate(zip(messages, pos_candidates)):
            try:
                if not candidates:
                    all_key_phrases.append([])
                    continue
                
                # Use KeyBERT to score the POS candidates
                keywords = self.keybert_model.extract_keywords(
                    message,
                    candidates=candidates,
                    top_n=min(top_k, len(candidates))
                )
                
                phrases = [kw[0] for kw in keywords]
                all_key_phrases.append(phrases)
                
            except Exception as e:
                print(f"Error with hybrid POS-KeyBERT extraction: {e}")
                all_key_phrases.append(candidates[:top_k] if candidates else [])
        
        return all_key_phrases
    
    def extract(self, messages: List[str], method: str = 'keybert', top_k: int = 5) -> List[List[str]]:
        """
        Extract key phrases using the specified method.
        
        Args:
            messages: List of messages
            method: Extraction method ('bert', 'tfidf', 'rake', 'yake', 'textrank',
                   'pos', 'keybert', 'hybrid_pos_tfidf', 'hybrid_pos_keybert')
            top_k: Number of top phrases to extract per message
            
        Returns:
            List of key phrases for each message
        """
        method_map = {
            'bert': self.extract_bert_attention,
            'tfidf': self.extract_tfidf,
            'rake': self.extract_rake,
            'yake': self.extract_yake,
            'textrank': self.extract_textrank,
            'pos': self.extract_pos_tagging,
            'keybert': self.extract_keybert,
            'hybrid_pos_tfidf': self.extract_hybrid_pos_tfidf,
            'hybrid_pos_keybert': self.extract_hybrid_pos_keybert
        }
        
        if method not in method_map:
            raise ValueError(f"Unknown method: {method}. Available methods: {list(method_map.keys())}")
        
        print(f"Extracting key phrases using {method.upper()}...")
        return method_map[method](messages, top_k)