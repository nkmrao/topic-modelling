import numpy as np
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings
from typing import Tuple, Dict, Any, Optional, List
warnings.filterwarnings('ignore')


class TopicClusterer:
    """
    A module for clustering embeddings to identify topics using various clustering algorithms.
    
    Supported clustering methods:
    1. DBSCAN (Density-based clustering)
    2. K-Means
    3. Agglomerative Clustering
    4. Adaptive clustering (automatically selects best parameters)
    """
    
    def __init__(self):
        """Initialize the TopicClusterer."""
        self.scaler = StandardScaler()
        self.last_clustering_info = {}
    
    def cluster_dbscan(self, 
                      embeddings: np.ndarray, 
                      eps: float = 0.3, 
                      min_samples: int = 2,
                      metric: str = 'cosine',
                      adaptive: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform DBSCAN clustering.
        
        Args:
            embeddings: Input embeddings
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN minimum samples parameter
            metric: Distance metric ('cosine', 'euclidean', 'manhattan')
            adaptive: Whether to adaptively tune parameters
            
        Returns:
            Tuple of (cluster_labels, clustering_info)
        """
        print("Performing DBSCAN clustering...")
        
        # Normalize embeddings
        normalized_embeddings = self.scaler.fit_transform(embeddings)
        
        if adaptive:
            return self._adaptive_dbscan(normalized_embeddings, eps, min_samples, metric)
        else:
            clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
            labels = clusterer.fit_predict(normalized_embeddings)
            
            info = self._get_clustering_info(labels, normalized_embeddings, 'DBSCAN')
            info.update({
                'eps': eps,
                'min_samples': min_samples,
                'metric': metric
            })
            
            return labels, info
    
    def _adaptive_dbscan(self, 
                        embeddings: np.ndarray, 
                        base_eps: float, 
                        min_samples: int, 
                        metric: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Adaptively tune DBSCAN parameters to find optimal clustering.
        """
        print("Adaptively tuning DBSCAN parameters...")
        
        # Test different eps values
        eps_values = [base_eps * factor for factor in [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]]
        min_samples_values = [max(2, min_samples - 1), min_samples, min_samples + 1]
        
        best_labels = None
        best_score = -1
        best_params = {}
        
        for eps in eps_values:
            for min_samp in min_samples_values:
                try:
                    clusterer = DBSCAN(eps=eps, min_samples=min_samp, metric=metric)
                    labels = clusterer.fit_predict(embeddings)
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    noise_ratio = np.sum(labels == -1) / len(labels) if len(labels) > 0 else 1
                    
                    # Score based on number of clusters and noise ratio
                    if n_clusters > 0 and n_clusters < len(embeddings) * 0.8:
                        score = n_clusters * (1 - noise_ratio)
                        
                        # Bonus for silhouette score if we have enough non-noise points
                        non_noise_mask = labels != -1
                        if np.sum(non_noise_mask) > 1 and n_clusters > 1:
                            try:
                                silhouette = silhouette_score(embeddings[non_noise_mask], labels[non_noise_mask])
                                score += silhouette * 2  # Weight silhouette score
                            except:
                                pass
                        
                        if score > best_score:
                            best_score = score
                            best_labels = labels
                            best_params = {'eps': eps, 'min_samples': min_samp}
                
                except Exception as e:
                    continue
        
        # Fallback to original parameters if no good clustering found
        if best_labels is None:
            clusterer = DBSCAN(eps=base_eps, min_samples=min_samples, metric=metric)
            best_labels = clusterer.fit_predict(embeddings)
            best_params = {'eps': base_eps, 'min_samples': min_samples}
        
        info = self._get_clustering_info(best_labels, embeddings, 'DBSCAN (Adaptive)')
        info.update(best_params)
        info['metric'] = metric
        
        return best_labels, info
    
    def cluster_kmeans(self, 
                      embeddings: np.ndarray, 
                      n_clusters: Optional[int] = None,
                      max_clusters: int = 10,
                      random_state: int = 42,
                      adaptive: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform K-Means clustering.
        
        Args:
            embeddings: Input embeddings
            n_clusters: Number of clusters (if None, will be determined adaptively)
            max_clusters: Maximum number of clusters to consider
            random_state: Random state for reproducibility
            adaptive: Whether to adaptively determine number of clusters
            
        Returns:
            Tuple of (cluster_labels, clustering_info)
        """
        print("Performing K-Means clustering...")
        
        # Normalize embeddings
        normalized_embeddings = self.scaler.fit_transform(embeddings)
        
        if adaptive and n_clusters is None:
            return self._adaptive_kmeans(normalized_embeddings, max_clusters, random_state)
        else:
            if n_clusters is None:
                n_clusters = min(8, len(embeddings) // 3)  # Default heuristic
            
            n_clusters = max(2, min(n_clusters, len(embeddings) - 1))
            
            clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            labels = clusterer.fit_predict(normalized_embeddings)
            
            info = self._get_clustering_info(labels, normalized_embeddings, 'K-Means')
            info.update({
                'n_clusters': n_clusters,
                'inertia': clusterer.inertia_,
                'random_state': random_state
            })
            
            return labels, info
    
    def _adaptive_kmeans(self, 
                        embeddings: np.ndarray, 
                        max_clusters: int, 
                        random_state: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Adaptively determine optimal number of clusters for K-Means using elbow method and silhouette analysis.
        """
        print("Adaptively determining optimal number of clusters...")
        
        n_samples = len(embeddings)
        min_clusters = 2
        max_clusters = min(max_clusters, n_samples - 1, 15)  # Reasonable upper bound
        
        if max_clusters < min_clusters:
            # Fallback for very small datasets
            clusterer = KMeans(n_clusters=2, random_state=random_state, n_init=10)
            labels = clusterer.fit_predict(embeddings)
            info = self._get_clustering_info(labels, embeddings, 'K-Means (Adaptive)')
            info.update({'n_clusters': 2, 'inertia': clusterer.inertia_})
            return labels, info
        
        inertias = []
        silhouette_scores = []
        k_values = range(min_clusters, max_clusters + 1)
        
        for k in k_values:
            clusterer = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = clusterer.fit_predict(embeddings)
            
            inertias.append(clusterer.inertia_)
            
            if k > 1:
                try:
                    silhouette_avg = silhouette_score(embeddings, labels)
                    silhouette_scores.append(silhouette_avg)
                except:
                    silhouette_scores.append(0)
            else:
                silhouette_scores.append(0)
        
        # Find optimal k using elbow method
        optimal_k = self._find_elbow_point(k_values, inertias)
        
        # Adjust based on silhouette scores
        if len(silhouette_scores) > 0:
            max_silhouette_idx = np.argmax(silhouette_scores)
            silhouette_k = k_values[max_silhouette_idx]
            
            # Choose k that balances elbow method and silhouette score
            if abs(silhouette_k - optimal_k) <= 2:
                optimal_k = silhouette_k
        
        # Final clustering with optimal k
        clusterer = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=10)
        labels = clusterer.fit_predict(embeddings)
        
        info = self._get_clustering_info(labels, embeddings, 'K-Means (Adaptive)')
        info.update({
            'n_clusters': optimal_k,
            'inertia': clusterer.inertia_,
            'all_inertias': inertias,
            'all_silhouette_scores': silhouette_scores,
            'k_range': list(k_values)
        })
        
        return labels, info
    
    def _find_elbow_point(self, k_values: range, inertias: List[float]) -> int:
        """
        Find the elbow point in the inertia curve.
        """
        if len(inertias) < 3:
            return k_values[0]
        
        # Calculate the rate of change
        diffs = np.diff(inertias)
        diff_ratios = np.diff(diffs)
        
        # Find the point where the rate of change stabilizes
        if len(diff_ratios) > 0:
            elbow_idx = np.argmax(diff_ratios) + 2  # +2 because of double diff
            elbow_idx = min(elbow_idx, len(k_values) - 1)
            return k_values[elbow_idx]
        
        return k_values[len(k_values) // 2]  # Default to middle
    
    def cluster_agglomerative(self, 
                             embeddings: np.ndarray, 
                             n_clusters: Optional[int] = None,
                             linkage: str = 'ward',
                             max_clusters: int = 10,
                             adaptive: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform Agglomerative clustering.
        
        Args:
            embeddings: Input embeddings
            n_clusters: Number of clusters
            linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
            max_clusters: Maximum number of clusters to consider
            adaptive: Whether to adaptively determine number of clusters
            
        Returns:
            Tuple of (cluster_labels, clustering_info)
        """
        print("Performing Agglomerative clustering...")
        
        # Normalize embeddings
        normalized_embeddings = self.scaler.fit_transform(embeddings)
        
        if adaptive and n_clusters is None:
            return self._adaptive_agglomerative(normalized_embeddings, linkage, max_clusters)
        else:
            if n_clusters is None:
                n_clusters = min(8, len(embeddings) // 3)
            
            n_clusters = max(2, min(n_clusters, len(embeddings) - 1))
            
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            labels = clusterer.fit_predict(normalized_embeddings)
            
            info = self._get_clustering_info(labels, normalized_embeddings, 'Agglomerative')
            info.update({
                'n_clusters': n_clusters,
                'linkage': linkage,
                'n_connected_components': clusterer.n_connected_components_
            })
            
            return labels, info
    
    def _adaptive_agglomerative(self, 
                               embeddings: np.ndarray, 
                               linkage: str, 
                               max_clusters: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Adaptively determine optimal number of clusters for Agglomerative clustering.
        """
        print("Adaptively determining optimal clusters for Agglomerative clustering...")
        
        n_samples = len(embeddings)
        min_clusters = 2
        max_clusters = min(max_clusters, n_samples - 1, 15)
        
        if max_clusters < min_clusters:
            clusterer = AgglomerativeClustering(n_clusters=2, linkage=linkage)
            labels = clusterer.fit_predict(embeddings)
            info = self._get_clustering_info(labels, embeddings, 'Agglomerative (Adaptive)')
            info.update({'n_clusters': 2, 'linkage': linkage})
            return labels, info
        
        best_score = -1
        best_k = 2
        silhouette_scores = []
        k_values = range(min_clusters, max_clusters + 1)
        
        for k in k_values:
            try:
                clusterer = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                labels = clusterer.fit_predict(embeddings)
                
                silhouette_avg = silhouette_score(embeddings, labels)
                silhouette_scores.append(silhouette_avg)
                
                if silhouette_avg > best_score:
                    best_score = silhouette_avg
                    best_k = k
            except:
                silhouette_scores.append(0)
        
        # Final clustering with optimal k
        clusterer = AgglomerativeClustering(n_clusters=best_k, linkage=linkage)
        labels = clusterer.fit_predict(embeddings)
        
        info = self._get_clustering_info(labels, embeddings, 'Agglomerative (Adaptive)')
        info.update({
            'n_clusters': best_k,
            'linkage': linkage,
            'n_connected_components': clusterer.n_connected_components_,
            'all_silhouette_scores': silhouette_scores,
            'k_range': list(k_values)
        })
        
        return labels, info
    
    def _get_clustering_info(self, 
                           labels: np.ndarray, 
                           embeddings: np.ndarray, 
                           method: str) -> Dict[str, Any]:
        """
        Calculate clustering quality metrics.
        """
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1) if -1 in labels else 0
        
        info = {
            'method': method,
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'n_total_points': len(labels),
            'noise_ratio': n_noise / len(labels) if len(labels) > 0 else 0
        }
        
        # Calculate quality metrics if we have valid clusters
        if n_clusters > 1:
            try:
                # Filter out noise points for metrics calculation
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) > 1:
                    filtered_embeddings = embeddings[non_noise_mask]
                    filtered_labels = labels[non_noise_mask]
                    
                    info['silhouette_score'] = silhouette_score(filtered_embeddings, filtered_labels)
                    info['calinski_harabasz_score'] = calinski_harabasz_score(filtered_embeddings, filtered_labels)
                    info['davies_bouldin_score'] = davies_bouldin_score(filtered_embeddings, filtered_labels)
            except Exception as e:
                print(f"Warning: Could not calculate clustering metrics: {e}")
        
        # Cluster size distribution
        cluster_sizes = []
        for cluster_id in set(labels):
            if cluster_id != -1:  # Exclude noise
                cluster_sizes.append(np.sum(labels == cluster_id))
        
        if cluster_sizes:
            info['cluster_sizes'] = cluster_sizes
            info['avg_cluster_size'] = np.mean(cluster_sizes)
            info['std_cluster_size'] = np.std(cluster_sizes)
            info['min_cluster_size'] = min(cluster_sizes)
            info['max_cluster_size'] = max(cluster_sizes)
        
        return info
    
    def cluster(self, 
               embeddings: np.ndarray, 
               method: str = 'dbscan',
               **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform clustering using the specified method.
        
        Args:
            embeddings: Input embeddings
            method: Clustering method ('dbscan', 'kmeans', 'agglomerative')
            **kwargs: Additional arguments specific to each method
            
        Returns:
            Tuple of (cluster_labels, clustering_info)
        """
        if len(embeddings) < 2:
            raise ValueError("Need at least 2 samples for clustering")
        
        method = method.lower()
        
        if method == 'dbscan':
            return self.cluster_dbscan(embeddings, **kwargs)
        elif method == 'kmeans':
            return self.cluster_kmeans(embeddings, **kwargs)
        elif method == 'agglomerative':
            return self.cluster_agglomerative(embeddings, **kwargs)
        else:
            raise ValueError(f"Unknown clustering method: {method}. "
                           f"Available methods: dbscan, kmeans, agglomerative")
    
    def get_cluster_summary(self, labels: np.ndarray, messages: List[str]) -> Dict[int, Dict[str, Any]]:
        """
        Generate a summary of clusters with sample messages.
        
        Args:
            labels: Cluster labels
            messages: Original messages
            
        Returns:
            Dictionary mapping cluster IDs to cluster information
        """
        cluster_summary = {}
        
        for cluster_id in set(labels):
            if cluster_id == -1:  # Skip noise
                continue
            
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_messages = [messages[i] for i in cluster_indices]
            
            cluster_summary[cluster_id] = {
                'size': len(cluster_messages),
                'indices': cluster_indices.tolist(),
                'sample_messages': cluster_messages[:3],  # First 3 as samples
                'all_messages': cluster_messages
            }
        
        return cluster_summary