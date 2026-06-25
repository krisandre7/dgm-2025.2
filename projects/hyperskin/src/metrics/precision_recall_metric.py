import torch
from typing import Dict

def compute_knn_precision_recall(
    real_features: torch.Tensor, 
    fake_features: torch.Tensor, 
    k: int = 3, 
    percentile: float = 90.0,
    batch_size: int = 1000  # Split distances into batches to manage memory
) -> Dict[str, float]:
    """
    Computes kNN-based Precision and Recall for GAN evaluation.
    
    Args:
        real_features (torch.Tensor): Tensor of real image features (N_r, D). Must be on CPU.
        fake_features (torch.Tensor): Tensor of fake image features (N_f, D). Must be on CPU.
        k (int): Number of nearest neighbors to consider (used implicitly for threshold estimation).
        percentile (float): Percentile (0-100) to determine the threshold T.
        batch_size (int): Batch size for distance calculation to prevent OOM.
        
    Returns:
        Dict[str, float]: Dictionary containing 'precision' and 'recall'.
    """
    
    N_r, D = real_features.shape
    N_f, D = fake_features.shape
    
    # 1. Calculate the Threshold T (Real-to-Real Nearest Neighbor Distances)
    
    # Calculate all real-to-real distances using batches
    real_to_real_dists = []
    for i in range(0, N_r, batch_size):
        r_batch = real_features[i : i + batch_size]
        # Calculate distances within the batch to all real features
        dists = torch.cdist(r_batch, real_features)
        
        # Exclude the distance to self (which is 0)
        # Find the k+1 smallest distance, the (k+1)-th is the kNN distance (if k=1, we take the 2nd smallest)
        k_plus_1 = min(k + 1, dists.shape[1])
        
        # Find the k-th nearest neighbor distance (k+1 smallest including self=0)
        # We use the (k+1)-th smallest value across the whole row if k_plus_1 < dists.shape[1]
        if dists.shape[1] > 1:
            knn_dists, _ = torch.kthvalue(dists, k=k_plus_1, dim=1)
            real_to_real_dists.append(knn_dists)

    real_to_real_dists = torch.cat(real_to_real_dists)
    
    # Determine the threshold T based on the percentile
    T_index = int(percentile * real_to_real_dists.numel() / 100.0)
    T_index = max(1, min(T_index, real_to_real_dists.numel()))
    
    # Find the T-th percentile distance
    T = torch.kthvalue(real_to_real_dists, k=T_index).values.item()
    
    # 2. Calculate Precision (Fake-to-Real Distance)
    # Fraction of fake samples whose nearest real neighbor is <= T
    
    fake_to_real_hits = 0
    for i in range(0, N_f, batch_size):
        f_batch = fake_features[i : i + batch_size]
        dists = torch.cdist(f_batch, real_features)
        
        # Find the minimum distance for each fake sample (NN in S_r)
        min_dist_to_real, _ = torch.min(dists, dim=1)
        
        # Count how many are within the threshold T
        fake_to_real_hits += (min_dist_to_real <= T).sum().item()
    
    precision = fake_to_real_hits / N_f
    
    # 3. Calculate Recall (Real-to-Fake Distance)
    # Fraction of real samples whose nearest fake neighbor is <= T
    
    real_to_fake_hits = 0
    for i in range(0, N_r, batch_size):
        r_batch = real_features[i : i + batch_size]
        dists = torch.cdist(r_batch, fake_features)
        
        # Find the minimum distance for each real sample (NN in S_f)
        min_dist_to_fake, _ = torch.min(dists, dim=1)
        
        # Count how many are within the threshold T
        real_to_fake_hits += (min_dist_to_fake <= T).sum().item()
        
    recall = real_to_fake_hits / N_r
    
    return {'precision': precision, 'recall': recall}