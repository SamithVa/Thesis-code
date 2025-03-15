import torch

def self_soft_matching_2d(metric: torch.Tensor, r: int):
    """
    Performs self-soft matching on the given 2D metric tensor.

    Args:
        metric (torch.Tensor): Input tensor of shape (seq_len, feature_dim).
        r (int): Number of remaining tokens after merging.

    Returns:
        torch.Tensor: The selected tokens after self-soft matching.
        torch.Tensor: A mask indicating selected tokens.
    """
    seq_len, hidden_dim = metric.shape
    metric_nomalized = metric.clone().detach()
    
    with torch.no_grad():
        # Normalize the metric tensor along the last dimension
        metric_nomalized = metric / metric.norm(dim=-1, keepdim=True, dtype=metric.dtype)
        
        # Compute similarity scores using dot product
        scores = metric_nomalized @ metric_nomalized.T
        
        # Create a diagonal mask to remove self-matching influence
        scores_diag = torch.tril(torch.ones(seq_len, seq_len, device=metric.device, dtype=metric.dtype)) * 2
        
        # Subtract diagonal influence
        scores -= scores_diag
        
        # Find the most similar node for each token
        node_max, _ = scores.max(dim=-1)
        
        # Sort indices by similarity in descending order
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        
        # Select unmerged tokens
        unm_idx = edge_idx[seq_len-r:, :]
        
    # Select tokens
    unm_idx_sort, _ = unm_idx.sort(dim=0)
    remained_tokens = metric.gather(dim=0, index=unm_idx_sort.expand(r, hidden_dim))
    mask = torch.zeros(seq_len, dtype=torch.bool, device=metric.device)
    mask.scatter_(0, unm_idx_sort.squeeze(-1), True)
    
    return remained_tokens, mask

# Test the function with random input
if __name__ == "__main__":
    seq_len, feature_dim, r = 10, 5, 5
    device = 'cuda'
    metric = torch.randn(size=[seq_len, feature_dim], dtype=torch.bfloat16, device=device)
    merged_result, mask = self_soft_matching_2d(metric=metric, r=r)
    print("Metric:", metric)
    print("Merged Result:", merged_result)
    print("Mask:", mask)
