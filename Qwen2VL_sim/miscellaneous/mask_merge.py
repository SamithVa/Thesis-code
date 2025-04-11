import torch

def self_soft_matching_indices(metric: torch.Tensor, r: int):
    """
    Performs self-soft matching on the given metric tensor and returns a boolean mask 
    indicating which tokens are selected.

    Args:
        metric (torch.Tensor): Input tensor of shape (seq_len, feature_dim).
        r (int): Number of remaining tokens after merging.

    Returns:
        torch.Tensor: A boolean mask of shape (seq_len,), where True indicates selected indices.
    """
    seq_len = metric.shape[0]  # Get sequence length

    metric_norm = metric.clone()
    # Normalize the metric tensor along the last dimension
    metric_norm = metric_norm / metric_norm.norm(dim=-1, keepdim=True)

    # Compute similarity scores using dot product
    scores = metric_norm @ metric_norm.transpose(-1, -2)

    # Create a diagonal mask to remove self-matching influence
    scores_diag = torch.tril(torch.ones(seq_len, seq_len, device=metric_norm.device)) * 2

    # Subtract diagonal influence
    scores -= scores_diag

    # Find the most similar node for each token
    node_max, _ = scores.max(dim=-1)

    # Sort indices by similarity in descending order
    edge_idx = node_max.argsort(dim=-1, descending=True)

    # Select indices of remaining tokens
    selected_indices = edge_idx[:r]  # Shape: (r,)

    # Create a boolean mask initialized to False
    mask = torch.zeros(seq_len, dtype=torch.bool, device=metric.device)

    # Set selected indices to True
    mask[selected_indices] = True

    return mask


if __name__=="__main__":
    seq_len, emb_dim = 5, 5
    x = torch.rand([seq_len, emb_dim])
    print(x)
    y = self_soft_matching_indices(x, 2)
    print(y)