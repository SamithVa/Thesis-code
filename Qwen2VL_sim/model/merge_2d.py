import torch

def self_soft_matching_2d(metric: torch.Tensor, r: float = 0.8, vis_dir = None):
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
    num_retain_token = round(seq_len * r)
    metric_nomalized = metric.clone().detach()
    
    # Normalize the metric tensor along the last dimension
    metric_nomalized = metric / metric.norm(dim=-1, keepdim=True, dtype=metric.dtype)
    
    # Compute similarity scores using dot product
    scores = metric_nomalized @ metric_nomalized.T

    # output visualization similarity between visual features (heatmap)
    if vis_dir is not None:
        import seaborn as sns
        import matplotlib.pyplot as plt

        num_tokens = scores.shape[0]
        n_select = 25

        if num_tokens > n_select:
            # Randomly pick a starting index such that there are n_select tokens available
            start_idx = torch.randint(0, num_tokens - n_select + 1, (1,)).item()
            # Create a tensor of ordered indices starting at the random start index, with n_select tokens
            ordered_indices = torch.arange(start_idx, start_idx + n_select)
            # Subset the similarity matrix using these indices (for both rows and columns)
            scores_subset = scores[ordered_indices][:, ordered_indices]
        else:
            scores_subset = scores
        scores_np = scores_subset.float().detach().cpu().numpy()

        tick_labels = list(range(1, n_select + 1))

        plt.figure(figsize=(10, 8))
        sns.heatmap(scores_np, cmap='inferno', square=True,
                    xticklabels=tick_labels, yticklabels=tick_labels, annot=False)
        plt.title('Token Similarity Heatmap')
        plt.savefig(f'{vis_dir}visualize_token_similarity.png')
        print('token similarity visualization is saved')

    
    # Create a diagonal mask to remove self-matching influence
    scores_diag = torch.tril(torch.ones(seq_len, seq_len, device=metric.device, dtype=metric.dtype)) * 2
    
    # Subtract diagonal influence
    scores -= scores_diag

    # Print for a range of thresholds (0.4 to 0.9) the number of tokens
    # that have at least one other token with similarity above the threshold.
    # print(f"Total Visual Tokens : {metric.shape[0]}")
    # thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # for thr in thresholds:
    #     # Create a boolean mask where each element is True if the score is higher than thr
    #     similar_mask = scores > thr
    #     # For each token (each row), check if there's at least one similar token
    #     token_has_similar = similar_mask.any(dim=-1)
    #     # Count how many tokens have at least one similar token above the threshold
    #     num_similar_tokens = token_has_similar.sum().item()
    #     print(f"Threshold {thr}: {num_similar_tokens} tokens have at least one similar token with score > {thr}.")
    
    # Find the most similar node for each token
    node_max, _ = scores.max(dim=-1)
    
    # Sort indices by similarity in descending order
    edge_idx = node_max.argsort(dim=-1, descending=False)[..., None]
    
    # Select unmerged tokens
    unm_idx = edge_idx[seq_len-num_retain_token:, :]
        
    # Select tokens
    unm_idx_sort, _ = unm_idx.sort(dim=0)
    # remained_tokens = metric.gather(dim=0, index=unm_idx_sort.expand(num_retain_token, hidden_dim))
    mask = torch.zeros(seq_len, dtype=torch.bool, device=metric.device)
    mask.scatter_(0, unm_idx_sort.squeeze(-1), True)
    
    # return remained_tokens, mask
    return mask 


# Test the function with random input
# if __name__ == "__main__":
#     seq_len, feature_dim, r = 10, 5, 5
#     device = 'cuda'
#     metric = torch.randn(size=[seq_len, feature_dim], dtype=torch.bfloat16, device=device)
#     merged_result, mask = self_soft_matching_2d(metric=metric, r=r)
#     print("Metric:", metric)
#     print("Merged Result:", merged_result)
#     print("Mask:", mask)
