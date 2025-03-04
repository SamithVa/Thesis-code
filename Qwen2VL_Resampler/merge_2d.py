import torch


def self_soft_matching(metric: torch.Tensor, r: int):
    """
    Performs self-soft matching on the given metric tensor.

    Args:
        metric (torch.Tensor): Input tensor of shape (seq_len, feature_dim).
        r (int): Number of remaining tokens after merging.

    Returns:
        function: A merge function that can be applied to another tensor.
    """
    t = metric.shape[0] # seq_len

    with torch.no_grad():
        # Normalize the metric tensor along the last dimension
        metric = metric / metric.norm(dim=-1, keepdim=True)

        # Compute similarity scores using dot product
        scores = metric @ metric.transpose(-1, -2)

        # Create a diagonal mask to remove self-matching influence
        scores_diag = torch.tril(torch.ones(t, t, device=metric.device)) * 2

        # Subtract diagonal influence
        scores -= scores_diag

        # Find the most similar node for each token
        node_max, _ = scores.max(dim=-1)

        # Sort indices by similarity in descending order
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        # Select indices of remaining tokens
        unm_idx = edge_idx[:r]
        print(unm_idx)

    def merge(src: torch.Tensor) -> torch.Tensor:
        """
        Merges selected remaining tokens.

        Args:
            src (torch.Tensor): Source tensor of shape (seq_len, feature_dim).

        Returns:
            torch.Tensor: Merged tensor.
        """
        # Gather the selected remaining tokens
        unm = src.gather(dim=-2, index=unm_idx.expand(r, src.shape[-1]))
        return unm.gather(dim=-2, index=unm_idx.argsort(dim=0).expand(r, src.shape[-1]))

    return merge

# Test the function with random input
if __name__ == "__main__":
    t, feature_dim, r = 256, 512, 128
    metric = torch.randn(t, feature_dim)
    print(metric)
    merge_fn = self_soft_matching(metric, r)
    test_tensor = torch.randn(t, feature_dim)
    merged_result = merge_fn(test_tensor)
    print("Merged Result:", merged_result.shape)