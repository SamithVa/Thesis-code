
import torch


def self_soft_matching(metric: torch.Tensor, r: int):
    """
    Performs self-soft matching on the given metric tensor.

    Args:
        metric (torch.Tensor): Input tensor of shape (batch_size, t, feature_dim).
        r (int): Number of remaining tokens after merging.

    Returns:
        function: A merge function that can be applied to another tensor.
    """
    batch_size, seq_len, hidden_dim = metric.shape
    metric_nomalized = metric.clone().detach()
    with torch.no_grad():
        # Normalize the metric tensor along the last dimension
        metric_nomalized = metric / metric.norm(dim=-1, keepdim=True, dtype=metric.dtype)

        # Compute similarity scores using dot product
        scores = metric_nomalized @ metric_nomalized.transpose(-1, -2)
        """
        tensor([[[ 1.0000,  0.1372,  0.3501,  0.1231, -0.5558],
                [ 0.1372,  1.0000, -0.2325, -0.4169,  0.0659],
                [ 0.3501, -0.2325,  1.0000, -0.0410, -0.0259],
                [ 0.1231, -0.4169, -0.0410,  1.0000, -0.2027],
                [-0.5558,  0.0659, -0.0259, -0.2027,  1.0000]]])
        """

        # Create a diagonal mask to remove self-matching influence
        scores_diag = torch.tril(torch.ones(seq_len, seq_len, device=metric.device, dtype=metric.dtype)) * 2
        """
        tensor([[2., 0., 0., 0., 0.],
                [2., 2., 0., 0., 0.],
                [2., 2., 2., 0., 0.],
                [2., 2., 2., 2., 0.],
                [2., 2., 2., 2., 2.]])
        """
        scores_diag = scores_diag.expand(batch_size, -1, -1) # shape [batch_size, seq_len, seq_len]

        # Subtract diagonal influence
        scores -= scores_diag
        """
        tensor([[[-1.0000,  0.1372,  0.3501,  0.1231, -0.5558],
                [-1.8628, -1.0000, -0.2325, -0.4169,  0.0659],
                [-1.6499, -2.2325, -1.0000, -0.0410, -0.0259],
                [-1.8769, -2.4169, -2.0410, -1.0000, -0.2027],
                [-2.5558, -1.9341, -2.0259, -2.2027, -1.0000]]])
        """
        # print(scores)

        # Find the most similar node for each token
        node_max, _ = scores.max(dim=-1)
        # print(node_max)
        # Sort indices by similarity in descending order
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        # print(edge_idx)
        # unmerge_tokens
        unm_idx = edge_idx[...,seq_len-r:, :] # unmerge tokens
        # print(unm_idx)

    # select tokens
    bsz, t, c = metric.shape
    unm_idx_sort, _ = unm_idx.sort(dim=1)
    remained_tokens = metric.gather(dim=-2, index=unm_idx_sort.expand(bsz, r, c))
    mask = torch.zeros([bsz, t], dtype=torch.bool, device=metric.device)
    # print(unm_idx_sort)
    mask.scatter_(1, unm_idx_sort.squeeze(-1), True)
    return remained_tokens, mask

# Test the function with random input
if __name__ == "__main__":
    t, feature_dim, r = 10, 5, 5 # 
    bsz = 1
    device = 'cuda'
    metric = torch.randn(size = [bsz, t, feature_dim], dtype=torch.bfloat16, device=device)
    merged_result, mask = self_soft_matching(metric=metric, r=r)
    print("Metric:", metric)
    print("Merged Result:", merged_result)
    print("Mask", mask)
