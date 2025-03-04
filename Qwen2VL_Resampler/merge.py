#
# import torch
#
#
# def self_soft_matching(
#     metric: torch.Tensor,
#     r: int,):
#
#     t = metric.shape[1]
#     with torch.no_grad():
#         metric = metric / metric.norm(dim=-1, keepdim=True)
#         a, b = metric[..., :, :], metric[..., :, :]
#         scores = a @ b.transpose(-1, -2) # a_lxb_l
#         b,_,_ = scores.shape
#         scores_diag = torch.tril(torch.ones(t,t))*2
#         scores_diag = scores_diag.expand(b, -1, -1).to(metric.device)
#
#         scores = scores-scores_diag
#         node_max, node_idx = scores.max(dim=-1) # a中最相似的点
#         edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] # a中相似度排序并得到idx，降序
#
#         unm_idx = edge_idx[..., t-r:, :]  # Unmerged Tokens # 后面的就是不merge的
#
#     def merge(src: torch.Tensor) -> torch.Tensor:
#         n, t1, c = src.shape
#         unm = src.gather(dim=-2, index=unm_idx.expand(n,  r, c))
#         unm_idx_new = unm_idx
#         all_idx = unm_idx_new
#         all_max,all_idx_idx = torch.sort(all_idx,dim=1)
#         return unm.gather(dim=-2, index=all_idx_idx.expand(n, r, c))
#
#     return merge


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
    t = metric.shape[1]

    with torch.no_grad():
        # Normalize the metric tensor along the last dimension
        metric = metric / metric.norm(dim=-1, keepdim=True)

        # Compute similarity scores using dot product
        scores = metric @ metric.transpose(-1, -2)

        batch_size = scores.shape[0]

        # Create a diagonal mask to remove self-matching influence
        scores_diag = torch.tril(torch.ones(t, t, device=metric.device)) * 2
        scores_diag = scores_diag.expand(batch_size, -1, -1)

        # Subtract diagonal influence
        scores -= scores_diag

        # Find the most similar node for each token
        node_max, _ = scores.max(dim=-1)

        # Sort indices by similarity in descending order
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        # Select indices of remaining tokens
        unm_idx = edge_idx[..., :r, :]

    def merge(src: torch.Tensor) -> torch.Tensor:
        """
        Merges selected remaining tokens.

        Args:
            src (torch.Tensor): Source tensor of shape (batch_size, t, feature_dim).

        Returns:
            torch.Tensor: Merged tensor.
        """
        n, _, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, r, c))
        return unm.gather(dim=-2, index=unm_idx.argsort(dim=1).expand(n, r, c))

    return merge


# Test the function with random input
if __name__ == "__main__":
    t, feature_dim, r = 256, 512, 128
    metric = torch.randn(t, feature_dim)
    merge_fn = self_soft_matching(metric, r)
    test_tensor = torch.randn(t, feature_dim)
    merged_result = merge_fn(test_tensor)
    print("Merged Result:", merged_result.shape)
