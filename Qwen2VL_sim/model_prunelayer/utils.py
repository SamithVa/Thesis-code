import pdb
import torch


def get_select_mask(tensor, skip_ratio=0, rand=False):
    """"
    Args:
        tensor: patch_pos [bsz, seq_len]
            e.g [-1, -1, 1, 1, ..., 2, 3, -1, -1] 

    Return:
        select mask [bsz, seq_len], bool
    """
    # Use tensor operations for efficiency
    retain_mask = (tensor == -1).clone()
    unique_vals, counts = torch.unique(
        tensor, return_counts=True
    )  # default select all tokens

    for i, (val, count) in enumerate(zip(unique_vals, counts)):
        if val == -1:  # text tokens, must select !
            continue
        positions = (tensor == val).nonzero(as_tuple=True)[0]
        num_positions = len(positions)

        if num_positions == 1:  # if only a patch within the ui component, select it
            retain_mask[positions] = True
        else:
            num_to_skip = int(round(num_positions * skip_ratio))
            num_to_retain = max(1, num_positions - num_to_skip)
            if rand:
                # rand means random select subset of selective tokens for layer-wise
                perm = torch.randperm(num_positions, device=tensor.device)
                positions_to_retain = positions[perm[:num_to_retain]]
            else:
                indices = torch.linspace(
                    0, num_positions - 1, steps=num_to_retain
                ).long()
                positions_to_retain = positions[indices]

            retain_mask[positions_to_retain] = True
    return retain_mask


# if __name__ == "__main__":
#     # Create a random tensor with values in the range [-1, 3]
#     # Here, -1 represents tokens that must always be selected.
#     tensor = torch.tensor([-1, -1, 2, 2, 2, 2, -1, -1])
#     print("Input tensor:")
#     print(tensor)

#     # Generate the selection mask with skip_ratio=0.5 and random selection enabled.
#     mask = get_select_mask(tensor, skip_ratio=0.5, rand=True)
#     print("Selection mask:")
#     print(mask)
