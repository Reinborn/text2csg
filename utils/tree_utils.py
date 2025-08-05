
import torch
import torch.nn.functional as F

def build_tree_masks(parent_ids):
    """
    parent_ids: (B, L) tensor, where each token has a parent index or -1
    Returns:
        parent_mask: (B, L, L) where parent_mask[b, i, j] = 1 if j is parent of i
        sibling_mask: (B, L, L) where sibling_mask[b, i, j] = 1 if i and j share same parent
    """
    B, L = parent_ids.shape
    parent_mask = torch.zeros(B, L, L, dtype=torch.bool, device=parent_ids.device)
    sibling_mask = torch.zeros(B, L, L, dtype=torch.bool, device=parent_ids.device)

    for b in range(B):
        for i in range(L):
            p_i = parent_ids[b, i].item()
            if p_i >= 0 and p_i < L:
                parent_mask[b, i, p_i] = 1
                for j in range(L):
                    if i != j and parent_ids[b, j].item() == p_i:
                        sibling_mask[b, i, j] = 1
    return parent_mask, sibling_mask

def structure_loss(pred, target, lambda_parent=0.4, lambda_sibling=0.4, lambda_syntax=0.2):
    """
    pred: predicted logits (B, L, D)
    target: ground truth tokens (B, L, D)
    """
    L_parent = F.l1_loss(pred[..., 6].float(), target[..., 6].float())  # parent_id
    L_sibling = F.l1_loss(pred[..., 7].float(), target[..., 7].float())  # depth
    L_syntax = F.cross_entropy(pred[..., 0].long().view(-1), target[..., 0].long().view(-1))  # type_id
    return lambda_parent * L_parent + lambda_sibling * L_sibling + lambda_syntax * L_syntax

def rampup_weight(step, warmup_steps, ramp_steps):
    if step < warmup_steps:
        return 0.0
    elif step < warmup_steps + ramp_steps:
        return (step - warmup_steps) / ramp_steps
    else:
        return 1.0
