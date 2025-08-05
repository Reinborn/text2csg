
import torch
from torch.utils.data import DataLoader
from dataset.csg_token_dataset import CSGTokenDataset
from model.csg_decoder import CSGTreeDecoder
from utils.tree_utils import build_tree_masks, structure_loss

class Config:
    max_surface_id = 1000
    max_parent_id = 1000
    max_depth = 20
    embed_dim = 64
    nhead = 8
    num_layers = 4
    ff_dim = 256
    output_dim = 64
    dropout = 0.1
    batch_size = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "checkpoints/final_model.pt"

def is_valid_tree(parent_ids):
    # 若某 token 的 parent_id >= token_id，则说明父亲在它后面，结构非法
    for i, pid in enumerate(parent_ids):
        if pid >= i and pid != -1:
            return False
    return True

def evaluate():
    dataset = CSGTokenDataset("tokens.npy")
    loader = DataLoader(dataset, batch_size=Config.batch_size)

    model = CSGTreeDecoder(Config()).to(Config.device)
    model.load_state_dict(torch.load(Config.model_path, map_location=Config.device))
    model.eval()

    ce_loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = total_ce = total_struct = 0.0
    valid_count = total_count = 0
    correct_parent = 0
    total_parent = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.long().to(Config.device)
            parent_ids = batch[..., 6]
            parent_mask, sibling_mask = build_tree_masks(parent_ids)

            logits = model(batch, parent_mask, sibling_mask)

            ce = ce_loss_fn(logits[..., 0].view(-1), batch[..., 0].view(-1))
            struct = structure_loss(logits, batch)
            loss = ce + struct

            total_loss += loss.item()
            total_ce += ce.item()
            total_struct += struct.item()

            pred_parent = logits[..., 6].argmax(dim=-1)
            correct_parent += (pred_parent == batch[..., 6]).sum().item()
            total_parent += batch[..., 6].numel()

            for i in range(batch.shape[0]):
                if is_valid_tree(pred_parent[i].tolist()):
                    valid_count += 1
                total_count += 1

    print("✅ Evaluation Results:")
    print(f"  Avg Total Loss:     {total_loss / len(loader):.4f}")
    print(f"  Avg CE Loss:        {total_ce / len(loader):.4f}")
    print(f"  Avg Struct Loss:    {total_struct / len(loader):.4f}")
    print(f"  Structure Accuracy: {correct_parent / total_parent:.4f}")
    print(f"  Invalid Ratio:      {1 - (valid_count / total_count):.4f}")

if __name__ == "__main__":
    evaluate()
