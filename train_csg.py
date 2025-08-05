
import torch
from torch.utils.data import DataLoader
from dataset.csg_token_dataset import CSGTokenDataset
from model.csg_decoder import CSGTreeDecoder
from utils.tree_utils import build_tree_masks, structure_loss, rampup_weight

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
    warmup_steps = 20000
    ramp_steps = 10000
    max_steps = 200000
    batch_size = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = "checkpoints"

def train():
    dataset = CSGTokenDataset("tokens.npy")
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    model = CSGTreeDecoder(Config()).to(Config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce_loss_fn = torch.nn.CrossEntropyLoss()

    global_step = 0
    model.train()

    while global_step < Config.max_steps:
        for batch in loader:
            batch = batch.to(Config.device).long()
            parent_ids = batch[..., 6]
            parent_mask, sibling_mask = build_tree_masks(parent_ids)
            logits = model(batch, parent_mask, sibling_mask)

            ce_loss = ce_loss_fn(logits[..., 0].view(-1), batch[..., 0].view(-1))
            alpha = rampup_weight(global_step, Config.warmup_steps, Config.ramp_steps)
            struct = structure_loss(logits, batch)
            loss = ce_loss + alpha * struct

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 1000 == 0:
                print(f"Step {global_step}, Loss: {loss.item():.4f}, CE: {ce_loss.item():.4f}, Struct: {struct.item():.4f}")

            if global_step >= Config.max_steps:
                break

    print("✅ Training complete.")
    torch.save(model.state_dict(), f"{Config.save_path}/final_model.pt")

if __name__ == "__main__":
    train()
