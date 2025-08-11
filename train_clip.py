# train_clip.py
import os, math, argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import open_clip

from dataset_pk import ShopeeImgDataset, PKSampler
from losses import ArcMarginProduct
from eval_utils import build_true_sets, cosine_topk, sweep_threshold, apply_threshold, micro_f1_kaggle

# --- SBERT fusion helpers ---
from sbert_utils import load_sbert, encode_titles
from fusion_utils import cosine_topk_from_embs, fuse_vals_by_image_candidates

# -------------- Model wrapper --------------
class ImgEncoder(nn.Module):
    def __init__(self, clip_model, emb_dim=256):
        super().__init__()
        self.backbone = clip_model.visual
        in_dim = clip_model.visual.output_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, emb_dim)
        )

    def forward(self, x):
        feats = self.backbone(x)             # [B, in_dim]
        z = self.head(feats)                 # [B, emb_dim]
        return nn.functional.normalize(z, dim=1)

def get_transforms(img_size=224):
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.05),
        T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),  # CLIP norm
    ])

def get_val_tf(img_size=224):
    return T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
    ])

@torch.no_grad()
def embed_split(model, loader, device):
    model.eval()
    embs, labels = [], []
    for x, y, _ in loader:
        x = x.to(device, dtype=torch.float32)
        z = model(x)
        embs.append(z.cpu())
        labels.extend(int(yy) for yy in y)  # ensure plain ints
    return torch.cat(embs), labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="shopee-product-matching/train_groups.csv")
    ap.add_argument("--val_csv",   default="shopee-product-matching/val_groups.csv")
    ap.add_argument("--test_csv",  default="shopee-product-matching/test_groups.csv")
    ap.add_argument("--img_root",  default="shopee-product-matching/train_images")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--out", default="checkpoints")
    # --- Fusion flags ---
    ap.add_argument("--fuse", action="store_true", help="Enable SBERT late fusion")
    ap.add_argument("--sbert_model", default="all-MiniLM-L6-v2")
    ap.add_argument("--wimg_grid", default="0.60,0.70,0.80,0.85,0.90,0.95",
                    help="Comma list for w_img grid search on VAL")
    ap.add_argument("--text_col", default="title")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    # ----- data -----
    df_tr = pd.read_csv(args.train_csv)
    df_va = pd.read_csv(args.val_csv)
    df_te = pd.read_csv(args.test_csv)

    if args.fuse:
        assert args.text_col in df_va.columns and args.text_col in df_te.columns, \
            f"Missing '{args.text_col}' in val/test CSVs for fusion."

    # label mapping (ArcFace expects labels in [0, C-1])
    uniq = sorted(df_tr["label_group"].unique().tolist())
    cls2id = {c: i for i, c in enumerate(uniq)}
    df_tr["y"] = df_tr["label_group"].map(cls2id)

    t_train = get_transforms(args.img_size)
    t_val   = get_val_tf(args.img_size)

    train_ds = ShopeeImgDataset(df_tr, img_root=args.img_root, transform=t_train)
    val_ds   = ShopeeImgDataset(df_va, img_root=args.img_root, transform=t_val)
    test_ds  = ShopeeImgDataset(df_te, img_root=args.img_root, transform=t_val)

    pin = False if device.type == "mps" else True
    train_sampler = PKSampler(train_ds.labels, P=args.P, K=args.K)
    train_loader  = DataLoader(train_ds, batch_size=args.batch, sampler=train_sampler,
                               num_workers=4, pin_memory=pin, drop_last=True)
    val_loader    = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                               num_workers=4, pin_memory=pin)
    test_loader   = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                               num_workers=4, pin_memory=pin)

    # ----- model -----
    clip_model, _, _ = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device
    )
    for p in clip_model.parameters():
        p.requires_grad = False

    model = ImgEncoder(clip_model, emb_dim=256).to(device)

    # ArcFace head
    num_classes = len(uniq)
    arc = ArcMarginProduct(emb_dim=256, num_classes=num_classes, s=30.0, m=0.2).to(device)
    ce  = nn.CrossEntropyLoss()

    # only train head + arc at start
    opt = torch.optim.AdamW(list(model.head.parameters()) + list(arc.parameters()),
                            lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val_f1, best_th, best_path = 0.0, 0.5, None

    # ----- training -----
    steps_per_epoch = math.ceil(len(df_tr) / args.batch)
    warmup_epochs = 1

    for epoch in range(1, args.epochs + 1):
        model.train()
        if epoch == warmup_epochs + 1:
            # unfreeze last N blocks of CLIP visual encoder for fine-tuning (ViT-B/32)
            for n, p in clip_model.visual.named_parameters():
                if any(k in n for k in ["transformer.resblocks.9",
                                        "transformer.resblocks.10",
                                        "transformer.resblocks.11"]):
                    p.requires_grad = True
            opt = torch.optim.AdamW(
                list(model.parameters()) + list(arc.parameters()),
                lr=args.lr, weight_decay=args.wd
            )
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs - epoch + 1)

        running = 0.0
        for step, (x, y, _) in enumerate(train_loader, start=1):
            x = x.to(device, dtype=torch.float32)
            y = torch.tensor([cls2id[int(yy)] for yy in y], device=device, dtype=torch.long)

            z = model(x)                 # [B, 256] normalized
            logits = arc(z, y)           # [B, C]
            loss = ce(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += loss.item()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{steps_per_epoch} loss {running/step:.4f}")

            if step >= steps_per_epoch:  # cap epoch length for infinite sampler
                break

        sched.step()

        # ----- eval on VAL -----
        val_embs, val_labels = embed_split(model, val_loader, device)
        val_embs = nn.functional.normalize(val_embs, dim=1)

        true_sets_full = build_true_sets(val_labels)
        mask = [i for i, s in enumerate(true_sets_full) if len(s) > 0]
        if len(mask) == 0:
            print("[VAL] Warning: no multi-item groups in val; cannot compute meaningful F1.")
            f1, th = 0.0, 0.5
        else:
            vals_all, idxs_all = cosine_topk(val_embs, topk=50)
            vals = vals_all[mask]
            idxs = idxs_all[mask]
            true_sets = [true_sets_full[i] for i in mask]
            f1, th = sweep_threshold(vals, idxs, true_sets, sweep=np.linspace(0.0, 0.95, 40))

        print(f"[VAL] epoch {epoch} F1={f1:.4f} @th={th:.2f}")

        # save-best checkpoint (.pth) + remember best_th
        if f1 > best_val_f1:
            best_val_f1, best_th = f1, th
            best_path = os.path.join(args.out, f"best_epoch{epoch}_f1{f1:.4f}.pth")
            try:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "arc_state_dict": arc.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": sched.state_dict(),
                    "cls2id": cls2id,
                    "best_f1": best_val_f1,
                    "best_th": best_th
                }, best_path)
                print(f"[SAVE] Best model updated at epoch {epoch} -> {best_path}")
            except Exception as e:
                print(f"[SAVE] warning: failed to save checkpoint: {e}")

    # ----- final TEST using best val threshold -----
    # load best checkpoint if available
    if best_path and os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        best_th = ckpt.get("best_th", best_th)

    # IMAGE embeddings
    test_embs, test_labels = embed_split(model, test_loader, device)
    test_embs = nn.functional.normalize(test_embs, dim=1)
    t_true_full = build_true_sets(test_labels)
    t_mask = [i for i, s in enumerate(t_true_full) if len(s) > 0]

    if len(t_mask) == 0:
        print("[TEST] Warning: no multi-item groups in test; F1 is undefined. Check your split.")
        return

    t_vals_all, t_idxs_all = cosine_topk(test_embs, topk=50)
    t_vals_img = t_vals_all[t_mask]
    t_idxs_img = t_idxs_all[t_mask]
    t_true     = [t_true_full[i] for i in t_mask]

    # test with VAL-chosen threshold (competition-style)
    preds = apply_threshold(t_vals_img, t_idxs_img, best_th)
    test_f1 = micro_f1_kaggle(preds, t_true)
    print(f"[TEST][image-only] micro-F1={test_f1:.4f} (val-th={best_th:.2f})")

    # also show best-possible test F1 by sweeping on test (image-only)
    test_f1_best, test_best_th = sweep_threshold(t_vals_img, t_idxs_img, t_true, sweep=np.linspace(0.0, 0.95, 40))
    print(f"[TEST][image-only] micro-F1-best={test_f1_best:.4f} (best_th={test_best_th:.2f})")

    if not args.fuse:
        return

    # ----- SBERT late fusion -----
    print("[FUSION] Building SBERT embeddings for VAL/TEST...")
    sbert_device = "cuda" if torch.cuda.is_available() else "cpu"
    sbert = load_sbert(args.sbert_model, device=sbert_device)

    # VAL image embeddings (recompute once here using best model)
    val_embs, val_labels = embed_split(model, val_loader, device)
    val_embs = nn.functional.normalize(val_embs, dim=1)
    v_vals_img, v_idxs_img = cosine_topk(val_embs, topk=50)
    v_true_full = build_true_sets(val_labels)
    v_mask = [i for i, s in enumerate(v_true_full) if len(s) > 0]
    v_vals_img = v_vals_img[v_mask]; v_idxs_img = v_idxs_img[v_mask]
    v_true = [v_true_full[i] for i in v_mask]

    # SBERT text embeddings (VAL & TEST)
    v_txt = encode_titles(df_va.reset_index(drop=True), sbert, batch_size=512, text_col=args.text_col)
    t_txt = encode_titles(df_te.reset_index(drop=True), sbert, batch_size=512, text_col=args.text_col)

    # Cosine top-k in text space
    v_vals_txt, v_idxs_txt = cosine_topk_from_embs(v_txt, topk=50)
    t_vals_txt, t_idxs_txt = cosine_topk_from_embs(t_txt, topk=50)

    # Align to non-singleton masks
    v_vals_txt = v_vals_txt[v_mask]; v_idxs_txt = v_idxs_txt[v_mask]
    t_vals_txt = t_vals_txt[t_mask]; t_idxs_txt = t_idxs_txt[t_mask]

    # Grid-search w_img on VAL (and sweep threshold) to maximize F1
    best_f1_fused, best_wimg, best_th_fused = 0.0, None, None
    wgrid = [float(x) for x in args.wimg_grid.split(",")]
    for w in wgrid:
        v_vals_fused, v_idxs_fused = fuse_vals_by_image_candidates(v_vals_img, v_idxs_img, v_vals_txt, v_idxs_txt, w_img=w)
        f1_fused, th_fused = sweep_threshold(v_vals_fused, v_idxs_fused, v_true, sweep=np.linspace(0.0, 0.95, 40))
        if f1_fused > best_f1_fused:
            best_f1_fused, best_wimg, best_th_fused = f1_fused, w, th_fused

    print(f"[VAL][fusion] best F1={best_f1_fused:.4f} @ w_img={best_wimg:.2f}, th={best_th_fused:.2f}")

    # Apply best (w_img, th) to TEST
    t_vals_fused, t_idxs_fused = fuse_vals_by_image_candidates(t_vals_img, t_idxs_img, t_vals_txt, t_idxs_txt, w_img=best_wimg)
    preds_fused = apply_threshold(t_vals_fused, t_idxs_fused, best_th_fused)
    test_f1_fused = micro_f1_kaggle(preds_fused, t_true)
    print(f"[TEST][fusion] micro-F1={test_f1_fused:.4f} (w_img={best_wimg:.2f}, th={best_th_fused:.2f})")

if __name__ == "__main__":
    main()
