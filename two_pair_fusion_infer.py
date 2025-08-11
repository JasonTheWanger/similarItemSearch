import argparse, torch, torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import open_clip
from sentence_transformers import SentenceTransformer

class ImgEncoder(torch.nn.Module):
    def __init__(self, clip_model, emb_dim=256):
        super().__init__()
        self.backbone = clip_model.visual
        in_dim = clip_model.visual.output_dim
        self.head = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.GELU(),
            torch.nn.Linear(512, emb_dim)
        )
    def forward(self, x):
        z = self.head(self.backbone(x))
        return F.normalize(z, dim=1)

def get_val_tf(img_size=224):
    return T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
    ])

@torch.no_grad()
def embed_image(path, encoder, tfm, device):
    x = tfm(Image.open(path).convert("RGB")).unsqueeze(0).to(device, dtype=torch.float32)
    return encoder(x)  # [1,256], L2-normalized

@torch.no_grad()
def embed_text(text, sbert, device):
    z = sbert.encode([text], convert_to_tensor=True, device=device, normalize_embeddings=True)
    return z  # [1,384], L2-normalized

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img1", required=True)
    ap.add_argument("--text1", required=True)
    ap.add_argument("--img2", required=True)
    ap.add_argument("--text2", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--sbert_model", default="all-MiniLM-L6-v2")
    ap.add_argument("--w_img", type=float, default=0.60)
    ap.add_argument("--th", type=float, default=0.54)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    # load image encoder
    clip_model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    for p in clip_model.parameters(): p.requires_grad = False
    img_enc = ImgEncoder(clip_model, emb_dim=256).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("model"))
    img_enc.load_state_dict(state, strict=True)
    img_enc.eval()

    # load SBERT
    sbert_device = "cuda" if torch.cuda.is_available() else "cpu"
    sbert = SentenceTransformer(args.sbert_model, device=sbert_device); sbert.max_seq_length = 64

    tfm = get_val_tf(args.img_size)

    # embeddings
    z1_img = embed_image(args.img1, img_enc, tfm, device)
    z2_img = embed_image(args.img2, img_enc, tfm, device)
    z1_txt = embed_text(args.text1, sbert, sbert_device)
    z2_txt = embed_text(args.text2, sbert, sbert_device)

    # cosine similarities
    sim_img = F.cosine_similarity(z1_img, z2_img).item()
    sim_txt = F.cosine_similarity(z1_txt.to(device), z2_txt.to(device)).item()

    # score-level fusion
    fused_sim = args.w_img * sim_img + (1.0 - args.w_img) * sim_txt
    is_match = fused_sim >= args.th

    print(f"image-only sim : {sim_img:.4f}")
    print(f"text-only  sim : {sim_txt:.4f}")
    print(f"FUSED sim       : {fused_sim:.4f}  (w_img={args.w_img:.2f}, th={args.th:.2f}) -> match={is_match}")

if __name__ == "__main__":
    main()
