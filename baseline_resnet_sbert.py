"""
Baseline‑1: ResNet‑18 + SBERT (Hadamard & Diff Early‑Fusion)
"""
import argparse, json, math, random
from pathlib import Path

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm
import wandb
from PIL import Image

# --------------------------- Dataset ------------------------------------ #
class JsonlVQADataset(Dataset):
    def __init__(self, jsonl: Path, img_root: Path, tfms, a2i):
        self.recs = [json.loads(l) for l in open(jsonl, "r", encoding="utf8")]
        self.img_root, self.tfms, self.a2i = img_root, tfms, a2i
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        img = self.tfms(Image.open(self.img_root / r["image"]).convert("RGB"))
        return img, r["question"], self.a2i[r["answer"]], r["type"]

# ---------------------------- Model ------------------------------------- #
class ResNetSBERT(nn.Module):
    def __init__(self, num_ans, fusion="cat_mix_diff",
                 unfreeze_layer4=False, device="cuda"):
        super().__init__()
        assert fusion in {"cat", "cat_mix", "cat_mix_diff"}
        self.fusion = fusion
        # image encoder
        self.resnet = models.resnet18(
            weights = models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()
        self.resnet.eval(); self.resnet.requires_grad_(False)
        if unfreeze_layer4:
            for n, p in self.resnet.named_parameters():
                if n.startswith("layer4"): p.requires_grad = True
        # text encoder
        self.sbert = SentenceTransformer("all-mpnet-base-v2", device=device)
        self.sbert.requires_grad_(False)
        self.txt_proj = nn.Linear(768, 512, bias=False)

        feat_dim = 512 + 768 + (512 if fusion != "cat" else 0) \
                           + (512 if fusion == "cat_mix_diff" else 0)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_ans)
        )

    def forward(self, imgs, qs):
        img_f = self.resnet(imgs)                                   # (B,512)
        txt_raw = self.sbert.encode(qs, convert_to_tensor=True,
                                    device=imgs.device)             # (B,768)
        txt_512 = self.txt_proj(txt_raw)
        feats = [img_f, txt_raw]
        if self.fusion in {"cat_mix", "cat_mix_diff"}:
            feats.append(img_f * txt_512)
        if self.fusion == "cat_mix_diff":
            feats.append((img_f - txt_512).abs())
        return self.classifier(torch.cat(feats, 1))

# -------------------------- Evaluation ---------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); P, G, T = [], [], []
    for im, q, y, t in loader:
        P.extend(model(im.to(device), q).argmax(1).cpu())
        G.extend(y)
        T.extend(t)
    acc = accuracy_score(G, P)
    f1_macro = f1_score(G, P, average="macro")
    # per‑type F1
    type_f1 = {}
    for typ in {"property", "count", "relation"}:
        idx = [i for i, tt in enumerate(T) if tt == typ]
        if idx:
            y_sub = [G[i] for i in idx]
            p_sub = [P[i] for i in idx]
            type_f1[typ] = f1_score(y_sub, p_sub, average="macro")
    return acc, f1_macro, type_f1

# ----------------------- Warm‑up Cosine --------------------------------- #
class WarmCos:
    def __init__(self, opt, warm, total, base):
        self.o, self.w, self.tot, self.b = opt, warm, total, base
        self.t = 0
    def step(self):
        self.t += 1
        lr = (self.b * self.t / self.w if self.t <= self.w else
              0.5 * self.b * (1 + math.cos(math.pi *
                   (self.t - self.w) / (self.tot - self.w))))
        for g in self.o.param_groups: g["lr"] = lr
        self.o.step()

# ----------------------------- Main ------------------------------------- #
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--fusion", choices=["cat", "cat_mix", "cat_mix_diff"],
                   default="cat_mix_diff")
    p.add_argument("--unfreeze_layer4", action="store_true")
    p.add_argument("--device", default="cuda"
                   if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb_project", default="mini-clevr")
    p.add_argument("--wandb_run", default=None)
    return p.parse_args()

def main():
    a = parse()
    random.seed(0); torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    a2i = json.load(open(a.data_dir / "answer2idx.json"))
    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    tr_ds = JsonlVQADataset(a.data_dir / "train.jsonl", a.data_dir, tfms, a2i)
    va_ds = JsonlVQADataset(a.data_dir / "val.jsonl",   a.data_dir, tfms, a2i)
    tr_ld = DataLoader(tr_ds, a.batch, shuffle=True,  num_workers=0,
                       pin_memory=True)
    va_ld = DataLoader(va_ds, a.batch, shuffle=False, num_workers=0,
                       pin_memory=True)

    model = ResNetSBERT(len(a2i), a.fusion, a.unfreeze_layer4,
                        a.device).to(a.device)

    cls_p  = list(model.classifier.parameters())
    proj_p = list(model.txt_proj.parameters())
    ref_ids = {id(p) for p in cls_p + proj_p}

    other_p = [p for p in model.parameters()
            if p.requires_grad and id(p) not in ref_ids]

    opt = torch.optim.AdamW([
        {"params": cls_p,  "lr": 1e-3},
        {"params": proj_p, "lr": 1e-3},
        {"params": other_p,"lr": 1e-4}
    ], weight_decay=5e-3)
    sched = WarmCos(opt, len(tr_ld)*2, len(tr_ld)*a.epochs, 1e-3)

    run = wandb.init(project=a.wandb_project, name=a.wandb_run,
                     config=vars(a))
    best = 0; step = 0
    ckpt = Path("checkpoints/best.pt"); ckpt.parent.mkdir(exist_ok=True)

    for ep in range(1, a.epochs + 1):
        model.train(); ep_loss = 0
        for im, q, y, _ in tqdm(tr_ld, desc=f"Ep{ep}"):
            im, y = im.to(a.device), y.to(a.device)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(
                model(im, q), y, label_smoothing=0.05)
            loss.backward(); sched.step()
            ep_loss += loss.item() * im.size(0); step += 1
            wandb.log({"train_loss_step": loss.item()}, step=step)

        ep_loss /= len(tr_ds)
        acc, f1_macro, f1_type = evaluate(model, va_ld, a.device)
        log_dict = {"train_loss_epoch": ep_loss,
                    "val_acc": acc, "val_f1": f1_macro, "epoch": ep}
        log_dict.update({f"val_f1_{k}": v for k, v in f1_type.items()})
        wandb.log(log_dict)
        print(f"Ep{ep}: loss={ep_loss:.4f} acc={acc:.4f} f1={f1_macro:.4f}")

        if acc > best:
            best = acc
            torch.save(model.state_dict(), ckpt)
            art = wandb.Artifact("best-model", type="model")
            art.add_file(str(ckpt)); run.log_artifact(art)

    print("done. best val_acc", best); run.finish()

if __name__ == "__main__":
    main()
