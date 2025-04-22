"""
Baseline-2: CLIP ViT-B/32 + LoRA (on last two Attention.proj & MLP)
"""

import argparse, json, math, random, warnings
from pathlib import Path

import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import open_clip
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
import wandb
from peft import get_peft_model, LoraConfig

warnings.filterwarnings("ignore", category=FutureWarning,
        message="Importing from timm.models.layers")

# ---------------------------- Dataset ------------------------------------ #
class JsonlClipDataset(Dataset):
    def __init__(self, jsonl: Path, img_root: Path, tfms, tokenizer, ans2idx):
        self.recs = [json.loads(l) for l in open(jsonl, "r", encoding="utf8")]
        self.img_root, self.tfms, self.tok, self.a2i = img_root, tfms, tokenizer, ans2idx
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        img = self.tfms(Image.open(self.img_root / r["image"]).convert("RGB"))
        txt = self.tok([r["question"]])[0]
        return img, txt, self.a2i[r["answer"]]

# ---------------------------- Model -------------------------------------- #
class ClipLoRAMLP(nn.Module):
    def __init__(self, clip_model, fusion, n_cls):
        super().__init__()
        assert fusion in {"cat","cat_mix","cat_mix_diff"}
        self.clip = clip_model.eval()
        self.fusion = fusion
        d = clip_model.visual.output_dim          # 512
        feat = d*2 + (d if fusion!="cat" else 0) + (d if fusion=="cat_mix_diff" else 0)
        self.bn  = nn.BatchNorm1d(feat)
        self.mlp = nn.Sequential(
            nn.Linear(feat,512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512,n_cls)
        )
    def forward(self, imgs, toks):
        vf = self.clip.encode_image(imgs)
        tf = self.clip.encode_text(toks)
        feats=[vf,tf]
        if self.fusion in {"cat_mix","cat_mix_diff"}: feats.append(vf*tf)
        if self.fusion=="cat_mix_diff": feats.append((vf-tf).abs())
        return self.mlp(self.bn(torch.cat(feats,1)))

# ----------------------- Eval ------------------------------------------- #
@torch.no_grad()
def evaluate(m,loader,dev):
    m.eval(); P,G=[],[]
    for im,tx,y in loader:
        P.extend(m(im.to(dev),tx.to(dev)).argmax(1).cpu().tolist()); G.extend(y.tolist())
    return accuracy_score(G,P)

# ----------------------- Warmup‑Cosine ---------------------------------- #
class WarmupCos:
    def __init__(self,opt,warm,total,base): self.o,self.w,self.tot,self.b=opt,warm,total,base; self.s=0
    def step(self):
        self.s+=1
        lr=self.b*self.s/self.w if self.s<=self.w else 0.5*self.b*(1+math.cos(math.pi*(self.s-self.w)/(self.tot-self.w)))
        for g in self.o.param_groups:g["lr"]=lr
        self.o.step()

# ----------------------------- Main ------------------------------------- #
def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--data_dir",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=15)
    p.add_argument("--batch",type=int,default=128)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--fusion",choices=["cat","cat_mix","cat_mix_diff"],default="cat_mix_diff")
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb_project",default="mini-clevr"); p.add_argument("--wandb_run",default=None)
    return p.parse_args()

def main():
    a=parse(); random.seed(0); torch.manual_seed(0)
    # --- CLIP base ---
    clip, tfms, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tok = open_clip.get_tokenizer("ViT-B-32")

    # --- auto‑detect Linear names in last 2 blocks ---
    lin_names=[]
    for n,m in clip.named_modules():
        if isinstance(m, nn.Linear) and (".10." in n or ".11." in n):
            lin_names.append(n.split('.')[-1])   # take leaf name like proj / c_fc / c_proj
    target_modules=list(set(lin_names))          # unique

    lora_cfg=LoraConfig(r=8,lora_alpha=32,target_modules=target_modules)
    clip=get_peft_model(clip,lora_cfg).eval()

    # --- data ---
    ans2idx=json.load(open(a.data_dir/'answer2idx.json'))
    tr=JsonlClipDataset(a.data_dir/'train.jsonl',a.data_dir,tfms,tok,ans2idx)
    va=JsonlClipDataset(a.data_dir/'val.jsonl',  a.data_dir,tfms,tok,ans2idx)
    tr_ld=DataLoader(tr,a.batch,shuffle=True ,num_workers=0,pin_memory=True)
    va_ld=DataLoader(va,a.batch,shuffle=False,num_workers=0,pin_memory=True)

    model=ClipLoRAMLP(clip,a.fusion,len(ans2idx)).to(a.device)

    # param groups
    lora_p=[p for n,p in model.named_parameters() if "lora_" in n and p.requires_grad]
    opt=torch.optim.AdamW([
        {"params": model.mlp.parameters(), "lr":1e-3},
        {"params": lora_p,                "lr":2e-4}
    ], weight_decay=1e-2)
    sched=WarmupCos(opt,len(tr_ld)*2,len(tr_ld)*a.epochs,a.lr)

    run=wandb.init(project=a.wandb_project,name=a.wandb_run,config=vars(a))
    best=0; step=0; ckpt=Path("checkpoints/clip_lora_best.pt"); ckpt.parent.mkdir(exist_ok=True)
    for ep in range(1,a.epochs+1):
        model.train(); ep_loss=0
        for im,tx,y in tqdm(tr_ld,desc=f"Ep{ep}"):
            im,tx,y=im.to(a.device),tx.to(a.device),y.to(a.device)
            opt.zero_grad(); loss=nn.functional.cross_entropy(model(im,tx),y,label_smoothing=0.1)
            loss.backward(); sched.step()
            ep_loss+=loss.item()*im.size(0); step+=1
            wandb.log({"train_loss_step":loss.item()},step=step)
        ep_loss/=len(tr)
        acc=evaluate(model,va_ld,a.device)
        wandb.log({"train_loss_epoch":ep_loss,"val_acc":acc,"epoch":ep})
        print(f"Ep{ep}: loss={ep_loss:.4f}  val={acc:.4f}")
        if acc>best:
            best=acc; torch.save(model.state_dict(),ckpt)
            art=wandb.Artifact("clip-lora-best",type="model"); art.add_file(str(ckpt)); run.log_artifact(art)
    print(" done. best", best); run.finish()

if __name__=="__main__":
    main()