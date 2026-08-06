# -*- coding: utf-8 -*-
# =====================================================================
# #14 MR-MKG — adaptation implementation
# Paper: "Multimodal Reasoning with Multimodal Knowledge Graph", ACL 2024.
# Protocol (team): Qwen2.5-VL-3B + Visual CoT-GQA, eval n=200.
# =====================================================================
# ADAPTATION NOTE (important): MR-MKG goc = RGAT (relation-aware GAT) tren
# Multimodal Knowledge Graph (MMKG) + knowledge adapter + cross-modal align
# loss + QLoRA, tren LLM co visual encoder rieng, dung scene graph VG.
# RGAT + soft-token injection + triplet loss nang va can sceneGraphs.json
# (KHONG attach tren Kaggle); truong `reasoning`/`thought` cua Visual CoT-GQA
# la gold program -> KHONG dung (gold-leaking). -> ADAPT (xem
# summaries/14_MR-MKG_verified.md §3): xay MMKG tu scene graph do VLM sinh ->
# RETRIEVE Top-N triplet lien quan cau hoi (buoc "sub-MMKG retrieval" cua
# paper) -> inject nhu knowledge context (knowledge adapter -> prompt) ->
# Qwen2.5-VL-3B tra loi VQA. Inference-only (reliable tren P100/T4 fp16).
# Faithful RGAT/adapter/triplet/QLoRA = future work (can sceneGraphs.json).
# =====================================================================


# %% [markdown]
# ## 0. Install dependencies (run once)

# %%
import subprocess, sys

# Kaggle thuong cap P100 (sm_60), nhung image PyTorch (>=2.5, cu126) DA BO sm_60
# -> moi CUDA op fail. Phat hien P100 qua nvidia-smi roi cai torch 2.4.1 (sm_60).
def _gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, text=True).stdout.strip().lower()
    except Exception:
        return ""
if "p100" in _gpu_name():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch==2.4.1", "torchvision==0.19.1",
                    "--index-url", "https://download.pytorch.org/whl/cu121"], check=False)
    print("P100 detected -> installed torch 2.4.1+cu121 (sm_60 support)")

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.45", "qwen-vl-utils", "peft", "bitsandbytes",
                "accelerate", "datasets", "scikit-learn", "pandas", "pillow", "kagglehub"],
               check=False)
print("deps ready")


# %% [markdown]
# ## 1. CONFIG

# %%
import os, re, time, json, random
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

SMOKE = True          # <-- True: smoke (n=4). False: full (n=200).

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
USE_4BIT  = True
CLIP_ID   = "openai/clip-vit-large-patch14"   # CLIP ViT-L (paper §3.2/§A.6). Luu y: ViT-L la patch14 (KHONG co patch32).

DATASET_HF      = "deepcs233/Visual-CoT"
GQA_TRAIN_JSONL = "cot_with_detailed_reasoning_steps/gqa_cot_train.jsonl"
GQA_VAL_JSONL   = "cot_with_detailed_reasoning_steps/gqa_cot_val.jsonl"
DATA_OUT_DIR    = "/kaggle/working/visual-cot"
GQA_IMG_CANDIDATES = [
    "{img}",
    "/kaggle/input/gqa-images/images/{img}",
    "/kaggle/working/visual-cot/cot/gqa/{img}",
    "/kaggle/input/visual-cot/cot/gqa/{img}",
    "/kaggle/input/test-dataset-visual-cot/visual-cot/cot/gqa/{img}",
]

# ---- MR-MKG / MMKG retrieval hyperparams ----
M_SEQ   = 2     # so scene-graph sequence sinh/anh (ASSUMPTION)
SG_LEN  = 120
TOP_P   = 0.9
TEMP    = 0.7
MAX_TRIP = 10   # kich thuoc MMKG max/anh
TOPN    = 5     # Top-N triplet lien quan retrieve (paper Top-N ∈ {10,20}; ASSUMPTION 5 cho compute)
DO_BASELINE = True

N_EVAL     = 200
OUTPUT_DIR = "./mrmkg_out"

if SMOKE:
    N_EVAL, M_SEQ, TOPN = 4, 1, 3
    print(f"!! SMOKE MODE -> N_EVAL={N_EVAL}, M_SEQ={M_SEQ}, TOPN={TOPN}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    _cap = torch.cuda.get_device_capability()[0]
    TORCH_DTYPE = torch.bfloat16 if _cap >= 8 else torch.float16
    USE_4BIT = USE_4BIT and (_cap >= 7)
    ATTN_IMPL = "sdpa" if _cap >= 7 else "eager"
else:
    TORCH_DTYPE = torch.float32
    USE_4BIT = False
    ATTN_IMPL = "eager"
print(f"GPU cap={torch.cuda.get_device_capability() if torch.cuda.is_available() else 'cpu'} "
      f"-> dtype={TORCH_DTYPE}, 4bit={USE_4BIT}, attn={ATTN_IMPL}")


# %% [markdown]
# ## 2. LOAD DATASET + GQA IMAGES

# %%
from datasets import Dataset
from huggingface_hub import hf_hub_download

def _find_data_file(rel_path):
    fname = os.path.basename(rel_path)
    roots = [r for r in ("/kaggle/input", "/kaggle/working", ".") if os.path.isdir(r)]
    for root in roots:
        cand = os.path.join(root, rel_path.lstrip("/"))
        if os.path.isfile(cand):
            return cand
    matches = []
    for root in roots:
        for p in Path(root).rglob(fname):
            if p.is_file():
                matches.append(str(p))
    matches.sort(key=lambda p: ("cot_with_detailed_reasoning_steps" not in p, len(p)))
    return matches[0] if matches else None

def _download_jsonl(filename):
    import time as _t
    last = None
    for attempt in range(6):
        try:
            return hf_hub_download(repo_id=DATASET_HF, repo_type="dataset",
                                   filename=filename, local_dir=DATA_OUT_DIR)
        except Exception as e:
            last = e; print(f"[retry {attempt+1}/6] {filename}: {repr(e)[:90]}"); _t.sleep(20*(attempt+1))
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    try:
        import huggingface_hub.constants as _hc
        _hc.ENDPOINT = "https://hf-mirror.com"; _hc.HF_HUB_ENDPOINT = "https://hf-mirror.com"
    except Exception:
        pass
    for attempt in range(3):
        try:
            return hf_hub_download(repo_id=DATASET_HF, repo_type="dataset",
                                   filename=filename, local_dir=DATA_OUT_DIR)
        except Exception as e:
            last = e; print(f"[mirror {attempt+1}/3] {filename}: {repr(e)[:90]}"); _t.sleep(15)
    raise RuntimeError(f"HF download failed: {filename}: {last!r}")

def load_jsonl(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs

def load_visual_cot_gqa():
    paths = {}
    for key, rel in (("train", GQA_TRAIN_JSONL), ("val", GQA_VAL_JSONL)):
        found = _find_data_file(rel)
        if found:
            print(f"[data] {key}: dung input -> {found}")
        else:
            print(f"[data] {key}: khong thay input, download tu HF ...")
            found = _download_jsonl(rel)
        paths[key] = found
    return load_jsonl(paths["train"]), load_jsonl(paths["val"])

train_recs, val_recs = load_visual_cot_gqa()
eval_recs = val_recs if val_recs else train_recs
print(f"train: {len(train_recs)} | val(eval): {len(eval_recs)}")

try:
    import kagglehub
    try:
        gqa_dir = kagglehub.dataset_download('lyte69/gqa-images')
        print('kagglehub gqa-images ->', gqa_dir)
    except Exception as e:
        print('kagglehub gqa-images skipped:', repr(e)[:120])
except Exception:
    pass
GQA_IMAGE_INDEX = {}
for root in ('/kaggle/input', '/kaggle/working/visual-cot', '/kaggle/working', '/root/.cache/kagglehub'):
    rp = Path(root)
    if not rp.exists():
        continue
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        for p in rp.rglob(ext):
            GQA_IMAGE_INDEX.setdefault(p.name, str(p))
print('GQA images indexed:', len(GQA_IMAGE_INDEX))


# %% [markdown]
# ## 3. MODEL (Qwen2.5-VL-3B, adaptive)

# %%
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

def _bnb():
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=TORCH_DTYPE,
                              bnb_4bit_use_double_quant=True) if USE_4BIT else None

print("Loading VLM (Qwen2.5-VL-3B) ...")
vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, quantization_config=_bnb(), torch_dtype=TORCH_DTYPE,
    device_map={"": 0}, attn_implementation=ATTN_IMPL)
vlm.eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
vtok = processor.tokenizer
print("VLM ready.")

# CLIP text encoder cho retrieval sub-MMKG (paper §A.6: cosine similarity tren CLIP space).
# Kaggle thuong fail download HF -> co fallback keyword retrieval.
from transformers import CLIPTextModel, CLIPTokenizer
CLIP_OK = False
_clip = _cliptok = None
for _ep in (None, "https://hf-mirror.com"):
    try:
        if _ep:
            os.environ["HF_ENDPOINT"] = _ep
            try:
                import huggingface_hub.constants as _hc
                _hc.ENDPOINT = _ep; _hc.HF_HUB_ENDPOINT = _ep
            except Exception:
                pass
        _clip = CLIPTextModel.from_pretrained(CLIP_ID, torch_dtype=TORCH_DTYPE).to(vlm.device).eval()
        _cliptok = CLIPTokenizer.from_pretrained(CLIP_ID)
        CLIP_OK = True
        print("CLIP text encoder ready (retrieval).")
        break
    except Exception as e:
        print(f"CLIP load attempt failed: {repr(e)[:110]}")
if not CLIP_OK:
    print("!! CLIP khong load duoc -> fallback keyword retrieval (it trung thuat hon paper §A.6).")

@torch.no_grad()
def _embed_text(texts):
    t = _cliptok(texts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(_clip.device)
    out = _clip(**t)
    f = out.pooler_output if (hasattr(out, "pooler_output") and out.pooler_output is not None) else out.last_hidden_state[:, 0]
    return torch.nn.functional.normalize(f.float(), dim=-1)


# %% [markdown]
# ## 4. MMKG construction + Top-N retrieval + knowledge-injected VQA
# # MR-MKG: xay MMKG (image-level) -> retrieve Top-N triplet lien quan cau hoi
# # (sub-MMKG) -> knowledge adapter (adapt thanh prompt context) -> VQA.

# %%
SG_PROMPT = (
    "Describe the scene graph of this image as a list of triplets. "
    "Each line MUST be exactly: 'subject - predicate - object'. "
    "Examples:\nman - riding - horse\ncup - on - table\nboy - holding - ball\n"
    f"List up to {MAX_TRIP} triplets. Output ONLY the triplets, one per line, no extra text."
)

@torch.no_grad()
def generate(prompt, image=None, max_new=96, do_sample=False, temperature=0.0, top_p=1.0):
    content = ([{"type": "image", "image": image}] if image is not None else []) + [{"type": "text", "text": prompt}]
    msgs = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs = [image] if image is not None else None
    inp = processor(text=[text], images=imgs, return_tensors="pt")
    inp = {k: v.to(vlm.device) for k, v in inp.items()}
    kw = dict(max_new_tokens=max_new)
    if do_sample:
        kw.update(do_sample=True, temperature=max(temperature, 1e-2), top_p=top_p)
    out = vlm.generate(**inp, **kw)
    gen = vtok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen.strip()

def parse_triples(text):
    triples = []
    for line in text.splitlines():
        line = line.strip().strip("-–—•").strip()
        if not line:
            continue
        m = re.split(r"\s+[-–—]\s+", line)
        if len(m) == 3 and all(len(x) <= 40 for x in m):
            s, p, o = (x.strip(" .,;") for x in m)
            if s and p and o and s.lower() != o.lower():
                triples.append((s.lower(), p.lower(), o.lower()))
    seen, out = set(), []
    for t in triples:
        if t not in seen:
            seen.add(t); out.append(t)
    return out[:MAX_TRIP]

def build_mmkg(image):
    """Sinh M_SEQ scene-graph sequence -> MMKG = tap triplet (entity, relation, entity)."""
    all_t = []
    for _ in range(M_SEQ):
        all_t += parse_triples(generate(SG_PROMPT, image=image, max_new=SG_LEN,
                                        do_sample=True, temperature=TEMP, top_p=TOP_P))
    seen, mmkg = set(), []
    for t in all_t:
        if t not in seen:
            seen.add(t); mmkg.append(t)
    return mmkg[:MAX_TRIP]

def retrieve_topn(mmkg, question, n=TOPN):
    """Sub-MMKG retrieval theo paper §A.6: CLIP cosine similarity question vs triple
    -> Top-n' candidate -> lay 1-hop neighbour (triple chia entity) -> rerank cosine -> Top-N.
    Neu CLIP khong load duoc -> fallback keyword overlap."""
    if not mmkg:
        return []
    if not CLIP_OK:
        qtoks = set(re.findall(r"[a-z0-9]+", question.lower())) - \
                {"the","a","an","is","are","of","in","on","at","to","what","who","where","how","this","that","with","and"}
        def _sc(t): return len(qtoks & (set(t[0].split()) | set(t[1].split()) | set(t[2].split())))
        return sorted(mmkg, key=_sc, reverse=True)[:n]
    trip_strs = [f"{s} {p} {o}" for (s, p, o) in mmkg]
    tt = _embed_text(trip_strs)          # (T, D)
    qt = _embed_text([question])         # (1, D)
    sims = (qt @ tt.T).squeeze(0).tolist()   # (T,)
    order = sorted(range(len(mmkg)), key=lambda i: -sims[i])
    nn = min(len(mmkg), max(n * 2, 6))   # Top-n' candidate
    cand = set(order[:nn])
    ents = set()
    for i in cand:
        ents |= set(mmkg[i][0].split()) | set(mmkg[i][2].split())
    # 1-hop: cand + cac triple chia entity voi cand
    hop = [i for i in range(len(mmkg))
           if i in cand or (set(mmkg[i][0].split()) & ents) or (set(mmkg[i][2].split()) & ents)]
    hop = sorted(set(hop), key=lambda i: -sims[i])
    return [mmkg[i] for i in hop[:n]]

def knowledge_ctx(triples):
    if not triples:
        return ""
    return "; ".join(f"{s} {p} {o}" for (s, p, o) in triples)

def resolve_image(img_field):
    if isinstance(img_field, Image.Image):
        return img_field.convert("RGB")
    name = img_field
    if isinstance(name, list) and name:
        name = name[0]
    name = str(name).split("###", 1)[0]
    if name in GQA_IMAGE_INDEX:
        return Image.open(GQA_IMAGE_INDEX[name]).convert("RGB")
    for tpl in GQA_IMG_CANDIDATES + [name]:
        p = tpl.format(img=name) if "{img}" in tpl else tpl
        if os.path.exists(p):
            return Image.open(p).convert("RGB")
    raise FileNotFoundError(f"Khong tim thay anh GQA: {name}")

def vqa(image, question, ctx=""):
    prefix = f"Relevant knowledge: {ctx}\n\n" if ctx else ""
    prompt = (f"{prefix}Look at the image and answer the question in 1-5 words only.\n"
              f"Question: {question}\nAnswer:")
    return generate(prompt, image=image, max_new=16, do_sample=False).strip().split("\n")[0].strip(" .,")


# %% [markdown]
# ## 5. EVALUATION

# %%
def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def macro_fpr_fnr(y_true, y_pred, labels):
    fprs, fnrs = [], []
    for c in labels:
        yt = [1 if y == c else 0 for y in y_true]
        yp = [1 if y == c else 0 for y in y_pred]
        try:
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            fprs.append(fp/(fp+tn) if (fp+tn) > 0 else 0)
            fnrs.append(fn/(fn+tp) if (fn+tp) > 0 else 0)
        except Exception:
            pass
    return float(np.mean(fprs)), float(np.mean(fnrs))

preds_kg, preds_base, golds, mmkg_sizes = [], [], [], []
n_eval = min(N_EVAL, len(eval_recs))
t0 = time.time()
for i in range(n_eval):
    try:
        rec = eval_recs[i]
        img = resolve_image(rec.get("image"))
        q = rec.get("question", "")
        gold = str(rec.get("answer", rec.get("full_answer", "")))
        mmkg = build_mmkg(img)
        mmkg_sizes.append(len(mmkg))
        sub = retrieve_topn(mmkg, q, n=TOPN)
        a_kg = vqa(img, q, ctx=knowledge_ctx(sub))
        preds_kg.append(a_kg); golds.append(gold)
        if DO_BASELINE:
            preds_base.append(vqa(img, q, ctx=""))
        if i % 25 == 0:
            print(f"[{i}/{n_eval}] gold={gold!r} kg={a_kg!r} mmkg={len(mmkg)} sub={knowledge_ctx(sub)[:70]!r}")
    except Exception as e:
        print(f"eval err {i}: {repr(e)[:120]}")
infer_time = time.time() - t0

kg_n   = [norm(p) for p in preds_kg]
gold_n = [norm(g) for g in golds]
labels = sorted(set(gold_n))
P, R, F1, _ = precision_recall_fscore_support(gold_n, kg_n, labels=labels, average="macro", zero_division=0)
acc = accuracy_score(gold_n, kg_n)
fpr, fnr = macro_fpr_fnr(gold_n, kg_n, labels)
acc_base = None
if DO_BASELINE and len(preds_base) == len(golds):
    acc_base = accuracy_score(gold_n, [norm(p) for p in preds_base])

params_total = sum(p.numel() for p in vlm.parameters())
print(f"\n=== RESULTS (#14 MR-MKG, Qwen2.5-VL-3B, Visual CoT-GQA, n={n_eval}) ===")
print(f"[knowledge-injected] Acc={acc:.4f} | Prec={P:.4f} | Recall={R:.4f} | F1={F1:.4f}")
print(f"FPR={fpr:.4f} | FNR={fnr:.4f} | ROC-AUC=N/A | PR-AUC=N/A | Hits@1=N/A (EM VQA, no rerank)")
if acc_base is not None:
    print(f"[baseline no-knowledge] Acc={acc_base:.4f}  (knowledge gain = {acc-acc_base:+.4f})")
print(f"Infer Time={infer_time:.1f}s | avg MMKG triples={np.mean(mmkg_sizes):.1f} | "
      f"Params={params_total} | Steps=0 (inference-only)")


# %% [markdown]
# ## 6. SAVE RESULTS

# %%
result = pd.DataFrame([{
    "STT": 14,
    "Paper": "MR-MKG (ACL 2024)",
    "Task Type": "Multimodal reasoning + MMKG -> adapt sang knowledge-injected VQA",
    "Method Type": "Build MMKG (VLM scene graph) -> retrieve Top-N question-relevant triples -> inject as knowledge context -> VQA",
    "Training Unit": "Inference-only (khong train); Qwen2.5-VL-3B lam VLM + MMKG builder",
    "Acc": round(acc, 4), "Prec": round(P, 4), "Recall": round(R, 4), "F1": round(F1, 4),
    "ROC-AUC": "N/A", "PR-AUC": "N/A", "FPR": round(fpr, 4), "FNR": round(fnr, 4),
    "Train Time": round(infer_time, 1), "Params": f"~3B (4bit={USE_4BIT}); inference-only",
    "Comm Cost": "N/A", "Training Steps": 0,
    "Note": (f"Qwen2.5-VL-3B; Visual CoT-GQA; eval={n_eval}; inference-only (Steps=0). "
             f"ADAPTATION: MR-MKG goc = RGAT tren MMKG + knowledge adapter + cross-modal align + QLoRA "
             f"(LLM+visual frozen, train ~2.25%), dung scene graph VG, ScienceQA/MARS. -> Adapt: "
             f"build MMKG tu scene graph do VLM sinh ({M_SEQ} seq, p={TOP_P}) -> retrieve Top-{TOPN} "
             f"triple bang CLIP cosine + 1-hop (paper §A.6; giong sub-MMKG retrieval) -> inject nhu knowledge "
             f"context (RGAT+adapter -> prompt, do KHONG co sceneGraphs.json va RGAT nang) -> VQA. "
             f"KHONG dung truong reasoning/thought (gold-leaking). "
             f"Knowledge-inj Acc={acc:.4f}" + (f"; baseline Acc={acc_base:.4f} (gain {acc-acc_base:+.4f})" if acc_base is not None else "") +
             f". Hits@1/ROC/PR-AUC=N/A (EM VQA). Faithful RGAT+triplet+QLoRA = future work (can GQA sceneGraphs.json). "
             f"adaptation, not full MR-MKG reproduction."),
}])
import os as _os; _os.makedirs(OUTPUT_DIR, exist_ok=True)
result.to_csv("result_14_MR-MKG.csv", index=False)
result.to_csv(_os.path.join(OUTPUT_DIR, "result_14_MR-MKG.csv"), index=False)
pd.DataFrame({"gold": gold_n, "pred_kg": kg_n}).to_csv("preds_14_MR-MKG.csv", index=False)
print(result.T)
print("\nSaved: result_14_MR-MKG.csv, preds_14_MR-MKG.csv")
print("=> Dien Acc/Prec/Recall/F1/FPR/FNR/Train Time/Params/Steps vao sheet 'Implement' dong 14.")
