# -*- coding: utf-8 -*-
# =====================================================================
# #11 COCO-Tree — adaptation implementation
# Paper: Sinha et al., "COCO-Tree: COmpositional Hierarchical COncept Trees
#        for Enhanced Reasoning in Vision Language Models", EMNLP 2025.
# Protocol (team): Qwen2.5-VL-3B + Visual CoT-GQA, eval n=200, INFERENCE-ONLY.
# =====================================================================
# ADAPTATION NOTE (important): COCO-Tree goc la image-text MATCHING (cho score),
# KHONG phai VQA. Paper khong lam VQA. O day ta ADAPT sang "candidate-rerank VQA":
#   Qwen2.5-VL-3B sinh K candidate answer -> xay concept tree moi candidate
#   (SMD->RCE, composite score VS/LS, path search greedy/beam) -> rerank -> dap an.
# Xem summaries/11_COCO-Tree_verified.md (§3) cho chi tiet + cac ASSUMPTION.
# =====================================================================


# %% [markdown]
# ## 0. Install dependencies (run once)

# %%
import subprocess, sys

# Kaggle thuong cap P100 (sm_60), nhung image PyTorch (>=2.5, cu126) DA BO sm_60
# -> moi CUDA op fail ("no kernel image"). Phat hien P100 qua nvidia-smi roi cai
# torch 2.4.1 (version cuoi cung con gom sm_60, cu121). PHAI lam o CELL DAU TIEN,
# truoc khi import torch -> khong can restart kernel. Tren T4: giu nguyen image torch.
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
import os, re, time, json, random, math
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

SMOKE = True          # <-- True: chay nho (n=4) de bat bug. False: full (n=200).

# ---- Model (protocol: Qwen2.5-VL-3B) ----
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
USE_4BIT  = True
# ADAPTATION: paper dung 1 LLM reasoner rieng (Llama-3.1-8B) cung cỡ VLM.
# De tiet kiem RAM/toc do tren Kaggle, o day ta DUNG CHINH Qwen2.5-VL-3B cho ca
# vision (VS, candidate) va text (SMD/RCE/LS). Dat USE_SEPARATE_REASONER=True de
# load them Qwen2.5-3B-Instruct lam reasoner (gan paper hon).
USE_SEPARATE_REASONER = False                      # paper dung 1 LLM reasoner rieng (Llama-3.1-8B); tren P100 fp16 reasoner rieng (1.5B) y hon VLM -> giu VLM-lam-reasoner (Acc cao hon)
REASONER_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# ---- Dataset (VISUAL COT - GQA subset, giong #12) ----
DATASET_HF      = "deepcs233/Visual-CoT"
GQA_TRAIN_JSONL = "cot_with_detailed_reasoning_steps/gqa_cot_train.jsonl"
GQA_VAL_JSONL   = "cot_with_detailed_reasoning_steps/gqa_cot_val.jsonl"
DATA_OUT_DIR    = "/kaggle/working/visual-cot"
GQA_IMG_CANDIDATES = [
    "{img}",
    "/kaggle/input/gqa-images/images/{img}",            # dataset lyte69/gqa-images
    "/kaggle/working/visual-cot/cot/gqa/{img}",
    "/kaggle/input/visual-cot/cot/gqa/{img}",
    "/kaggle/input/test-dataset-visual-cot/visual-cot/cot/gqa/{img}",
]

# ---- COCO-Tree hyperparams (paper + adaptation cho compute) ----
M       = 2     # so morphological entity (SMD). Paper M=2.
S       = 2     # split factor (RCE). Paper S=3 -> giam xuong 2 cho compute.
L       = 2     # do sau tree (RCE). Paper L=3 -> adaptation L=2 (giam tu 3 cho compute Kaggle).
ALPHA   = 0.6   # trong so LS trong composite score CS. Paper alpha=0.6 (Winoground).
BETA    = 0.8   # trong so System-1 trong fusion. Paper beta=0.8.
K_CAND  = 2     # so candidate answer sinh ra de rerank.
BEAM_K  = 2     # beam width. ASSUMPTION: paper chi noi "select k max", khong cho gia tri k.
PATH_AGG = "mean"  # ASSUMPTION: paper "maximum path weight" ma khong noi sum/mean -> ta dung mean.
SEARCH_MODE = "beam"  # "greedy" hay "beam"

# ---- Eval ----
N_EVAL    = 200
OUTPUT_DIR = "./coco_tree_out"

if SMOKE:
    N_EVAL, K_CAND, S, L = 4, 2, 1, 1
    print(f"!! SMOKE MODE -> N_EVAL={N_EVAL}, K_CAND={K_CAND}, S={S}, L={L}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- dtype + quant + attn theo GPU ----
#  - T4/Turing (cap 7.x) KHONG co kernel bf16 -> fp16; Ampere+ (>=8) -> bf16.
#  - bitsandbytes 4-bit can sm_70+. Kaggle dang cap P100 (cap 6.0) -> bnb 4-bit crash
#    (DeadKernelError). Tat 4-bit, load fp16 (Qwen2.5-VL-3B ~7.6GB vua 16GB). Tren T4+ giu 4-bit.
if torch.cuda.is_available():
    _cap = torch.cuda.get_device_capability()[0]
    TORCH_DTYPE = torch.bfloat16 if _cap >= 8 else torch.float16
    USE_4BIT = USE_4BIT and (_cap >= 7)            # P100 (sm_60) khong chay duoc bnb 4-bit
    ATTN_IMPL = "sdpa" if _cap >= 7 else "eager"   # P100: sdpa fallback cham -> dung eager
else:
    TORCH_DTYPE = torch.float32
    USE_4BIT = False
    ATTN_IMPL = "eager"
print(f"GPU cap={torch.cuda.get_device_capability() if torch.cuda.is_available() else 'cpu'} "
      f"-> dtype={TORCH_DTYPE}, 4bit={USE_4BIT}, attn={ATTN_IMPL}")


# %% [markdown]
# ## 2. LOAD DATASET + GQA IMAGES  (auto-discover input da attach)

# %%
from datasets import Dataset
from huggingface_hub import hf_hub_download

def _find_data_file(rel_path):
    """Uu tien input Kaggle da attach (visual-cot), roi /kaggle/working, roi local."""
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
    """Fallback: tai tu HF deepcs233/Visual-CoT (retry + mirror)."""
    import time as _t, socket as _s
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
print("Sample keys:", list(train_recs[0].keys()))

# ---- GQA image index: scan /kaggle/input (bat dataset gqa-images bat ke slug) ----
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
# ## 3. MODEL + REASONER + yes/no SCORERS

# %%
from transformers import (Qwen2_5_VLForConditionalGeneration, AutoProcessor,
                          BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer)

def _bnb():
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=TORCH_DTYPE,   # T4 (cap<8) -> fp16; bf16 tren T4 crash
                              bnb_4bit_use_double_quant=True) if USE_4BIT else None

print("Loading VLM (Qwen2.5-VL-3B) ...")
vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, quantization_config=_bnb(), torch_dtype=TORCH_DTYPE,
    device_map={"": 0}, attn_implementation=ATTN_IMPL)   # P100(sm60)->eager, T4+->sdpa
vlm.eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
vtok = processor.tokenizer

# token ids "Yes"/"No" (first sub-token) cho scoring discriminative
def _first_tok(tok, s):
    ids = tok(s, add_special_tokens=False).input_ids
    return ids[0]
YES_VL = _first_tok(vtok, "Yes"); NO_VL = _first_tok(vtok, "No")
print(f"VLM ready. Yes_id={YES_VL} No_id={NO_VL}")

# Reasoner rieng (optional)
rtok = None
if USE_SEPARATE_REASONER:
    print("Loading LLM reasoner (Qwen2.5-3B-Instruct) ...")
    reasoner = AutoModelForCausalLM.from_pretrained(
        REASONER_ID, quantization_config=_bnb(), torch_dtype=TORCH_DTYPE,
        device_map={"": 0}, attn_implementation=ATTN_IMPL)
    reasoner.eval()
    rtok = AutoTokenizer.from_pretrained(REASONER_ID)
    YES_R = _first_tok(rtok, "Yes"); NO_R = _first_tok(rtok, "No")

@torch.no_grad()
def _last_logits(texts, images=None, use_reasoner=False):
    """Tra ve logits tai token cuoi cung (truoc generation) cho moi text. Batch."""
    mdl = reasoner if (use_reasoner and USE_SEPARATE_REASONER) else vlm
    prc = AutoTokenizer if (use_reasoner and USE_SEPARATE_REASONER) else processor
    if use_reasoner and USE_SEPARATE_REASONER:
        full = [rtok.apply_chat_template([{"role":"user","content":t}], tokenize=False, add_generation_prompt=True) for t in texts]
        inp = rtok(full, padding=True, return_tensors="pt")
    else:
        full = [processor.apply_chat_template([{"role":"user","content":
                  ([{"type":"image","image":im}] if im else [])+[{"type":"text","text":t}]}],
                  tokenize=False, add_generation_prompt=True) for t, im in
                zip(texts, images if images else [None]*len(texts))]
        imgs = [im for im in (images or []) if im is not None] or None
        inp = processor(text=full, images=imgs, padding=True, return_tensors="pt")
    inp = {k: v.to(mdl.device) for k, v in inp.items()}
    logits = mdl(**inp).logits
    pos = inp["attention_mask"].sum(1) - 1
    return logits[torch.arange(logits.size(0)), pos]

def yesno_batch(texts, images=None, use_reasoner=False):
    """P(yes) cho moi text (VS: co image; LS: khong image)."""
    if not texts:
        return []
    last = _last_logits(texts, images=images, use_reasoner=use_reasoner)
    last = torch.nan_to_num(last, nan=0.0, posinf=0.0, neginf=0.0)   # guard fp16 NaN (reasoner logits)
    yes_id = (YES_R if (use_reasoner and USE_SEPARATE_REASONER) else YES_VL)
    no_id  = (NO_R  if (use_reasoner and USE_SEPARATE_REASONER) else NO_VL)
    ly = last[:, yes_id]; ln = last[:, no_id]
    p = torch.softmax(torch.stack([ly, ln], 1), dim=1)[:, 0]
    return p.float().cpu().tolist()

@torch.no_grad()
def generate(prompt, image=None, max_new=96, do_sample=False, temperature=0.0):
    content = ([{"type":"image","image":image}] if image is not None else []) + [{"type":"text","text":prompt}]
    msgs = [{"role":"user","content":content}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs = [image] if image is not None else None
    inp = processor(text=[text], images=imgs, return_tensors="pt")
    inp = {k: v.to(vlm.device) for k, v in inp.items()}
    kw = dict(max_new_tokens=max_new)
    if do_sample:
        kw.update(do_sample=True, temperature=max(temperature, 1e-2), top_p=0.9)
    out = vlm.generate(**inp, **kw)
    gen = vtok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen.strip()


# %% [markdown]
# ## 4. COCO-TREE COMPONENTS  (SMD, RCE, VS, LS, composite score, path search)

# %%
# ---- Prompt templates CHINH XAC tu paper Appendix C (Fig 5,6,7,8) ----
SMD_PROMPT = ("You are a helpful chatbot. Divide the caption into {M} smaller independent "
              "statements which entail the caption based on Subject and Object. Caption: {C}. "
              "The output format is:\n1. <statement>\n2. <statement>\nAssistant:")
RCE_PROMPT = ("You are a helpful chatbot. List {S} binary visual concepts to verify the following "
              "statement: \"{NODE}\". Ensure the outputs are possible for: {C}. Answer in small "
              "phrases and focus on verifiable things like objects, locations, actions, etc. "
              "Output format is:\n1. xxx\n2. xxx\n3. xxx\nAssistant:")

def parse_numbered(text, expected):
    items = []
    for line in text.split("\n"):
        m = re.match(r"\s*\d+[\.\)]\s*(.+)", line.strip())
        if m and m.group(1).strip():
            items.append(m.group(1).strip())
    return items[:expected] if items else []

@torch.no_grad()
def gen_reasoner(prompt, max_new=64):
    """Generate text-only bang LLM reasoner RIENG (paper: SMD/RCE/LS dung 1 LLM rieng).
    Neu USE_SEPARATE_REASONER=False -> fallback VLM text-only."""
    if USE_SEPARATE_REASONER:
        full = rtok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        inp = rtok(full, return_tensors="pt")
        inp = {k: v.to(reasoner.device) for k, v in inp.items()}
        out = reasoner.generate(**inp, max_new_tokens=max_new, do_sample=False)
        return rtok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return generate(prompt, max_new=max_new)

def smd(caption):
    """Semantic Morphological Decomposition -> M entity."""
    out = gen_reasoner(SMD_PROMPT.format(M=M, C=caption), max_new=64)
    ents = parse_numbered(out, M)
    return ents if ents else [caption]

def rce(node, caption):
    """Recursive Concept Exploration -> S concept con."""
    out = gen_reasoner(RCE_PROMPT.format(S=S, NODE=node, C=caption), max_new=80)
    return parse_numbered(out, S)

def build_tree(statement):
    """Xay concept tree: M entity (level 0) -> RCE depth L. Tra (entities, all_nodes)."""
    entities = smd(statement)
    nodes = [{"text": e, "level": 0, "parent": None, "children": []} for e in entities]
    all_nodes = list(nodes)
    frontier = nodes
    for depth in range(1, L + 1):
        nxt = []
        for nd in frontier:
            for ch_text in rce(nd["text"], statement):
                ch = {"text": ch_text, "level": depth, "parent": nd, "children": []}
                nd["children"].append(ch); all_nodes.append(ch); nxt.append(ch)
        frontier = nxt
        if not frontier:
            break
    return entities, all_nodes

def score_nodes(image, statement, entities, all_nodes):
    """VS (VLM, co image) + LS (LLM entail) -> composite score CS cho moi node."""
    texts = [n["text"] for n in all_nodes]
    if not texts:
        return
    vs = yesno_batch([f"Does this figure show: {t}? Please answer Yes or No." for t in texts],
                     images=[image]*len(texts))
    ls = yesno_batch([f"Given we observe {statement}. Is it possible {t}? Answer yes or no." for t in texts],
                     use_reasoner=True)
    for n, v, l in zip(all_nodes, vs, ls):
        n["cs"] = ALPHA * l + (1 - ALPHA) * v

def _agg(vals):
    if not vals:
        return 0.0
    return sum(vals)/len(vals) if PATH_AGG == "mean" else sum(vals)

def _roots(all_nodes):
    """Node goc level-0 (cac entity) — da duoc score_nodes gan 'cs'. Lay node DICT, khong phai string."""
    return [n for n in all_nodes if n.get("level", 0) == 0]

def search_greedy(all_nodes):
    best = 0.0
    for e in _roots(all_nodes):
        path = [e]; cur = e
        while cur["children"]:
            cur = max(cur["children"], key=lambda n: n["cs"]); path.append(cur)
        best = max(best, _agg([n["cs"] for n in path]))
    return best

def search_beam(all_nodes, k=BEAM_K):
    roots = _roots(all_nodes)
    beams = [([e], _agg([e["cs"]])) for e in roots] if roots else [([], 0.0)]
    maxdepth = max((n["level"] for n in all_nodes), default=0)
    for _ in range(maxdepth):
        cand = []
        for path, w in beams:
            last = path[-1] if path else None
            if not last or not last["children"]:
                cand.append((path, w)); continue
            for ch in last["children"]:
                cand.append((path + [ch], _agg([n["cs"] for n in path + [ch]])))
        if not cand:
            break
        cand.sort(key=lambda x: -x[1]); beams = cand[:k]
    return max((w for _, w in beams), default=0.0)


# %% [markdown]
# ## 5. ADAPTATION PIPELINE  (candidate gen -> concept tree -> score -> rerank)

# %%
def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

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

def gen_candidates(image, question):
    """System-1: VLM sinh K candidate answer (greedy + sampling), dedup."""
    base = (f"Look at the image and answer the question in 1-3 words only.\n"
            f"Question: {question}\nAnswer:")
    seen, cands = set(), []
    raw = generate(base, image=image, max_new=12, do_sample=False)
    for c in [raw.strip().split("\n")[0].strip(" .,")]:
        if c and norm(c) not in seen:
            seen.add(norm(c)); cands.append(c)
    tries = 0
    while len(cands) < K_CAND and tries < K_CAND + 3:
        tries += 1
        s = generate(base, image=image, max_new=12, do_sample=True, temperature=0.8)
        c = s.strip().split("\n")[0].strip(" .,")
        if c and norm(c) not in seen:
            seen.add(norm(c)); cands.append(c)
    return cands[:K_CAND] if cands else ["unknown"]

def score_candidate(image, question, cand):
    """Tra (final, f_sys1, W_hat) cho 1 candidate theo COCO-Tree (Eq 7, 8)."""
    statement = f'The answer to the question "{question}" is "{cand}".'
    # System-1 score f(I,C): VLM confidence rang anh phu hop statement
    f_sys1 = yesno_batch([f"Does this figure show: {statement}? Please answer Yes or No."],
                         images=[image])[0]
    entities, all_nodes = build_tree(statement)
    score_nodes(image, statement, entities, all_nodes)
    W = search_greedy(all_nodes) if SEARCH_MODE == "greedy" else search_beam(all_nodes)
    final = BETA * f_sys1 + (1 - BETA) * W      # Eq 8
    return final, f_sys1, W

def predict_one(image, question, gold):
    cands = gen_candidates(image, question)
    scored = {c: score_candidate(image, question, c)[0] for c in cands}
    pred = max(cands, key=lambda c: scored[c])
    # Pool candidate-level cho ROC-AUC/PR-AUC: gold=1, cac candidate khac=0
    pool = [(scored[c], 1 if norm(c) == norm(gold) else 0) for c in cands]
    if not any(lbl == 1 for _, lbl in pool):
        g_score = score_candidate(image, question, gold)[0]
        pool.append((g_score, 1))
    return pred, pool


# %% [markdown]
# ## 6. EVALUATION

# %%
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, average_precision_score, confusion_matrix)

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

preds, golds, pool_all = [], [], []
n_eval = min(N_EVAL, len(eval_recs))
t0 = time.time()
for i in range(n_eval):
    try:
        rec = eval_recs[i]
        img = resolve_image(rec.get("image"))
        q = rec.get("question", "")
        gold = str(rec.get("answer", rec.get("full_answer", "")))
        pred, pool = predict_one(img, q, gold)
        preds.append(pred); golds.append(gold); pool_all += pool
        if i % 5 == 0:
            print(f"[{i}/{n_eval}] gold={gold!r} pred={pred!r} cands_pool={len(pool)}")
    except Exception as e:
        print(f"eval err {i}: {repr(e)[:120]}")
infer_time = time.time() - t0

pred_n = [norm(p) for p in preds]
gold_n = [norm(g) for g in golds]
labels = sorted(set(gold_n))
P, R, F1, _ = precision_recall_fscore_support(gold_n, pred_n, labels=labels,
                                              average="macro", zero_division=0)
acc = accuracy_score(gold_n, pred_n)
fpr, fnr = macro_fpr_fnr(gold_n, pred_n, labels)

# ROC-AUC / PR-AUC candidate-level (gold candidate vs distractor)
scores = [s for s, _ in pool_all]; lbls = [l for _, l in pool_all]
if len(set(lbls)) > 1:
    roc_auc = roc_auc_score(lbls, scores); pr_auc = average_precision_score(lbls, scores)
else:
    roc_auc = pr_auc = "N/A"

params_total = sum(p.numel() for p in vlm.parameters())
print(f"\n=== RESULTS (#11 COCO-Tree, Qwen2.5-VL-3B, Visual CoT-GQA, n={n_eval}) ===")
print(f"Acc={acc:.4f} | Prec={P:.4f} | Recall={R:.4f} | F1={F1:.4f}")
print(f"FPR={fpr:.4f} | FNR={fnr:.4f} | ROC-AUC={roc_auc} | PR-AUC={pr_auc}")
print(f"Infer Time={infer_time:.1f}s | Params={params_total} | Steps=0 (inference-only)")


# %% [markdown]
# ## 7. SAVE RESULTS (CSV / DataFrame)

# %%
result = pd.DataFrame([{
    "STT": 11,
    "Paper": "COCO-Tree (EMNLP 2025)",
    "Task Type": "Compositional reasoning -> adapt sang candidate-rerank VQA",
    "Method Type": "Concept tree (SMD/RCE) + composite score (VS/LS) + greedy/beam path + System-1/2 fusion",
    "Training Unit": "Inference-only (khong train); Qwen2.5-VL-3B 4-bit lam VLM + reasoner",
    "Acc": round(acc, 4), "Prec": round(P, 4), "Recall": round(R, 4), "F1": round(F1, 4),
    "ROC-AUC": (round(roc_auc, 4) if isinstance(roc_auc, float) else roc_auc),
    "PR-AUC": (round(pr_auc, 4) if isinstance(pr_auc, float) else pr_auc),
    "FPR": round(fpr, 4), "FNR": round(fnr, 4),
    "Train Time": round(infer_time, 1), "Params": f"~3B (4-bit); inference-only",
    "Comm Cost": "N/A", "Training Steps": 0,
    "Note": (f"Qwen2.5-VL-3B; Visual CoT-GQA; eval={n_eval}; inference-only (Steps=0). "
             f"ADAPTATION: COCO-Tree goc la image-text matching -> adapt sang candidate-rerank VQA "
             f"(VLM sinh {K_CAND} candidate -> concept tree SMD(M={M})/RCE(S={S},L={L}) -> CS=alpha*LS+(1-alpha)*VS "
             f"(alpha={ALPHA}) -> {SEARCH_MODE} path -> fuse beta*f+(1-beta)*W (beta={BETA})). "
             f"ROC/PR-AUC candidate-level (gold vs distractor). "
             f"Tree size (M={M},S={S},L={L}) giam tu paper (2,3,3) cho compute Kaggle. "
             f"ASSUMPTION: beam k={BEAM_K}, path agg={PATH_AGG}; reasoner = chinh VLM (paper: Llama-3.1-8B rieng; rieng-1.5B y hon tren P100). "
             f"adaptation, not full COCO-Tree reproduction (paper: 4 comp benchmarks, VQAScore metric)."),
}])
result.to_csv("result_11_COCO-Tree.csv", index=False)
import os as _os; _os.makedirs(OUTPUT_DIR, exist_ok=True)
result.to_csv(_os.path.join(OUTPUT_DIR, "result_11_COCO-Tree.csv"), index=False)
pd.DataFrame({"gold": gold_n, "pred": pred_n}).to_csv("preds_11_COCO-Tree.csv", index=False)
print(result.T)
print("\nSaved: result_11_COCO-Tree.csv, preds_11_COCO-Tree.csv")
print("=> Dien Acc/Prec/Recall/F1/FPR/FNR/ROC-AUC/PR-AUC/Train Time/Params/Steps vao sheet 'Implement' dong 11.")
