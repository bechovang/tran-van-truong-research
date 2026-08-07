# -*- coding: utf-8 -*-
# =====================================================================
# #15 LLM4SGG — adaptation implementation
# Paper: "LLM4SGG: Large Language Models for Weakly Supervised Scene Graph
#         Generation", Kim et al., CVPR 2024.
# Protocol (team): Qwen2.5-VL-3B + Visual CoT-GQA, eval n=200.
# =====================================================================
# ADAPTATION NOTE (important): LLM4SGG goc = weakly-supervised Scene Graph
# Generation: KHONG dung dense bbox annotation (expensive) -> dung CAPTION
# (text) lam nguon "weak" supervision. Cho LLM (GPT-3.5/4 trong paper) doc
# caption roi voi CoT + few-shot + danh sach predicate (relation vocab) trich
# (subject, predicate, object) triplets -> align predicate ve vocab class ->
# ghep thanh scene graph. Benchmark goc = VG/GQA, metric Recall@K (SGG).
# -> ADAPT (Huong A, dong nhat #13/#14): lay graph LLM4SGG-build duoc lam
# context -> Qwen2.5-VL-3B tra loi VQA (Visual CoT-GQA). Inference-only
# (reliable tren P100/T4 fp16/4bit).
#   - caption do VLM tu sinh (KHONG dung truong reasoning/thought cua Visual
#     CoT -> do la gold program, se gold-leak).
#   - extraction = CoT + 2 few-shot + constrained predicate vocab (align nhe).
#   - gi nguyen dong gop dac trung cua paper: text-mediated (caption->LLM)
#     extraction, weak supervision, no bbox.
# Faithful VG/GQA Recall@K + full predicate alignment + GPT-4 = future work.
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

# ---- LLM4SGG hyperparams ----
MAX_TRIP    = 10     # kich thuoc scene graph (triplet) max/anh (ASSUMPTION)
CAP_MAXNEW  = 64     # caption ngan
EXT_MAXNEW  = 220    # extraction (CoT + triplets)
DO_BASELINE = True   # VQA khong graph (de do gain cua knowledge)

N_EVAL     = 200
OUTPUT_DIR = "./llm4sgg_out"

if SMOKE:
    N_EVAL, MAX_TRIP = 4, 6
    print(f"!! SMOKE MODE -> N_EVAL={N_EVAL}, MAX_TRIP={MAX_TRIP}")

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
# ## 3. MODEL (Qwen2.5-VL-3B, adaptive) — dung cho ca caption va LLM extraction

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


# %% [markdown]
# ## 4. LLM4SGG: caption -> LLM (CoT + few-shot) triplet extraction -> scene graph -> VQA
# # Dong gop dac trung cua LLM4SGG: SGG weakly-supervised tu TEXT caption
# # (khong can bbox). VLM sinh caption -> LLM (Qwen o che do text) voi CoT +
# # few-shot + constrained predicate vocab trich triplets -> scene graph.
# # Graph do dua vao prompt VQA nhu knowledge context.

# %%
CAPTION_PROMPT = (
    "Describe this image in one or two concise sentences, focusing on the main "
    "objects, people, and their relationships, actions and positions. "
    "Do not answer any question; only describe the scene."
)

# Danh sach predicate (relation vocab) - giong LLM4SGG: constrained generation
# de align extracted relation ve 1 tap class co gioi han (paper dung tap predicate
# cua VG/PSD; day la subset cho adaptation).
REL_VOCAB = ("on", "in", "under", "behind", "next to", "in front of", "of",
             "wearing", "holding", "has", "sitting on", "standing on", "eating",
             "riding", "looking at", "near", "to the left of", "to the right of",
             "using", "made of")

FEWSHOT = (
    "Extract relation triplets (subject, predicate, object) that describe the scene "
    "in the image caption.\n"
    "The predicate MUST be one of: " + ", ".join(REL_VOCAB) + ".\n"
    "Think step by step, then write 'Triplets:' followed by one triplet per line "
    "as 'subject - predicate - object'.\n\n"
    "Caption: A man riding a horse on a dirt road next to a wooden fence.\n"
    "Reasoning: The actors are a man and a horse. 'riding' relates the man to the horse. "
    "The horse is on a dirt road. The man and horse are next to a fence.\n"
    "Triplets:\n"
    "man - riding - horse\n"
    "horse - on - dirt road\n"
    "man - next to - wooden fence\n\n"
    "Caption: A woman holding a cup and sitting on a chair near a table.\n"
    "Reasoning: The woman holds a cup, sits on a chair, and is near a table.\n"
    "Triplets:\n"
    "woman - holding - cup\n"
    "woman - sitting on - chair\n"
    "woman - near - table\n\n"
    "Now do the same for the caption below. Output ONLY the reasoning and the Triplets list.\n"
    "Caption: "
)

@torch.no_grad()
def generate(prompt, image=None, max_new=96, do_sample=False, temperature=0.0, top_p=1.0):
    """Generate dung chung: co anh (VQA/caption) hoac chi text (LLM extraction)."""
    content = ([{"type": "image", "image": image}] if image is not None else []) + \
              [{"type": "text", "text": prompt}]
    msgs = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    proc_kwargs = {"text": [text], "return_tensors": "pt"}
    if image is not None:
        proc_kwargs["images"] = [image]
    inp = processor(**proc_kwargs)
    inp = {k: v.to(vlm.device) for k, v in inp.items()}
    kw = dict(max_new_tokens=max_new)
    if do_sample:
        kw.update(do_sample=True, temperature=max(temperature, 1e-2), top_p=top_p)
    out = vlm.generate(**inp, **kw)
    gen = vtok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen.strip()

@torch.no_grad()
def caption_image(image):
    """Buoc 1: VLM sinh caption ngan (weak text supervision; KHONG dung gold)."""
    return generate(CAPTION_PROMPT, image=image, max_new=CAP_MAXNEW, do_sample=False)

def parse_triples(text):
    """Parse 'subject - predicate - object' (1/dong); giu predicate trong vocab."""
    vocab = set(REL_VOCAB)
    triples = []
    for line in text.splitlines():
        line = line.strip().strip("-–—•").strip()
        if not line:
            continue
        m = re.split(r"\s+[-–—]\s+", line)
        if len(m) == 3 and all(len(x) <= 40 for x in m):
            s, p, o = (x.strip(" .,;") for x in m)
            if s and p and o and s.lower() != o.lower():
                # align nhe: chi giu predicate thuoc vocab (giong LLM4SGG class align)
                if p.lower() in vocab:
                    triples.append((s.lower(), p.lower(), o.lower()))
    seen, out = set(), []
    for t in triples:
        if t not in seen:
            seen.add(t); out.append(t)
    return out[:MAX_TRIP]

@torch.no_grad()
def llm_extract_triples(caption):
    """Buoc 2 (LLM4SGG core): LLM (text) CoT + few-shot trich triplets tu caption."""
    prompt = FEWSHOT + caption + "\nReasoning:"
    raw = generate(prompt, image=None, max_new=EXT_MAXNEW, do_sample=False)
    seg = raw
    idx = raw.rfind("Triplets")
    if idx != -1:
        seg = raw[idx:]              # chi parse phan sau "Triplets:"
    return parse_triples(seg), raw

def build_graph(image):
    """LLM4SGG pipeline: VLM caption -> LLM CoT+few-shot extraction -> scene graph."""
    cap = caption_image(image)
    triples, raw_ext = llm_extract_triples(cap)
    return cap, triples, raw_ext

def graph_ctx(triples):
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
    prefix = f"Scene graph (from caption): {ctx}\n\n" if ctx else ""
    prompt = (f"{prefix}Look at the image and answer the question in 1-5 words only.\n"
              f"Question: {question}\nAnswer:")
    return generate(prompt, image=image, max_new=16, do_sample=False).strip().split("\n")[0].strip(" .,;")


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

preds_g, preds_base, golds, graph_sizes = [], [], [], []
n_eval = min(N_EVAL, len(eval_recs))
t0 = time.time()
for i in range(n_eval):
    try:
        rec = eval_recs[i]
        img = resolve_image(rec.get("image"))
        q = rec.get("question", "")
        gold = str(rec.get("answer", rec.get("full_answer", "")))
        cap, triples, _raw = build_graph(img)
        graph_sizes.append(len(triples))
        a_g = vqa(img, q, ctx=graph_ctx(triples))
        preds_g.append(a_g); golds.append(gold)
        if DO_BASELINE:
            preds_base.append(vqa(img, q, ctx=""))
        if i % 25 == 0:
            print(f"[{i}/{n_eval}] gold={gold!r} graph_ans={a_g!r} "
                  f"cap={cap[:55]!r} triples={len(triples)} ctx={graph_ctx(triples)[:60]!r}")
    except Exception as e:
        print(f"eval err {i}: {repr(e)[:120]}")
infer_time = time.time() - t0

g_n   = [norm(p) for p in preds_g]
gold_n = [norm(g) for g in golds]
labels = sorted(set(gold_n))
P, R, F1, _ = precision_recall_fscore_support(gold_n, g_n, labels=labels, average="macro", zero_division=0)
acc = accuracy_score(gold_n, g_n)
fpr, fnr = macro_fpr_fnr(gold_n, g_n, labels)
acc_base = None
if DO_BASELINE and len(preds_base) == len(golds):
    acc_base = accuracy_score(gold_n, [norm(p) for p in preds_base])

params_total = sum(p.numel() for p in vlm.parameters())
print(f"\n=== RESULTS (#15 LLM4SGG, Qwen2.5-VL-3B, Visual CoT-GQA, n={n_eval}) ===")
print(f"[graph-injected] Acc={acc:.4f} | Prec={P:.4f} | Recall={R:.4f} | F1={F1:.4f}")
print(f"FPR={fpr:.4f} | FNR={fnr:.4f} | ROC-AUC=N/A | PR-AUC=N/A | Recall@K=N/A (EM VQA, no rerank)")
if acc_base is not None:
    print(f"[baseline no-graph] Acc={acc_base:.4f}  (graph gain = {acc-acc_base:+.4f})")
print(f"Infer Time={infer_time:.1f}s | avg scene-graph triples={np.mean(graph_sizes) if graph_sizes else 0:.1f} | "
      f"Params={params_total} | Steps=0 (inference-only)")


# %% [markdown]
# ## 6. SAVE RESULTS

# %%
result = pd.DataFrame([{
    "STT": 15,
    "Paper": "LLM4SGG (CVPR 2024)",
    "Task Type": "Weakly-supervised Scene Graph Generation -> adapt sang graph-injected VQA",
    "Method Type": "VLM caption -> LLM CoT+few-shot triplet extraction (constrained predicate vocab) -> scene graph -> inject as VQA context",
    "Training Unit": "Inference-only (khong train); Qwen2.5-VL-3B lam VLM caption + LLM extractor",
    "Acc": round(acc, 4), "Prec": round(P, 4), "Recall": round(R, 4), "F1": round(F1, 4),
    "ROC-AUC": "N/A", "PR-AUC": "N/A", "FPR": round(fpr, 4), "FNR": round(fnr, 4),
    "Train Time": round(infer_time, 1), "Params": f"~3B (4bit={USE_4BIT}); inference-only",
    "Comm Cost": "N/A", "Training Steps": 0,
    "Note": (f"Qwen2.5-VL-3B; Visual CoT-GQA; eval={n_eval}; inference-only (Steps=0). "
             f"ADAPTATION: LLM4SGG goc = weakly-supervised SGG: dung CAPTION (text) thay vi dense bbox "
             f"(weak supervision); LLM (GPT-3.5/4 trong paper) voi CoT + few-shot + predicate vocab trich "
             f"(subject,predicate,object) tu caption -> scene graph; benchmark VG/GQA, metric Recall@K. "
             f"-> Adapt (Huong A): VLM tu sinh caption ({CAP_MAXNEW} tok; KHONG dung gold reasoning/thought) "
             f"-> Qwen (che do text) CoT+2-few-shot+{len(REL_VOCAB)}-predicate-vocab trich triplets "
             f"(align predicate ve vocab, giong class align cua paper) -> scene graph (cap {MAX_TRIP}) "
             f"-> inject nhu VQA context -> tra loi. Giu nguyen dong gop dac trung: text-mediated "
             f"(caption->LLM), weak supervision, no bbox. "
             f"Graph-inj Acc={acc:.4f}" + (f"; baseline Acc={acc_base:.4f} (gain {acc-acc_base:+.4f})" if acc_base is not None else "") +
             f". Recall@K/ROC/PR-AUC=N/A (EM VQA, khong rerank). "
             f"Faithful VG/GQA Recall@K + full predicate alignment + GPT-4 = future work. "
             f"adaptation, not full LLM4SGG reproduction."),
}])
import os as _os; _os.makedirs(OUTPUT_DIR, exist_ok=True)
result.to_csv("result_15_LLM4SGG.csv", index=False)
result.to_csv(_os.path.join(OUTPUT_DIR, "result_15_LLM4SGG.csv"), index=False)
pd.DataFrame({"gold": gold_n, "pred_graph": g_n}).to_csv("preds_15_LLM4SGG.csv", index=False)
print(result.T)
print("\nSaved: result_15_LLM4SGG.csv, preds_15_LLM4SGG.csv")
print("=> Dien Acc/Prec/Recall/F1/FPR/FNR/Train Time/Params/Steps vao sheet 'Implement' dong 15.")
