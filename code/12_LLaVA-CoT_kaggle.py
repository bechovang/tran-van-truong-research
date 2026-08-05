# -*- coding: utf-8 -*-
# =====================================================================
# #12 LLaVA-CoT — adaptation implementation
# Paper: Xu et al., "LLaVA-CoT: Let Vision Language Models Reason Step-by-Step", ICCV 2025
# Protocol (team): Qwen2.5-VL-3B + Visual CoT-GQA + QLoRA r=8, train=256, eval=200
# Target: Kaggle (P100 16GB / T4x2) or Colab T4 free-tier
# ---------------------------------------------------------------------
# Cells are marked with "# %% [markdown]" / "# %%". Paste each block into a Kaggle cell.
# =====================================================================


# %% [markdown]
# ## 0. Install dependencies (run once)

# %%
# ---- Cai dat deps (Kaggle default image co the co transformers cu; Qwen2.5-VL can >=4.45) ----
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.45", "qwen-vl-utils", "peft", "bitsandbytes",
                "accelerate", "datasets", "scikit-learn", "pandas", "pillow", "kagglehub"],
               check=False)
print("deps ready")


# %% [markdown]
# ## 1. CONFIG  (sua duong dan dataset o day)

# %%
import os, re, time, json, random
import torch
import numpy as np
import pandas as pd
from PIL import Image

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---- Smoke test: chay nho (n=4, 2 steps) de bat bug GPU truoc khi chay full ----
SMOKE = os.environ.get("SMOKE", "0") == "1"

# ---- Model ----
MODEL_ID        = "Qwen/Qwen2.5-VL-3B-Instruct"     # paper dung Llama-3.2-11B -> ta adapt xuong Qwen2.5-VL-3B
USE_4BIT        = True                                # free-tier: can 4-bit de vua RAM

# ---- Dataset (VISUAL COT - GQA subset) ----
# Dung dung cau truc ma team dang dung (notebook Khoa):
#   HF deepcs233/Visual-CoT -> cot_with_detailed_reasoning_steps/gqa_cot_{train,val}.jsonl
#   moi mau co: question, answer (ngan), full_answer (NL), thought (NL reasoning),
#               bboxs [[x1,y1,x2,y2]], image (filename), reasoning (scene-graph program)
DATASET_HF      = "deepcs233/Visual-CoT"
GQA_TRAIN_JSONL = "cot_with_detailed_reasoning_steps/gqa_cot_train.jsonl"
GQA_VAL_JSONL   = "cot_with_detailed_reasoning_steps/gqa_cot_val.jsonl"
DATA_OUT_DIR    = "/kaggle/working/visual-cot"        # noi luu jsonl + anh (Kaggle)
# Thu muc chua anh GQA. Anh trong jsonl chi la filename (vd '2331819.jpg').
# download_visual_cot.py (kaggle dataset khoangoo/test-data) dat anh tai: cot/gqa/<id>.jpg
GQA_IMAGES_DIR  = "/kaggle/input/gqa-images/images"     # dataset lyte69/gqa-images: images/<id>.jpg
GQA_IMG_CANDIDATES = [                                  # thu lan luot cac path co the
    "{img}",
    "/kaggle/input/gqa-images/images/{img}",            # CHINH: Kaggle dataset lyte69/gqa-images
    "/kaggle/working/visual-cot/cot/gqa/{img}",
    "/kaggle/input/visual-cot/cot/gqa/{img}",
]

# ---- Adaptation setup (team protocol) ----
N_TRAIN         = 256
N_EVAL          = 200
LORA_R          = 8                                   # team protocol: QLoRA r=8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.05
MAX_STEPS       = 32                                  # team protocol ~32 steps
LR              = 2e-4                                # ASSUMPTION: paper khong noi ro Appendix C trong version nay
BATCH_SIZE      = 1
GRAD_ACCUM      = 4                                   # effective batch 4
MAX_LEN         = 1024
MAX_PIXELS      = 602112                              # ~ giong default Qwen2.5-VL fine-tune
OUTPUT_DIR      = "./llava_cot_out"

if SMOKE:
    N_TRAIN, N_EVAL, MAX_STEPS = 4, 4, 2
    print("!! SMOKE MODE -> N_TRAIN=4, N_EVAL=4, MAX_STEPS=2 (de bat bug GPU)")

# ---- SWIRES (test-time, OPTIONAL / ablation) ----
USE_SWIRES      = False        # True de chay ablation test-time scaling
SWIRES_M        = 4            # ASSUMPTION: paper khong noi ro M/N/C/threshold (Appendix D thieu)
SWIRES_N        = 2
SWIRES_MAX_RETRACE = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# %% [markdown]
# ## 2. LOAD DATASET

# %%
from datasets import Dataset
from huggingface_hub import hf_hub_download
from pathlib import Path

def _find_data_file(rel_path):
    """
    Tim file dataset (vd 'cot_with_detailed_reasoning_steps/gqa_cot_train.jsonl')
    o moi noi: UU TIEN input Kaggle da attach (dataset visual-cot ban vua add),
    roi den /kaggle/working (neu da download HF), roi den thu muc local.
    -> khong phu thuoc Internet/DNS cho data (nhanh + on dinh hon download HF).
    Tra ve path dau tien tim thay, hoac None.
    """
    fname = os.path.basename(rel_path)            # gqa_cot_train.jsonl
    roots = [r for r in ("/kaggle/input", "/kaggle/working", ".") if os.path.isdir(r)]
    # 1) Khop dung relative path (vd /kaggle/input/visual-cot/cot_with_detailed_reasoning_steps/gqa_cot_train.jsonl)
    for root in roots:
        cand = os.path.join(root, rel_path.lstrip("/"))
        if os.path.isfile(cand):
            return cand
    # 2) rglob theo basename -> chong sai slug dataset / long thu muc khac
    matches = []
    for root in roots:
        for p in Path(root).rglob(fname):
            if p.is_file():
                matches.append(str(p))
    # uu tien file nam trong thu muc 'cot_with_detailed_reasoning_steps'
    matches.sort(key=lambda p: ("cot_with_detailed_reasoning_steps" not in p, len(p)))
    return matches[0] if matches else None

def _download_jsonl(filename):
    """Tai 1 file jsonl tu HF deepcs233/Visual-CoT ve DATA_OUT_DIR."""
    path = hf_hub_download(repo_id=DATASET_HF, repo_type="dataset",
                           filename=filename, local_dir=DATA_OUT_DIR)
    return path

def load_jsonl(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs

def load_visual_cot_gqa():
    """Load GQA subset. UU TIEN input Kaggle da attach (dataset visual-cot);
    neu khong thay moi download tu HF deepcs233/Visual-CoT (co retry/mirror o build)."""
    paths = {}
    for key, rel in (("train", GQA_TRAIN_JSONL), ("val", GQA_VAL_JSONL)):
        found = _find_data_file(rel)
        if found:
            print(f"[data] {key}: dung input -> {found}")
        else:
            print(f"[data] {key}: khong thay input, download tu HF ...")
            found = _download_jsonl(rel)
        paths[key] = found
    train_recs = load_jsonl(paths["train"])
    val_recs   = load_jsonl(paths["val"])
    print(f"train: {len(train_recs)} | val: {len(val_recs)}")
    return train_recs, val_recs

train_recs, val_recs = load_visual_cot_gqa()
raw_train = Dataset.from_list(train_recs)
raw_eval  = Dataset.from_list(val_recs) if val_recs else raw_train
idx = 0
print("Sample keys:", list(train_recs[idx].keys()))
print("Sample:", {k: (str(v)[:120]) for k, v in train_recs[idx].items()})


# %% [markdown]
# ## 3. PREPROCESSING — xay 4-stage target (SUMMARY / CAPTION / REASONING / CONCLUSION)

# %%
# Tag dung ten chinh xac theo paper Sec 3.1.1
TAG_SUMMARY   = "<SUMMARY>"
TAG_CAPTION   = "<CAPTION>"
TAG_REASONING = "<REASONING>"
TAG_CONCLUSION= "<CONCLUSION>"

def _get(d, *keys, default=""):
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    return default

def resolve_image(img_field):
    """img trong jsonl chi la filename (vd '2331819.jpg') -> thu cac path candidate."""
    import io
    if isinstance(img_field, Image.Image):
        return img_field.convert("RGB")
    name = img_field
    if isinstance(name, list) and name:
        name = name[0]
    name = str(name).split("###", 1)[0]   # bo phan crop box
    for tpl in GQA_IMG_CANDIDATES + [name]:
        p = tpl.format(img=name) if "{img}" in tpl else tpl
        if os.path.exists(p):
            return Image.open(p).convert("RGB")
    raise FileNotFoundError(f"Khong tim thay anh GQA: {name}. Set GQA_IMAGES_DIR / GQA_IMG_CANDIDATES dung.")

def build_four_stage(sample):
    """
    Map 1 sample Visual CoT-GQA (gqa_cot jsonl) thanh 4-stage target co tag.
    Field dung theo notebook team:
      question, full_answer (NL), answer (ngan), thought (NL reasoning), bboxs, image.
      SUMMARY    : tom tat cau hoi (template)          -- paper: 'high-level summary of the question'
      CAPTION    : mieu ta vung lien quan (tu bboxs)    -- paper: 'visual elements relevant to the question'
      REASONING  : 'thought' (NL reasoning steps)       -- paper: 'logical reasoning'
      CONCLUSION : answer (ngan) truoc, fallback full_answer
                   -- paper: 'final answer (user-facing)'; Sec 3.1.1: neu user muon
                   cau tra loi ngan, conclusion se ngan. Protocol team: Acc=EM tren
                   short answer -> CONCLUSION phai la short answer de EM co nghia.
                   ADAPTATION NOTE (de xuat cua toi, khong phai paper noi ro):
                   paper khong ghi ro conclusion ngan hay dai tuy y.
    """
    question    = _get(sample, "question", "query")
    full_ans    = _get(sample, "full_answer", default="")
    short_ans   = _get(sample, "answer", default=full_ans)
    thought     = _get(sample, "thought", "reasoning", "rationale", default="")
    bboxs       = _get(sample, "bboxs", "bbox", default=None)
    image       = resolve_image(_get(sample, "image", "img"))

    summary_txt = f"The task is to answer the question: {question}"

    if bboxs:
        try:
            bb = bboxs[0] if isinstance(bboxs, list) and isinstance(bboxs[0], list) else bboxs
            cap_txt = (f"The region most relevant to the question is within bounding box "
                       f"[{int(bb[0])}, {int(bb[1])}, {int(bb[2])}, {int(bb[3])}].")
        except Exception:
            cap_txt = "Focus on the image region relevant to the question."
    else:
        cap_txt = "Focus on the image region relevant to the question."

    reas_txt = thought if thought else "Analyze the image step by step to find the answer."
    # CONCLUSION = short answer (EM-able theo protocol team). Fallback full_answer.
    concl    = short_ans if short_ans else full_ans

    target = (f"{TAG_SUMMARY} {summary_txt} </SUMMARY>\n"
              f"{TAG_CAPTION} {cap_txt} </CAPTION>\n"
              f"{TAG_REASONING} {reas_txt} </REASONING>\n"
              f"{TAG_CONCLUSION} {concl} </CONCLUSION>")
    return image, question, target, short_ans   # short_ans = gold cho metric

# Kiem tra 1 sample
img, q, tgt, gold = build_four_stage(train_recs[idx])
print("Q:", q); print("GOLD:", gold); print("TARGET:\n", tgt)


# %% [markdown]
# ## 4. MODEL / MODULES — Qwen2.5-VL-3B + 4-bit + LoRA

# %%
from transformers import (Qwen2_5_VLForConditionalGeneration, AutoProcessor,
                          BitsAndBytesConfig, Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

def build_model_and_processor():
    bnb = None
    if USE_4BIT:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model)
    # LoRA chi len LLM (vision encoder frozen) — giong paper "visual encoder frozen"
    peft_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=MAX_PIXELS, do_resize=True)
    return model, processor

model, processor = build_model_and_processor()
tok = processor.tokenizer


# %% [markdown]
# ## 5. DATA COLLATOR (xu ly image + mask prompt)

# %%
class FourStageCollator:
    """
    Tao input_ids + pixel_values + image_grid_thw; labels mask:
      - phan PROMPT (chi tinh loss phan 4-stage target cua assistant)
      - token image_pad (khong dua vao loss)
    Dung chuan Qwen2.5-VL: apply_chat_template cho full conversation (user+assistant).
    """
    def __init__(self, processor, max_len=MAX_LEN):
        self.processor = processor
        self.tok = processor.tokenizer
        self.max_len = max_len
        self.image_token_id = self.tok.convert_tokens_to_ids("<|image_pad|>")

    def __call__(self, batch):
        messages_list, _ = [], []
        for image, question, target in batch:
            messages_list.append([
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": (f"{question}\nPlease reason step by step and present the "
                                              f"final answer within <CONCLUSION> ... </CONCLUSION>.")},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": target}]},
            ])

        # Full text (user+assistant, co image placeholder) va prompt-only text (de tinh bien mask)
        full_texts = [self.processor.apply_chat_template(m, tokenize=False) for m in messages_list]
        prompt_texts = [self.processor.apply_chat_template(m[:1], tokenize=False,
                                                           add_generation_prompt=True)
                        for m in messages_list]

        image_inputs, video_inputs = process_vision_info(messages_list)
        inputs = self.processor(
            text=full_texts, images=image_inputs, videos=video_inputs,
            padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")

        labels = inputs["input_ids"].clone()
        labels[inputs["input_ids"] == self.tok.pad_token_id] = -100
        labels[labels == self.image_token_id] = -100
        # FIX (verified 2026-08-05): tokenize prompt-only qua CUNG processor (co image)
        # de <|image_pad|> duoc expand giong nhu full input -> plen chinh xac.
        # Truoc day dung tok(pt, add_special_tokens=False) -> under-mask ~60 token
        # (prompt co image 224x224), de lo <|im_start|>assistant\n vao loss.
        prompt_inputs = self.processor(
            text=prompt_texts, images=image_inputs, videos=video_inputs,
            padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        for i in range(len(prompt_texts)):
            plen = int((prompt_inputs["input_ids"][i] != self.tok.pad_token_id).sum())
            labels[i, :plen] = -100            # mask phan prompt (user + image)
        inputs["labels"] = labels
        return inputs

# Build train list (chi giu 3-tuple (image, question, target); bo gold cho training)
def make_list(ds, n):
    out = []
    for i in range(min(n, len(ds))):
        try:
            out.append(build_four_stage(ds[i])[:3])
        except Exception as e:
            continue
    print(f"built {len(out)} samples (asked {n})")
    return out

train_list = make_list(raw_train, N_TRAIN)
# Eval: dung split val rieng (gqa_cot_val). Neu val trong, lay cuoi train.
eval_list_raw = raw_eval if len(raw_eval) > 0 else raw_train


# %% [markdown]
# ## 6. TRAINING (QLoRA SFT, ~32 steps)

# %%
train_time = 0.0
if len(train_list) >= BATCH_SIZE:
    # FIX: P100 (Kaggle) khong ho tro bf16 -> tu chon dtype theo capability.
    #   Ampere+ (>=8.0): bf16; Turing/Volta (7.x/6.x): fp16 (dung 4-bit + fp16).
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        use_bf16, use_fp16 = True, False
    else:
        use_bf16, use_fp16 = False, True
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=MAX_STEPS,
        learning_rate=LR,
        logging_steps=4,
        save_strategy="no",
        bf16=use_bf16,
        fp16=use_fp16,
        optim="paged_adamw_8bit",
        report_to=[],
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )
    collator = FourStageCollator(processor)
    trainer = Trainer(model=model, args=args, train_dataset=train_list, data_collator=collator)
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print(f"Train time: {train_time:.1f}s | steps: {MAX_STEPS}")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "lora"))
else:
    print("Skip training: not enough samples.")


# %% [markdown]
# ## 7. EVALUATION — trich <CONCLUSION>, tinh metrics

# %%
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              roc_auc_score, average_precision_score, confusion_matrix)

def norm_text(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def extract_conclusion(text):
    m = re.search(r"<CONCLUSION>(.*?)</CONCLUSION>", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip().split("\n")[-1]

@torch.no_grad()
def predict(image, question):
    messages = [{"role":"user","content":[
        {"type":"image","image":image},
        {"type":"text","text": f"{question}\nThink step by step using <SUMMARY>,<CAPTION>,<REASONING>,<CONCLUSION> tags."},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=384, do_sample=False)
    gen = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen

preds_raw, golds = [], []
n_eval = min(N_EVAL, len(eval_list_raw))
for i in range(n_eval):
    try:
        img, q, tgt, gold = build_four_stage(eval_list_raw[i])
        gen = predict(img, q)
        preds_raw.append(gen); golds.append(str(gold))
        if i % 25 == 0: print(f"[{i}/{n_eval}] gold={gold!r} concl={extract_conclusion(gen)!r}")
    except Exception as e:
        print("eval err", i, e)

preds = [norm_text(extract_conclusion(g)) for g in preds_raw]
gold_n = [norm_text(g) for g in golds]

# --- Metrics ---
labels = sorted(set(gold_n))
P, R, F1, _ = precision_recall_fscore_support(gold_n, preds, labels=labels,
                                              average="macro", zero_division=0)
acc = accuracy_score(gold_n, preds)
# FPR/FNR: macro 1-vs-rest (standard multiclass version). Note: team co the dung dinh ngha khac.
def macro_fpr_fnr(y_true, y_pred, labels):
    fprs, fnrs = [], []
    n = len(y_true)
    for c in labels:
        yt = [1 if y==c else 0 for y in y_true]
        yp = [1 if y==c else 0 for y in y_pred]
        try:
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
            fprs.append(fp/(fp+tn) if (fp+tn)>0 else 0)
            fnrs.append(fn/(fn+tp) if (fn+tp)>0 else 0)
        except Exception:
            pass
    return float(np.mean(fprs)), float(np.mean(fnrs))
fpr, fnr = macro_fpr_fnr(gold_n, preds, labels)

# ROC-AUC / PR-AUC: VQA khong co score tu nhien -> N/A (giong cac row cua team).
# Chi co gia tri khi co rerank (SWIRES) -> danh gia candidate-level o ablation.
roc_auc = "N/A"; pr_auc = "N/A"

params_total = sum(p.numel() for p in model.parameters())
params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n=== RESULTS (#12 LLaVA-CoT, Qwen2.5-VL-3B, Visual CoT-GQA, n={n_eval}) ===")
print(f"Acc={acc:.4f} | Prec={P:.4f} | Recall={R:.4f} | F1={F1:.4f}")
print(f"FPR={fpr:.4f} | FNR={fnr:.4f} | ROC-AUC={roc_auc} | PR-AUC={pr_auc}")
print(f"Train Time={train_time:.1f}s | Params(total)={params_total} | Params(trainable LoRA)={params_trainable} | Steps={MAX_STEPS}")


# %% [markdown]
# ## 8. SAVE RESULTS (CSV / DataFrame)

# %%
result = pd.DataFrame([{
    "STT": 12,
    "Paper": "LLaVA-CoT (ICCV 2025)",
    "Task Type": "Multimodal VQA + structured 4-stage CoT reasoning",
    "Method Type": "QLoRA SFT 4-stage (SUMMARY/CAPTION/REASONING/CONCLUSION); optional SWIRES",
    "Training Unit": "QLoRA r=8 on Qwen2.5-VL-3B (vision encoder frozen)",
    "Acc": round(acc,4), "Prec": round(P,4), "Recall": round(R,4), "F1": round(F1,4),
    "ROC-AUC": roc_auc, "PR-AUC": pr_auc, "FPR": round(fpr,4), "FNR": round(fnr,4),
    "Train Time": round(train_time,1), "Params": f"~3B; LoRA {params_trainable}",
    "Comm Cost": "N/A", "Training Steps": MAX_STEPS,
    "Note": (f"Qwen2.5-VL-3B; Visual CoT-GQA; train={N_TRAIN}; eval={n_eval}; "
             f"4-stage structured CoT SFT (QLoRA r=8); Acc=EM, macro P/R/F1 over gold classes; "
             f"adaptation, not full LLaVA-CoT reproduction (paper: Llama-3.2-11B full FT). "
             f"Appendix C/D (lr, SWIRES M/N/C/threshold) not in this paper version."),
}])
result.to_csv("result_12_LLaVA-CoT.csv", index=False)
result.to_csv(os.path.join(OUTPUT_DIR, "result_12_LLaVA-CoT.csv"), index=False)
print(result.T)

# Luu predictions de debug
pd.DataFrame({"gold": gold_n, "pred": preds, "raw": preds_raw}).to_csv("preds_12_LLaVA-CoT.csv", index=False)
print("\nSaved: result_12_LLaVA-CoT.csv, preds_12_LLaVA-CoT.csv")
print("=> Chay xong: lay cac gia tri Acc/Prec/Recall/F1/FPR/FNR/Train Time/Params/Steps dien vao sheet 'Implement' dong 12.")
