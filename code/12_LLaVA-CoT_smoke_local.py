# -*- coding: utf-8 -*-
"""
#12 LLaVA-CoT — LOCAL smoke test (CPU only, KHONG load model)
=============================================================
Chay duoc tren may khong co GPU de verify:
  1. Load Visual CoT-GQA jsonl (HF deepcs233/Visual-CoT)
  2. build_four_stage -> 4-stage target dung tag cua paper Sec 3.1.1
  3. Collator (Qwen2.5-VL processor thật, CPU) -> loss masking chinh xac
  4. Metrics (EM, macro P/R/F1, FPR/FNR) tren mock predictions

Cach chay:
    python 12_LLaVA-CoT_smoke_local.py

Can cai: transformers>=4.45 qwen-vl-utils datasets scikit-learn pillow
(Khong can GPU, khong can model weights - chi tai processor ~10MB)

Paper: Xu et al., "LLaVA-CoT: Let Vision Language Models Reason Step-by-Step", ICCV 2025
"""
import os, re, json, logging
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
logging.disable(logging.WARNING)

import torch
import numpy as np
from PIL import Image

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

TAG_SUMMARY, TAG_CAPTION, TAG_REASONING, TAG_CONCLUSION = (
    "<SUMMARY>", "<CAPTION>", "<REASONING>", "<CONCLUSION>",
)
GQA_VAL_JSONL = "cot_with_detailed_reasoning_steps/gqa_cot_val.jsonl"


def _get(d, *keys, default=""):
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    return default


def build_four_stage(sample):
    """Map 1 sample GQA -> (question, target_4stage, gold_short_answer)."""
    question  = _get(sample, "question", "query")
    full_ans  = _get(sample, "full_answer", default="")
    short_ans = _get(sample, "answer", default=full_ans)
    thought   = _get(sample, "thought", "reasoning", "rationale", default="")
    bboxs     = _get(sample, "bboxs", "bbox", default=None)

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
    # CONCLUSION = short answer (EM theo protocol team; fallback full_answer)
    concl = short_ans if short_ans else full_ans

    target = (f"{TAG_SUMMARY} {summary_txt} </SUMMARY>\n"
              f"{TAG_CAPTION} {cap_txt} </CAPTION>\n"
              f"{TAG_REASONING} {reas_txt} </REASONING>\n"
              f"{TAG_CONCLUSION} {concl} </CONCLUSION>")
    return question, target, short_ans


def load_val():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id="deepcs233/Visual-CoT", repo_type="dataset",
                           filename=GQA_VAL_JSONL, local_dir="./hf_cache")
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def test_collator(processor, samples):
    """Test FourStageCollator (cung logic nhu code/12_LLaVA-CoT_kaggle.py)."""
    from qwen_vl_utils import process_vision_info
    tok = processor.tokenizer
    image_token_id = tok.convert_tokens_to_ids("<|image_pad|>")
    pad_token_id = tok.pad_token_id

    dummy_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    messages_list = []
    for q, target, _ in samples[:2]:
        messages_list.append([
            {"role": "user", "content": [
                {"type": "image", "image": dummy_img},
                {"type": "text", "text": (f"{q}\nPlease reason step by step and present the "
                                          f"final answer within <CONCLUSION> ... </CONCLUSION>.")},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": target}]},
        ])
    full_texts = [processor.apply_chat_template(m, tokenize=False) for m in messages_list]
    prompt_texts = [processor.apply_chat_template(m[:1], tokenize=False, add_generation_prompt=True)
                    for m in messages_list]
    image_inputs, video_inputs = process_vision_info(messages_list)
    inputs = processor(text=full_texts, images=image_inputs, videos=video_inputs,
                       padding=True, truncation=True, max_length=1024, return_tensors="pt")
    labels = inputs["input_ids"].clone()
    labels[inputs["input_ids"] == pad_token_id] = -100
    labels[labels == image_token_id] = -100
    # FIX: prompt qua cung processor -> length chinh xac
    prompt_inputs = processor(text=prompt_texts, images=image_inputs, videos=video_inputs,
                              padding=True, truncation=True, max_length=1024, return_tensors="pt")
    for i in range(len(prompt_texts)):
        plen = int((prompt_inputs["input_ids"][i] != pad_token_id).sum())
        labels[i, :plen] = -100

    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    asst = tok.encode("assistant", add_special_tokens=False)[0]
    ok = True
    for i in range(len(prompt_texts)):
        ids = inputs["input_ids"][i].tolist()
        asst_pos = next(j for j in range(len(ids) - 2) if ids[j] == im_start and ids[j + 1] == asst)
        lab = labels[i].tolist()
        unmasked_prompt = [v for v in lab[:asst_pos + 3] if v != -100]
        tgt_ok = all(v != -100 for v in lab[asst_pos + 3: asst_pos + 8])
        if unmasked_prompt or not tgt_ok:
            ok = False
            print(f"  sample {i}: FAIL unmasked_prompt={unmasked_prompt[:5]} tgt_ok={tgt_ok}")
        else:
            print(f"  sample {i}: OK prompt mask chinh xac (0 non-masked), target giu nguyen")
    return ok


def test_metrics():
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

    def norm_text(s):
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9 ]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    def extract_conclusion(text):
        m = re.search(r"<CONCLUSION>(.*?)</CONCLUSION>", text, re.DOTALL)
        return m.group(1).strip() if m else text.strip().split("\n")[-1]

    golds = ["hot dog", "van", "plants"]
    preds = ["<CONCLUSION> hot dog </CONCLUSION>",
             "<CONCLUSION> a van </CONCLUSION>",
             "<SUMMARY> x </SUMMARY><CONCLUSION> plants </CONCLUSION>"]
    gold_n = [norm_text(g) for g in golds]
    pred_n = [norm_text(extract_conclusion(p)) for p in preds]
    labels = sorted(set(gold_n))
    P, R, F1, _ = precision_recall_fscore_support(gold_n, pred_n, labels=labels,
                                                  average="macro", zero_division=0)
    acc = accuracy_score(gold_n, pred_n)
    print(f"  Acc={acc:.3f} P={P:.3f} R={R:.3f} F1={F1:.3f}")
    return acc == 2 / 3


def main():
    print("=== 1. LOAD dataset (val) ===")
    val = load_val()
    print(f"  val samples: {len(val)} | keys: {list(val[0].keys())}")

    print("\n=== 2. build_four_stage (5 mau) ===")
    for i in range(5):
        q, tgt, gold = build_four_stage(val[i])
        for tag in ("SUMMARY", "CAPTION", "REASONING", "CONCLUSION"):
            assert f"<{tag}>" in tgt and f"</{tag}>" in tgt, f"missing tag {tag}"
        print(f"  [{i}] Q={q[:40]!r} gold={gold!r}")
    print("  -> 4 stage + tag: OK")
    empty = val[0].copy(); empty["bboxs"] = []
    q, tgt, gold = build_four_stage(empty)
    assert "Focus on the image region" in tgt, "bbox-empty fallback fail"
    print("  -> bbox empty fallback: OK")

    print("\n=== 3. Collator (processor Qwen2.5-VL that, CPU) ===")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=602112, do_resize=True)
    samples = [build_four_stage(val[i]) for i in range(4)]
    ok = test_collator(processor, samples)

    print("\n=== 4. Metrics (mock) ===")
    ok2 = test_metrics()

    print("\nRESULT:", "ALL LOCAL PIPELINE TESTS PASSED" if (ok and ok2)
          else "SOME TESTS FAILED (xem log tren)")
    print("Note: khong chay duoc model forward/backward (khong co GPU). Chay tren Kaggle (SMOKE=1).")


if __name__ == "__main__":
    main()
