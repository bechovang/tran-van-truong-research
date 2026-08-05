# #12 LLaVA-CoT — Tóm tắt kiến trúc đã VERIFY (Prompt 1+2 equivalent)

> Nguồn: Xu et al., "LLaVA-CoT: Let Vision Language Models Reason Step-by-Step", ICCV 2025.
> Phương pháp verify: đọc trực tiếp paper (version ICCV open-access). Chỗ paper không nói rõ → ghi thẳng.

## 1. Mục tiêu paper
Xây VLM có khả năng **reasoning có cấu trúc, tự chủ, nhiều giai đoạn** (thay vì CoT prompting đơn thuần), + phương pháp **test-time scaling** (SWIRES) để tự sửa lỗi. Kết quả: vượt base model +9.4%, vượt nhiều model lớn/close-source (Gemini-1.5-pro, GPT-4o-mini, Llama-3.2-90B).

## 2. Kiến trúc đề xuất
Không phải kiến trúc mạng mới — là **paradigm training + inference** đặt lên VLM có sẵn:

**(A) Structured Thinking — 4 giai đoạn reasoning** (mỗi giai đoạn có tag đóng/mở):
| Stage | Tag | Chức năng |
|---|---|---|
| Summary | `<SUMMARY>...</SUMMARY>` | Tóm tắt cấp cao câu hỏi/vấn đề cần giải quyết |
| Caption | `<CAPTION>...</CAPTION>` | Miêu tả các yếu tố thị giác liên quan tới câu hỏi |
| Reasoning | `<REASONING>...</REASONING>` | Suy luận logic từng bước → đáp án sơ bộ |
| Conclusion | `<CONCLUSION>...</CONCLUSION>` | Đáp án cuối (output gửi user). 3 stage trước là "hidden" |

- 4 stage được sinh **trong 1 lần inference duy nhất**, model **tự chuyển giai đoạn** (không cần prompt bên ngoài sau khi train).
- Đây là **System-2 reasoning** (chậm, logic) đặt vào System-1.

**(B) Dataset LLaVA-CoT-100k** — ~99k image-QA pair, GPT-4o sinh 4-stage annotation:
ShareGPT4V(31.3k), ChartQA(17.2k), A-OKVQA(16.1k), AI2D(11.4k), GeoQA+(11.4k), ScienceQA(5.6k), DocVQA(4.0k), PISC(1.0k), CLEVR(0.5k), CLEVR-Math(0.5k).

**(C) Training** — SFT (full fine-tuning) trên base model. Paper gốc: **Llama-3.2-11B-Vision-Instruct**, 8×H100.

**(D) SWIRES (Stage-WIse Retracing Search)** — test-time scaling:
- Mỗi stage sinh **M candidate**.
- Nếu ≥1 candidate vượt **reward threshold** → giữ **top N** theo reward model, sang stage kế.
- Nếu KHÔNG candidate nào vượt threshold → **retrace** về stage trước, sinh lại; lặp tối đa **C lần**.
- Đáp án cuối = candidate reward cao nhất sau stage cuối.
- Dùng **reward model** ngoài để chấm điểm stage.

## 3. Workflow implement (adapt lên setup team)
Setup team: **Qwen2.5-VL-3B + Visual CoT–GQA + n=200 eval, train=256, QLoRA r=8, ~32 steps** (Kaggle/Colab T4/P100 free-tier).

1. **Load** Visual CoT (`deepcs233/Visual-CoT`), filter subset **GQA**.
2. **Build 4-stage target** từ mỗi sample (GQA đã có Reasoning steps + CoT BBox → map trực tiếp vào REASONING + CAPTION):
   - `<SUMMARY>` = tóm tắt câu hỏi (template).
   - `<CAPTION>` = mô tả vùng liên quan (từ CoT BBox).
   - `<REASONING>` = reasoning steps NL (đã có sẵn trong Visual CoT-GQA).
   - `<CONCLUSION>` = answer.
3. **SFT** Qwen2.5-VL-3B + QLoRA r=8 trên 256 sample, mask loss chỉ phần response.
4. **Inference**: sinh 1 lần, trích `<CONCLUSION>` → Acc=EM. macro Prec/Recall/F1 over gold classes (theo protocol team).
5. **(Optional/ablation) SWIRES**: best-of-N + stage-wise với scorer đơn giản.

## 4. Bảng module / chức năng / input / output
| Module | Input | Output |
|---|---|---|
| Data builder | sample GQA (image, Q, A, bbox, reasoning) | text 4-stage có tag |
| Tokenizer/Processor | image + prompt text | input_ids + pixel_values |
| Model (Qwen2.5-VL-3B + LoRA) | batch | logits → loss |
| Trainer (QLoRA) | dataset | adapter weights |
| Stage extractor | raw generation | `<CONCLUSION>` text |
| Evaluator | preds, golds | Acc,P,R,F1,... |
| SWIRES (opt) | model + reward | best answer |

## 5. Thông tin để code lại
- Base: `Qwen/Qwen2.5-VL-3B-Instruct` (paper dùng Llama-3.2-11B → ta adapt).
- Tags: `<SUMMARY>` `<CAPTION>` `<REASONING>` `<CONCLUSION>` (paper nêu rõ, Sec 3.1.1).
- Dataset GQA source có sẵn reasoning steps + bbox (Visual CoT paper Sec 3.1 + App E.3).
- Train: QLoRA r=8 (team protocol). Paper gốc full FT 8×H100 (không khả thi free-tier).
- Metrics: Acc=EM; macro P/R/F1; FPR/FNR; Train Time; Params (~3B + LoRA r=8); Steps.

## 6. Điểm paper NÓI RÕ
- 4 stage + tên tag + vai trò từng stage.
- 4 stage sinh trong 1 inference pass, model tự chuyển stage.
- LLaVA-CoT-100k: danh sách 12 nguồn + số lượng.
- Base model + full FT + 8×H100.
- SWIRES: thuật toán (M candidate, threshold, top N, retrace max C, reward model).
- Benchmarks: MMStar-R, MMBench-R, MMVet-R, MathVista, AI2D, HallusionBench.

## 7. Điểm paper KHÔNG NÓI RÕ (version này)
- **Hyperparam training (lr, epochs, batch size, optimizer)** — paper dẫn "Appendix C" nhưng **version ICCV open-access KHÔNG chứa Appendix C**.
- **Hằng số SWIRES (M, N, C max-retrace, reward threshold)** — "Appendix D" cũng **không có** trong version này.
- **Reward model cụ thể** — chỉ dẫn "a simple yet effective multi-modal reward model", không nêu tên/model rõ (1 citation).
- Tiêu chí chọn task reasoning-only (Appendix E) — không có.

> → Code dùng default hợp lý + comment `ASSUMPTION:` cho các điểm này.

## 8. Checklist đem đi code
- [x] Đã xác định paradigm: 4-stage structured CoT SFT + SWIRES
- [x] Tags, vai trò stage rõ
- [x] Map Visual CoT-GQA → 4-stage (CAPTION←bbox, REASONING←reasoning steps)
- [x] Điểm không rõ đã đánh dấu
- [x] Code Kaggle (file kèm: `code/12_LLaVA-CoT_kaggle.py` + `kernels/llava_cot_12/12_LLaVA-CoT_kaggle.ipynb`)
- [ ] Chạy → metrics → điền row 12

---

## 9. KẾT QUẢ VERIFY CODE vs PAPER (2026-08-05, Phúc)

Đã đọc lại paper đầy đủ (12 trang, ICCV open-access) và đối chiếu với code:

| Hạng mục | Paper | Code | Khớp? |
|---|---|---|---|
| 4 stage + tag | Sec 3.1.1, p.4: `<SUMMARY>/<CAPTION>/<REASONING>/<CONCLUSION>` | `TAG_*` + `build_four_stage` | ✅ |
| Single inference pass | Sec 3.1.1, p.4: "all stages completed by the model in a single inference pass" | 1 conversation user+assistant, sinh 1 lần | ✅ |
| Stage chuyển tự động | Sec 3.1.1: "transitioning between different stages without any external intervention" | SFT học format, inference không prompt thêm tag | ✅ |
| SFT data | Sec 3.1.2: LLaVA-CoT-100k (GPT-4o sinh 4 stage) | Visual CoT–GQA có sẵn reasoning+bbox → map 4 stage | ✅ (adapt) |
| Base model | Sec 3.1.2: Llama-3.2-11B-Vision-Instruct | Qwen2.5-VL-3B (protocol team) | ✅ (adapt) |
| Training | Sec 3.1.2: full FT 8×H100, Appendix C (thiếu) | QLoRA r=8, 32 steps (protocol) | ✅ (adapt) |
| SWIRES | Sec 3.2.2 p.5-6: M candidates, threshold, top N, retrace max C; reward model InternLM-XComposer2.5-Reward (Sec 5, p.7) | Chưa implement (USE_SWIRES=False) — blocker | ⚠️ |
| Eval | Sec 4.1: VLMEvalKit, 6 benchmarks | Visual CoT–GQA n=200, Acc=EM, macro P/R/F1 (protocol) | ✅ (adapt) |

### BUG đã tìm & fix (đã verify bằng processor thật Qwen2.5-VL-3B trên CPU)

1. **Loss masking under-mask** (collator): `plen = len(tok(prompt))` sai vì `<|image_pad|>` được expand 1→n token. Với ảnh 224×224, plen=52 nhưng prompt thật dài 115 → `<|im_start|>assistant\n` lọt vào loss.
   - Fix: tokenize prompt-only qua CÙNG processor (có image) rồi lấy length. Đã verify: 0 token prompt còn non-masked.
2. **CONCLUSION = full_answer thì EM=0**: gold là short answer (`hot dog`), nếu conclusion là "The food is a hot dog." → Acc=EM không bao giờ khớp.
   - Fix: CONCLUSION = `answer` (short) trước, fallback `full_answer`. Paper Sec 3.1.1 cho phép conclusion ngắn khi user muốn brief answer. (Đây là đề xuất adapt của tôi, paper không nói rõ.)
3. **bf16 fail trên P100** (Kaggle P100 không hỗ trợ bf16): tự chọn bf16 (Ampere+) / fp16 (Turing/Volta).

### Kết quả test local (CPU, không model)
- Dataset thật: train=88.294, val=9.855 (HF deepcs233/Visual-CoT, gqa_cot_*.jsonl).
- Field đúng: `question, answer, full_answer, thought, bboxs, image, reasoning` ✅
- `build_four_stage` tạo 4-stage đủ tag, extract được nội dung từng stage ✅ (test 5 mẫu + case bbox rỗng)
- Collator fix: verified với processor thật (mask chính xác, target không bị mask) ✅
- Metrics (EM, macro P/R/F1, FPR/FNR): chạy được trên mock data ✅

### Blocker chạy full
- **Không có GPU local** → chưa chạy được model forward/backward ở đây.
- Cần chạy notebook trên **Kaggle** (SMOKE=1 trước: n=4, 2 steps).
- Image GQA không nằm trong HF repo (chỉ có filename) → trên Kaggle cần dataset ảnh (`lyte69/gqa-images` hoặc `khoangoo/test-data` đã download). Code có 4 candidate path, cần xác nhận path thật lúc chạy.
- SWIRES chưa implement (cần reward model InternLM-XComposer2.5-Reward + Appendix D không có trong paper version này).
