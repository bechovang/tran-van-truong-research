# #11 COCO-Tree — Tóm tắt kiến trúc đã VERIFY (Prompt 1+2 equivalent)

> Nguồn: Sinha, Xiong, Zhang. "COCO-Tree: COmpositional Hierarchical COncept Trees for Enhanced Reasoning in Vision Language Models", EMNLP 2025 (main 135).
> Phương pháp verify: trích text trực tiếp từ PDF paper (pypdf, 17 trang) + đọc. Chỗ paper không nói rõ → ghi thẳng.

## 1. Mục tiêu paper
VLM yếu ở **compositionality** (hiểu quan hệ giữa nhiều object/attribute/relation). Paper tăng reasoning ngôn ngữ cho VLM bằng **concept tree neurosymbolic** do một LLM(reasoner) xây, + **beam-search path finding**, rồi **fuse** với prediction gốc của VLM (System-2 bù cho System-1). Kết quả: +5-10% trên Winoground/EqBench/ColorSwap/SugarCrepe với 7 VLM open-source. Code: github.com/sanchit97/compositionality-low-res-vlm.

> ⚠️ **QUAN TRỌNG (để adapt):** Task gốc paper là **image–text matching** (compositionality), KHÔNG phải VQA. VLM matching function `f: I×C→ℝ` cho alignment score. Bench: Winoground/EqBench/ColorSwap = cặp 2 ảnh + 2 caption → task Text/Image/Group; SugarCrepe = 1 ảnh, Text-only. Metric = **VQAScore** (xác suất token yes/no, discriminative). → Output của COCO-Tree là **SCORE**, không phải câu trả lời → cần adapt sang VQA (xem §3).

## 2. Kiến trúc / method (paper nói rõ)
3 bước xây concept tree + path finding:

**(A) Semantic Morphological Decomposition (SMD)** — `E = F_SMD(C)`, tách caption C thành M "morphological entity" (cụm tự chứa, structurally distinct, individually entail caption). Mặc định **M=2**.
- Prompt (App C, Fig 5): *"You are a helpful chatbot. Divide the caption into M smaller independent statements which entail the caption based on Subject and Object. Caption: {C}. Output format: 1. Subject 2. Object"*

**(B) Recursive Concept Exploration (RCE)** — `N_{l+1} = ∪ F_RCE(n_i, C, S)`, BFS-like, mỗi node sinh S concept con (binary visual concept để verify node), depth **L**, split **S**. Mặc định **S=3, L=3**.
- Prompt (App C, Fig 6): *"List {S} binary visual concepts to verify the {n_i}. Ensure the outputs are possible for {C}. Answer in small phrases and focus on verifiable things like objects, locations, actions, etc. Output: 1. xxx 2. xxx ..."*

**(C) Composite Vision-Language Score** mỗi node:
`CS(n) = α·LS(n,C) + (1−α)·VS(I,n)`  (Eq 7)
- **VS (Visual Score)** = `P_VLM("yes" | I, C)` — VLM trả lời yes/no "Does this figure show: C?" (Fig 7). Xác suất = softmax 2 nhãn yes/no (Eq 11).
- **LS (Linguistic Score)** = `P_LLM("yes" | C1, C2)` — LLM entailment "Given we observe C1. Is it possible C2?" (Fig 8). Softmax yes/no (Eq 12).
- **α=0.6** (Winoground/EqBench), **α=0.5** (ColorSwap).

**(D) Dynamic Path Selection** — path `p={e,n1,..,nl}` rooted ở entity e; weight `W_p={CS(n): n∈p}`. Hai biến thể:
- **Greedy (Max)**: mỗi bước chọn con có CS cao nhất (`SRCH_max`).
- **Beam**: giữ k path, chọn k node CS cao nhất, lấy path có tổng weight lớn nhất (`SRCH_beam`).
- System-2 output = path weight lý tưởng `Ŵ_p`.

**(E) Fusion** — `final = β·f(I,C) + (1−β)·Ŵ_p)` (Eq 8). **β=0.8**. `f(I,C)` = System-1 VLM score (VQAScore).

**(F) Neurosymbolic rule (interpretability)** — kết các node trên path bằng AND/OR → tạo rule giải thích. GPT-4o làm judge chấm entailment (Table 6: rule+caption > caption-only).

## 3. Workflow ADAPT lên setup team (đề xuất, paper KHÔNG làm VQA)
Setup team: **Qwen2.5-VL-3B + Visual CoT–GQA, eval n=200, inference-only** (không train).

**Vấn đề cốt lõi:** COCO-Tree cho ra **score** (rerank image-caption), không cho answer. → Adapt sang **candidate-rerank VQA**:
1. **System-1 (VLM)**: Qwen2.5-VL-3B sinh **K candidate answer** (sampling, dedup) cho câu hỏi. Cũng lấy greedy answer làm baseline EM.
2. **Concept tree mỗi candidate**: coi statement `S_c = "The answer to '{Q}' is '{candidate}'."` là "caption" gốc → SMD (M entity) → RCE (depth L, split S) → các concept verify.
3. **Score từng candidate**: VS (VLM yes/no trên ảnh cho mỗi concept) + LS (LLM entailment) → CS → path search (greedy/beam) → `Ŵ_c`. System-1 score `f_c = VS của chính statement S_c`.
4. **Final score** `= β·f_c + (1−β)·Ŵ_c` (Eq 8). Chọn candidate score cao nhất → predicted answer (EM vs gold).
5. **Metrics**: Acc=EM, macro P/R/F1, FPR/FNR; **ROC-AUC/PR-AUC candidate-level** (gold=positive, candidate khác=negative, score=final score) — đây là chỗ COCO-Tree tỏa sáng.

> ĐÂY LÀ ADAPTATION (đề xuất của tôi), paper không đề cập VQA. Ghi rõ ở Note khi điền sheet.

## 4. Bảng module / input / output
| Module | Input | Output |
|---|---|---|
| SMD (LLM) | statement S_c | M morphological entity |
| RCE (LLM) | entity + S_c | S concept con / node (đệ quy depth L) |
| VS (VLM) | image + concept | P(yes) softmax yes/no |
| LS (LLM) | concept + S_c | P(yes) softmax yes/no |
| Composite score | VS, LS | CS = αLS+(1−α)VS |
| Path search | cây + CS | Ŵ (greedy/beam) |
| Candidate gen (VLM) | image + Q | K candidate answer |
| Rerank | score mỗi candidate | predicted answer |
| Evaluator | preds, golds | Acc,P/R/F1,FPR/FNR,AUC |

## 5. Thông tin để code lại
- VLM (System-1): `Qwen/Qwen2.5-VL-3B-Instruct` 4-bit (sinh candidate + VS).
- LLM reasoner (SMD/RCE/LS): **Qwen2.5-3B-Instruct** 4-bit — cùng họ, cùng cỡ với VLM (paper dùng Llama-3.1-8B ≈ cỡ VLM; ta dùng 3B vì protocol). `temperature=0`.
- α=0.6, β=0.8 (mặc định Winoground).
- Tree: **M=2, S=2, L=2** (giảm từ 3/3/3 của paper cho vừa compute Kaggle với n=200). Greedy + Beam(k=2).
- K candidate = 2-4 (sampling). ROC-AUC/PR-AUC: thêm gold vào pool candidate khi tính AUC (eval setup, có note).
- Inference-only → Train Time = wall-clock inference, Training Steps = 0.

## 6. Điểm paper NÓI RÕ
- 3 bước SMD/RCE + công thức CS (Eq 7), VS (Eq 9/11), LS (Eq 10/12).
- Path search 2 biến thể Greedy/Beam + fusion Eq 8.
- Hyperparam: M=2, S=3, L=3, α=0.6/0.5, β=0.8; LLM=Llama-3.1-8B temp 0.
- Prompt template chính xác SMD/RCE/VS/LS (App C, Fig 5-8).
- 4 benchmark + 7 VLM; metric VQAScore.
- Compute O(M·S·L);Wilcoxon significance.

## 7. Điểm paper KHÔNG NÓI RÕ / CHƯA TRÍCH ĐƯỢC
- **Pseudocode Fig 9** render thành **ảnh**, không trích text → chi tiết implementation path search tôi suy ra từ mô tả Greedy/Beam (rõ trong text). ASSUMPTION.
- **Beam width k** — paper chỉ nói "select k maximum composite score nodes", không cho giá trị k cụ thể. → ASSUMPTION: k=2.
- Cách cụ thể kết hợp score沿 path (mean? sum? product?) — paper viết `W_p={CS(n): n∈p}` rồi "maximum path weight" → tôi dùng **mean CS along path** (ASSUMPTION; sum cũng hợp lý).
- Paper không làm VQA → toàn bộ §3 adaptation là đề xuất của tôi.

> → Code dùng default hợp lý + comment `ASSUMPTION:` cho các điểm này.

## 8. Checklist đem đi code
- [x] Đã xác định paradigm: concept-tree neurosymbolic + composite score + beam path + System-1/2 fusion
- [x] Prompts SMD/RCE/VS/LS rõ (App C)
- [x] Adapt sang VQA: candidate-rerank
- [x] Điểm chưa rõ đã đánh dấu (k beam, aggregation path, pseudocode)
- [ ] Code Kaggle (`code/11_COCO-Tree_kaggle.py`)
- [ ] Chạy → metrics → điền row 11

## 9. Rủi ro compute (lưu ý khi chạy)
- Số node = M·S·L ≈ 2·2·2 = 8/candidate × K candidate × n sample. Với K=3, n=200 → ~4800 VLM forward (VS) + LLM calls. Trên T4 ~30-60 min. Smoke (n=4) trước.
- 2 model (VL 3B + LLM 3B) cùng 4-bit trên T4 16GB: OK (~2GB mỗi model + activation).
- LS/RCE/SMD là text-only → batch được để tăng tốc.
