# #13 Pix2Graph (PGSG) — Tóm tắt kiến trúc đã VERIFY (Prompt 1+2 equivalent)

> Nguồn: Li, Zhang, Lin, Chen, He. "From Pixels to Graphs: Open-Vocabulary Scene Graph Generation with Vision-Language Models", CVPR 2024.
> Framework tên paper: **PGSG** (Pixels to Scene Graph Generation with Generative VLM). Team gọi là Pix2Graph.
> Phương pháp verify: trích text trực tiếp từ PDF paper (pypdf, 11 trang; phần "supplementary materials" paper nhắc KHÔNG có trong PDF này → các HP training ghi "không nêu"). Chỗ paper không nói rõ → ghi thẳng/đánh dấu ASSUMPTION.
> Code gốc: https://github.com/SHTUPLUS/Pix2Grp_CVPR2024

## 1. Mục tiêu paper
SGG (Scene Graph Generation) mở vốn từ vựng (open-vocabulary) — parse ảnh thành graph `Gsg = {V, R}` (entity có category + bbox, relation triplet `r=(subject, predicate, object)`). Khác SGG truyền thống (classifier predicate trên entity proposal đóng), paper làm **end-to-end image→graph** với predicate unseen, bằng cách **biến SGG thành bài image-to-sequence** dùng VLM generative. Đưa ra: (i) scene-graph prompt + relation-aware tokens, (ii) module trích relation triplet (entity grounding + category conversion), (iii) dùng trọng số SGG làm init cho VL task downstream (VQA/captioning/grounding). Benchmark: PSG, Visual Genome (VG), OpenImages V6 (OIv6) → **R@K / mR@K** (K=20/50/100), cả novel class.

> ⚠️ **QUAN TRỌNG (để adapt):** Task gốc paper là **Scene Graph Generation** (KHÔNG phải VQA). Output là graph (triplet + bbox), metric R@K. Paper CÓ nhúng SGG vào VQA GQA (Tab 7) nhưng chỉ như **kiến trúc init transfer** (train SGG → fine-tune VQA), không phải "đưa graph vào prompt làm context". → Để khớp protocol team (VQA Acc), ta adapt theo Hướng A: **dùng method sinh scene graph → inject graph vào prompt Qwen → trả lời VQA** (xem §3). Recall@K chỉ đo được khi có gold scene-graph (Xem §3, §9).

## 2. Kiến trúc / method (paper nói rõ)
PGSG = image-to-sequence. 3 thành phần (Fig 2):

**(A) Scene Graph Sequence Generation (Sec 4.2)** — VLM (detector-free) sinh `ssg` dạng text từ ảnh.
- **Scene-graph prompt** (template chính xác):
  `"Generate the scene graph of [triplet sequence] and [triplet sequence] ..."`
  Gồm prefix `"Generate the scene graph of"` + K triplet, phân cách bằng `"and"` hoặc `","`.
- Mỗi **triplet sequence** theo grammar subject-predicate-object:
  `" t_i^v [ENT] t_ij^e [REL] t_j^v [ENT] "`
  trong đó `t_i^v, t_j^e, t_ij^e` = token category name của subject, object, predicate. **[ENT] và [REL] là relation-aware token đặc biệt** (đánh dấu compositional structure + vị trí entity).
- Text decoder nhận `Zv` (vision feature) + prefix → sinh `ssg` auto-regressive. "Giống image captioning" — tận dụng luôn vocabulary ngôn ngữ tự nhiên → sinh được predicate mới (open-vocab).

**(B) Relationship Construction (Sec 4.3)** — trích triplet từ `ssg`. Hai submodule:

- **(B1) Entity Grounding Module (Sec 4.3.1, Eq 1-3)** — predict bbox cho mỗi entity. Lấy hidden states của token sequence `"t_i^v [ENT]"` làm query. Average-pool + linear project → `q_i^ent` (Eq 1). Cross-attention giữa `2N` query entity với `Zv` qua transformer enc/dec (Eq 2) → FFN predict bbox `B` (Eq 3). Có ablation số layer `L ∈ {0,3,6,12}` (Tab 3: plateau tại L=6).
- **(B2) Category Conversion Module (Sec 4.3.2, Eq 4-5)** — **parameter-free**. Map vocabulary score `Ps` (không gian vocab ngôn ngữ `Cvoc`) → category score `pv, pe` (không gian category của benchmark `Ov, Oe`). Tokenize tên category → token index → index vào `Ps` lấy điểm (Eq 5). Nếu token sinh ra **khớp exact** tên category thì amplify điểm bởi `βi`.

**(C) Learning (Sec 4.4.1)** — multi-task loss:
`L = L_lm + L_pos`
- `L_lm`: standard autoregressive next-token LM loss (maximize `Σ log P(t_i | Z, t_<i; Θ_sg)` qua K token).
- `L_pos`: bbox regression = `GIOU(B, B_gt) + ||B - B_gt||_1`.

**(D) Inference (Sec 4.4.2)** —
- **Sinh ssg**: **nucleus sampling** + multiple round; mỗi ảnh sinh **M** sequence, max length **L** (không có giá trị cụ thể — ASSUMPTION).
- **Triplet construction**: heuristic rule match pattern `"subject [ENT] predicate [REL] object [ENT]"`; lấy bbox + category score + predicate score; chọn **top-3 category** mỗi thành phần.
- **Post-processing**: (1) bỏ relation tự nối (subject==object); (2) **NMS** khử redundant; (3) rank theo triplet score `St = ` tích điểm entity × entity × predicate.

**(E) Adaptation sang VL task (Sec 4.5)** — lấy `Θ_sg` (visual encoder + text decoder + token predictor) làm **init** khi fine-tune VLM cho VL task (VQA/grounding/captioning). Module không dùng trong SGG (text encoder) giữ pre-trained. Đây là cách paper "transfer relation knowledge".

**VLM backbone (paper nói rõ):** **BLIP** detector-free — ViT-B/16 (visual) + BERTbase (text decoder), ảnh đầu vào **384×384**. VL task còn dùng thêm BLIPv2. → Khác base model team (Qwen2.5-VL-3B).

## 3. Workflow ADAPT lên setup team (đề xuất, paper KHÔNG làm graph-prompt VQA)
Setup team: **Qwen2.5-VL-3B-Instruct (4-bit) + Visual CoT–GQA, eval n=200**. Paper dùng BLIP (khác base). PGSG nguyên bản cần train entity-grounding module + category conversion trên dataset SGG có bbox gold — nặng, và Visual CoT–GQA **không có gold scene-graph/bbox relation**. → Đề xuất 2 nhánh, **chọn nhánh inference-only làm default**:

**Nhánh A (KHUYẾN NGHỊ, inference-only, buildable trên Kaggle T4/P100):**
1. **Sinh scene-graph sequence bằng prompt** (bắt chước §4.2 nhưng zero-shot, không train): prompt Qwen2.5-VL-3B sinh theo đúng format paper — prefix `"Generate the scene graph of"` + output dạng `subject [ENT] predicate [REL] object [ENT], ...`. Dùng **nucleus sampling**, sinh **M=1–3** sequence, max length **L=256** (compute-avg).
2. **Parse triplet**: regex/heuristic match `(.+?) \[ENT\] (.+?) \[REL\] (.+?) \[ENT\]`; dedup, bỏ self-loop (giữ logic post-proc §D nhưng bỏ NMS vì không có bbox/score ở nhánh này — hoặc dùng confidence token nếu dễ).
3. **Graph-augmented VQA**: build text graph → đưa vào prompt Qwen dạng `"Scene graph: <triples>. Question: {Q}. Answer:"`. Qwen trả lời. So sánh với **baseline không graph** để đo gain do augmentation.
4. **Metric**: Acc=EM (vs gold `answer`), macro P/R/F1 over gold answer class, FPR/FNR. **ROC-AUC/PR-AUC: N/A** (không rerank candidate). Recall@K: xem §9.

**Nhánh B (đúng paper hơn, nặng hơn — tùy chọn nếu có gold graph):** QLoRA r=8 SFT Qwen2.5-VL-3B (train=256) sinh image→graph-sequence, cần gold scene-graph từ GQA gốc (map theo image id; Visual CoT-GQA có thể không chứa sceneGraph.json → phải load riêng). Mất entity-grounding module (Qwen không expose hidden states dễ) → chỉ replicate phần image-to-sequence + parse. Train Time, Params(QLoRA), Steps điền đủ.

> ĐÂY LÀ ADAPTATION (đề xuất của tôi). Paper không làm "graph-in-prompt VQA" — paper chỉ transfer trọng số. Ghi rõ Note khi điền sheet.

## 4. Bảng module / input / output
| Module | Input | Output |
|---|---|---|
| VLM image-to-seq (BLIP / Qwen) | image `I` + prefix `"Generate the scene graph of"` | scene-graph sequence `ssg` (triplet + [ENT]/[REL]) |
| Entity Grounding | hidden states của `"t^v [ENT]"` + image feat `Zv` | bbox `B` mỗi entity (Eq 1-3) |
| Category Conversion (param-free) | vocab score `Ps` + category set `Ov, Oe` | category score `pv, pe` (Eq 4-5) |
| Triplet parser (heuristic) | `ssg` | raw triplet `(sub, pred, obj)` + score |
| Post-proc (NMS, bỏ self-loop, rank) | raw triplet + score | scene graph `Gsg` |
| Graph→context injector (ADAPT) | graph text + câu hỏi Q | prompt VQA tăng cường |
| VLM VQA (Qwen, ADAPT) | image + graph-context + Q | answer |
| Evaluator (ADAPT) | preds, golds | Acc, P/R/F1, FPR/FNR; (Recall@K nếu có gold) |

## 5. Thông tin để code lại
- Base VLM (paper): **BLIP** (ViT-B/16 + BERTbase), ảnh **384×384**. → ADAPT: **Qwen2.5-VL-3B-Instruct** 4-bit.
- Scene-graph prompt template (§4.2): prefix `"Generate the scene graph of"`; triplet `"<sub> [ENT] <pred> [REL] <obj> [ENT]"`; sep `"and"`/`","`.
- Relation-aware tokens **[ENT], [REL]** — trong ADAPT nhánh A ta **không thêm token mới vào vocab** (zero-shot), mà yêu cầu Qwen output literal chuỗi `"[ENT]"`/`"[REL]"` rồi regex. (ASSUMPTION — paper thì thêm special token thật.)
- Inference: nucleus sampling, M sequence/ảnh, max length L. ADAPT: **M=1–3, L=256** (compute-avg).
- Loss (nhánh B): autoregressive LM loss + (`GIOU + L1`) cho bbox (paper §4.4.1). QLoRA r=8 trên Qwen.
- Top-3 category mỗi thành phần (§4.4.2); bỏ self-loop + NMS + rank theo tích điểm (§D).
- Entity grounding layer L=6 (Tab 3) — không áp dụng nhánh A (Qwen không expose hidden). Chỉ áp dụng nhánh B nếu xây module riêng.
- Recall@K cần gold scene-graph; nếu load GQA sceneGraphs.json theo image id thì tính được (Xem §9).

## 6. Điểm paper NÓI RÕ
- Paradigm image-to-sequence + scene-graph prompt chính xác + relation-aware token [ENT]/[REL] (Sec 4.2).
- Entity Grounding: công thức Eq 1-3 (avg-pool query + cross-attn + FFN), ablation L∈{0,3,6,12} plateau L=6 (Tab 3).
- Category Conversion parameter-free, Eq 4-5 (tokenize category → index vào Ps), amplify βi khi exact match.
- Loss multi-task `L_lm + L_pos` với `L_pos = GIOU + L1` (Sec 4.4.1).
- Inference: nucleus sampling + multiple round (M seq, max length L); heuristic parse pattern `"s [ENT] p [REL] o [ENT]"`; post-proc bỏ self-loop + NMS + rank tích điểm; top-3 category (Sec 4.4.2).
- Backbone BLIP ViT-B/16 + BERTbase, input 384×384; BLIPv2 cho 1 số VL task.
- Dataset/benchmark: PSG, VG, OIv6; 50% predicate = novel; protocol SGDet/PCls/SGCls; metric R@K, mR@K, mR@K novel, zR@K (zero-shot triplet); K=20/50/100.
- VL downstream: GQA accuracy theo question type (Tab 7: PGSG +1.7 overall, relation +1.9; BLIPv2 zeroshot 32.3→33.9); RefCOCO/+/g (Tab 6); COCO captioning Bleu4/CIDEr/SPICE (Tab 8).
- Sequence length analysis (Tab 4-5): SL=256→1.8s/ảnh, SL=512→2.2s, SL=768→4.8s, SL=1024→6.9s; % valid triplet 93–97%.
- Limitation: input resolution thấp (384) → miss object nhỏ; single-stage training → yếu detect object nhỏ.

## 7. Điểm paper KHÔNG NÓI RÕ / CHƯA TRÍCH ĐƯỢC
- **Learning rate, batch size, optimizer, epoch/schedule, số GPU** — paper ghi "More implementation details in supplementary materials"; **PDF này chỉ 11 trang, KHÔNG gồm supplementary** → KHÔNG xác định được. ASSUMPTION khi code (vd AdamW, lr 1e-5–2e-5).
- **M = số sequence sinh mỗi ảnh** (multiple round) — không cho giá trị cụ thể. ASSUMPTION: M=1–3.
- **Max length L cho experiment chính** — Tab 4 so sánh SL∈{256,512,768,1024} nhưng không nói "main result dùng SL=X". ASSUMPTION: main dùng SL=1024 (vì Tab 5 SL=1024 → 87.2 triplet, khớp nicely).
- **Nucleus sampling p (top-p)** — không cho giá trị. ASSUMPTION (vd p=0.9).
- **βi (hệ số amplify category conversion)** — không cho giá trị cụ thể. ASSUMPTION.
- **NMS threshold, score threshold filter** — không cho. ASSUMPTION.
- **[ENT]/[REL] có phải token mới add vào vocab không** — paper gọi "specified relation-aware tokens" → hiểu là special token thêm vào, nhưng không mô tả cách train embedding. ASSUMPTION.
- **Cách merge M sequence** (concat? union triplet? dedup?) — không rõ. ASSUMPTION: union + dedup triplet.
- Paper **không** làm "đưa graph vào prompt VQA" → toàn bộ Hướng A §3 là đề xuất của tôi.
- **Visual CoT–GQA có gold scene-graph/bbox không** — chưa verify trong data schema (question/answer/full_answer/image/bboxs/reasoning/thought) → Recall@K khả năng N/A (Xem §9).

> → Code dùng default hợp lý + comment `ASSUMPTION:` cho các điểm này.

## 8. Checklist đem đi code
- [x] Đã xác định paradigm: image-to-sequence (scene-graph prompt + [ENT]/[REL]) + relationship construction + (VL adapt)
- [x] Template prompt + token relation-aware rõ (Sec 4.2)
- [x] Adapt sang VQA: graph-augmented prompt (Hướng A, inference-only)
- [x] Điểm chưa rõ đã đánh dấu (LR/batch/M/L/β/NMS — supplementary không có)
- [ ] Code Kaggle (`code/13_Pix2Graph_kaggle.py`) — cấu trúc Config/Load/Preprocess/Model/Gen/Eval/Save
- [ ] Chạy → metrics → điền row 13 (Acc, P/R/F1, FPR/FNR; ROC-AUC/PR-AUC=N/A; Recall@K ở Note nếu có gold)

## 9. Rủi ro compute (lưu ý khi chạy)
- Nhánh A (inference-only) nhẹ: mỗi ảnh ~1–3 lần Qwen sinh (L=256) + 1 lần VQA. T4 16GB, fp16, 4-bit: ~0.5–1.5s/ảnh → n=200 khoảng 10–20 phút. Smoke (n=4) trước.
- Nhánh B (QLoRA SFT) nếu chạy: train=256, r=8, ~32 step; cần gold scene-graph GQA (load riêng sceneGraphs.json, map theo image id) — Visual CoT–GQA theo schema mô tả **không có** relation/bbox gold → chỉ tự build pseudo-label hoặc bỏ bbox loss.
- **Recall@K**: chỉ đo được nếu có gold scene-graph. Visual CoT–GQA (schema question/answer/full_answer/image/bboxs/reasoning/thought) — `bboxs` là bbox region của reasoning, KHÔNG phải scene-graph object. → **Recall@K = N/A cho sheet chính**, ghi Note. Có thể compute phụ nếu load GQA sceneGraphs.json gốc theo image id (GQA release có sceneGraphs_train/val.json đầy đủ object+relation).
- 1 model Qwen 3B 4-bit trên T4: OK (~2GB + activation). fp16 cho generate.
- Risk chính: Qwen zero-shot có thể không output đúng format `[ENT]/[REL]` → cần robust parser + fallback (nếu parse fail, dùng caption hoặc trả prompt VQA không graph). Đây là điểm yếu lớn nhất của nhánh A (Xem §7).
