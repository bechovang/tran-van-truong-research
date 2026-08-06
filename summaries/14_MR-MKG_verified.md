# #14 MR-MKG — Tóm tắt kiến trúc đã VERIFY (Prompt 1+2 equivalent)

> Nguồn: Lee, Wang, Li, Zhang. "Multimodal Reasoning with Multimodal Knowledge Graph (MR-MKG)", ACL 2024 Long (paper 2024.acl-long.579, pp. 10767–10782).
> Phương pháp verify: trích text trực tiếp từ PDF (pypdf, 16 trang) + đọc kỹ toàn bộ (kể cả Appendix A). Chỗ paper không nói rõ → ghi thẳng vào §7 (ASSUMPTION).

## 1. Mục tiêu paper
LLM/VLM hay hallucinate và bị thiếu/outdated knowledge. KG text-only chỉ 1 modality → giới hạn cross-modal understanding. Paper đề xuất **MR-MKG**: nạp **multimodal knowledge graph (MMKG)** vào LLM bằng (1) **RGAT** (Relation-aware Graph Attention Network) encode sub-MMKG → knowledge node embeddings, (2) **knowledge adapter + visual adapter** (linear + single-head attention) map về word-embedding space của LLM, (3) **cross-modal alignment** bằng Triplet loss trên image-entity vs text-entity trong MMKG. Freeze LLM + visual encoder, chỉ train adapter/RGAT/alignment (~2.25% params LLM). Kết quả SOTA trên ScienceQA (Acc 92.78% với FLAN-T5-11B, 93.63% với FLAN-UL2-19B) và MARS (Hits@1 0.405 vs baseline 0.286, +10.4%).

> ⚠️ **QUAN TRỌNG (để adapt):** Task gốc paper là **ScienceQA** (multiple-choice multimodal science QA, chỉ 48.7% sample có ảnh) và **MARS** (multimodal analogical reasoning, dự đoán entity). Base model paper là **FLAN-T5-3B/11B, FLAN-UL2-19B, LLaMA-2-7B** (text LLM + visual encoder riêng, KHÔNG phải VLM end-to-end). Visual encoder là **CLIP ViT-L/32** (frozen). → Adapt lên team protocol phải đổi cả base (Qwen2.5-VL-3B, đã có visual encoder built-in) lẫn dataset (Visual CoT–GQA) — xem §3.

## 2. Kiến trúc / method (paper nói rõ, §3 + §A)

MR-MKG có **5 thành phần** (Fig 2): language encoder, visual encoder, KG encoder (RGAT), knowledge adapter, cross-modal alignment module.

**(A) Language Encoder** — lấy embedding layer có sẵn của LLM (LLaMA hoặc T5). **Freeze** cả train + inference. Output: text embedding `H_T`.

**(B) Visual Encoder** — **CLIP ViT-L/32** (Radford 2021), **frozen**. Image → visual feature `X_I`. Visual adapter = linear map sang dim word-embedding của LLM, sau đó **single-head attention** với `H_T`:
- `H_I = W_I · X_I + b_I`  (Eq 1, visual adapter linear)
- `H'_I = Softmax(H_T · H_I^⊤ / sqrt(d_k)) · H_I`  (Eq 2, attention với text query; `d_k` = dim của `H_T`)

**(C) KG Encoder — RGAT** (Ishiwatari 2020, relation-aware GAT). Từ text/image, retrieve subgraph `G` gồm **Top-N triple** liên quan nhất từ MMKG. Vì nhét raw triple vào prompt gây noise + mất structure → dùng RGAT:
- Khởi tạo node embedding + relation embedding bằng **CLIP**.
- `X_K = f_RGAT(G)`  (Eq 3, knowledge node embeddings có encode graph structure)
- **Hop distance cho retrieval = 1**; **Top-N ∈ {10, 20}** (paper test cả hai, tốt nhất rơi vào khoảng này, xấu đi khi >20).

**(D) Knowledge Adapter** — map `X_K` về text-embedding space của LLM, tương tự visual adapter:
- `H_K = W_K · X_K + b_K`  (Eq 4)
- `H'_K = Softmax(Q · H_K^⊤ / sqrt(d_k)) · H_K`  (Eq 5; `Q` = `H_T` hoặc `H_I` tuỳ scenario — paper nói "based on the specific scenario at hand", không fix).

**(E) Cross-Modal Alignment** — chọn ngẫu nhiên 1 tập **image entity** từ `G`, yêu cầu model match với text entity tương ứng. Dùng **Triplet Loss** (Schroff 2015):
- `L_a = Σ_i max(d(x_a, x_p) − d(x_a, x_n) + α, 0)`  (Eq 6)
- `x_a` = embedding image-entity (anchor), `x_p` = text-entity tương ứng (positive), `x_n` = các text-entity khác (negative).
- `d` = **Euclidean distance**. `M` = số image entity được chọn. `α` = margin constant.

**Prompt construir + Loss:**
- `prompt = H'_K ⊕ H'_I ⊕ H_T` (concat knowledge + visual + text embedding).
- Generative loss: `L_g = Σ_i log p(A_i | prompt, A_{0:i−1}; θ_a)`  (Eq 7, auto-regressive).
- **Total: `L = L_g + λ · L_a`**  (Eq 8).

**Training 2 stage** (cả 2 đều freeze LLM + visual encoder):
1. **Pretrain** trên **MMKG-grounded dataset** (xây từ Visual Genome, 18,448 instance — xem §A.3) để có nền visual + hiểu MMKG.
2. **Fine-tune** trên downstream task (ScienceQA hoặc MARS).

**MMKG-Grounded Dataset (§A.3, rất quan trọng để adapt):** xây từ **Visual Genome (VG)**. Mỗi instance = (image, QA pair, **modified scene graph**). Cụ thể: object entity trong scene graph được link với **crop ảnh của object theo bbox** qua quan hệ **"image of"**, và link với **attribute** qua quan hệ **"attribute of"**. Chỉ dùng **Region-based QA** của VG. → Đây chính là template để ta build MMKG từ scene graph GQA. (Fig 6 minh hoạ: scene graph các object "man/clock/jacket" + relation "behind/wearing/talking" + attribute "green/grey/tall" → MMKG bằng cách thêm các edge "image of" và "attribute of".)

**Retrieval scheme (§A.6):** embed text/image + tất cả triple của MMKG vào cùng không gian → cosine similarity → Top-n triple cho ra tập entity `E'` → lấy **1-hop neighbour** của `E'` + relation → chọn **Top-N** triple liên quan nhất bằng cosine similarity.

## 3. Workflow ADAPT lên setup team (đề xuất)
Setup team: **Qwen2.5-VL-3B-Instruct** (4-bit QLoRA r=8 nơi train) + **Visual CoT–GQA**, eval n=200, train=256, ~32 steps. Paper dùng base khác (FLAN-T5/LLaMA-2) → adaptation, không reproduction.

**Câu hỏi then chốt team đã flag:** Visual CoT–GQA có cung cấp scene graph để build MMKG không?
→ **ĐÁP (đã verify notebook + paper): KHÔNG.** Visual CoT–GQA records chỉ có các field `image` (đường dẫn GQA image), `conversations` (chứa Q/A + bbox + thought/reasoning), `dataset`, `split` — **không có scene graph**. **NHƯNG** GQA release gốc **CÓ sẵn scene graph** (objects + attributes + relations per image, nguồn: VG-derived). → Lấy scene graph theo **image id** từ GQA release (file `sceneGraphs.json` trên GQA website / Kaggle dataset GQA-SceneGraphs).

**Adaptation đề xuất (buildable, nhất quán với paper §A.3):**
1. **Build MMKG mỗi ảnh từ GQA scene graph** (theo đúng §A.3 của paper): với mỗi object `o` trong scene graph → node entity `o`; relation `r` giữa 2 object → edge `(o1, r, o2)`; mỗi object crop theo bbox → node ảnh, nối với `o` qua edge **"image of"**; mỗi attribute `a` của `o` → node attribute, nối qua **"attribute of"**. Đây chính là recipe paper mô tả cho VG.
2. **RGAT nhẹ**: node/relation init bằng CLIP embedding (dùng CLIP text encoder cho object/attribute name, CLIP image encoder cho crop) → 2-layer RGAT (ASSUMPTION: số layer paper không fix, chỉ nói "appropriate stacking") → `X_K` (graph embedding pooling hoặc danh sách node embedding).
3. **Knowledge adapter** = linear `W_K` + attention với `H_T` (Eq 4–5) → `H'_K` (vài soft token, ví dụ 4–8). Vì Qwen2.5-VL đã có visual encoder riêng → ta **KHÔNG cần replicate visual adapter (B)**, chỉ cần knowledge adapter.
4. **Inject** `H'_K` làm **soft prompt token** prepended vào câu hỏi (gần đúng cách paper concat `H'_K ⊕ H'_I ⊕ H_T`). Với QLoRA, soft token này nối vào input embedding của Qwen2.5-VL-3B.
5. **Cross-modal alignment** (tuỳ chọn, vì compute Kagggle chật): triplet loss (Eq 6) với anchor = image-entity embedding, positive = text-entity embedding tương ứng, negative = các text-entity khác cùng batch.
6. **Train** QLoRA r=8 trên Qwen2.5-VL-3B + train luôn RGAT + knowledge adapter (cả 3 cực nhỏ). Loss = `L_g + λ·L_a` (Eq 8). 32 step, batch small, fp16.
7. **Metric**: Acc=EM của answer, macro P/R/F1 over gold answer classes, FPR/FNR. ROC-AUC/PR-AUC = **N/A** (không có rerank, trừ khi mình mở rộng candidate-rerank như #11). Hits@1 (paper-native cho MARS) → ghi ở Note.

**Fallback nếu RGAT quá nặng cho P100/T4 (ASSUMPTION, paper KHÔNG làm vậy):** **Verbalize scene graph** thành text triple (e.g. `"(man, behind, clock); (clock, attribute, green); ..."`) nhồi vào text prompt của Qwen2.5-VL-3B + QLoRA. Đỡ trung thực với paper hơn (mất RGAT) nhưng cùng tinh thần "inject MMKG as knowledge". Khi đó metric vẫn tính như trên, Note ghi rõ "RGAT replaced by verbalized scene graph".

> ĐÂY LÀ ADAPTATION (đề xuất của tôi), paper không dùng Qwen2.5-VL-3B và không đánh giá trên GQA. Ghi rõ ở Note khi điền sheet.

## 4. Bảng module / input / output
| Module | Input | Output |
|---|---|---|
| GQA scene-graph loader | image id | scene graph (objects/attributes/relations + bbox) |
| MMKG builder | scene graph | MMKG `G` (entity node + image-of + attribute-of edges) |
| Subgraph retriever | question text + `G` | Top-N triple subgraph (paper: Top-N=10/20, hop=1) |
| CLIP encoder (init) | object/attr name + crop | node/relation init embedding |
| RGAT encoder | subgraph `G` + init | `X_K` (knowledge node embeddings) |
| Knowledge adapter | `X_K`, `H_T`/`H_I` | `H'_K` (soft prompt token) |
| Qwen2.5-VL-3B (frozen 4-bit + QLoRA) | image + question + `H'_K` | generated answer |
| Triplet align loss | image-entity emb, text-entity emb | `L_a` |
| Generator loss | logit answer | `L_g` |
| Evaluator | preds, golds | Acc,P/R/F1,FPR/FNR,Hits@1 |

## 5. Thông tin để code lại
- Base model: **`Qwen/Qwen2.5-VL-3B-Instruct`** 4-bit (BitsAndBytes) + **QLoRA r=8** (peft). Paper dùng FLAN-T5/LLaMA-2 — KHÔNG dùng Qwen → đây là điểm khác so với paper.
- Visual encoder: dùng **của chính Qwen2.5-VL-3B** (không cần CLIP riêng như paper). Chỉ dùng CLIP (e.g. `openai/clip-vit-large-patch32`) cho **khởi tạo node/relation embedding** của RGAT (giống paper §3.2 "we first use CLIP to initialize node and relation embeddings").
- RGAT: tự implement (PyG có `RGAT` không chính thức; có thể implement theo Ishiwatari 2020) hoặc dùng `torch_geometric.nn.GAT` + relation encoding. **2 layer, hidden 256** (ASSUMPTION — paper không đưa số layer/dim cụ thể, chỉ khảo sát đồ thị Figure 4).
- Knowledge adapter: `Linear(RGAT_dim → Qwen_embed_dim)` + single-head attention theo Eq 4–5.
- Top-N triple = **10** (giá trị an toàn trong khoảng 10–20 paper test).
- Loss: `L = L_g + λ·L_a` với **λ=0.5, α (margin)=0.5** (ASSUMPTION — paper không đưa giá trị λ và α cụ thể).
- Scene graph GQA: load `gqa_sceneGraphs.json` (file ~600MB), map theo `image id` trong Visual CoT–GQA. Field `image` trong Visual CoT–GQA có dạng path GQA (vd `GQA/imgs/...` hoặc id thuần) → tách id để join.
- Hyperparam train team: lr=2e-5 (ASSUMPTION,ScienceQA dùng 4e-5; QLoRA hay dùng 2e-4 — đề xuất 1e-4 tới 2e-4 cho QLoRA), batch=1–2, 32 step, AdamW, weight_decay=0.01, fp16 (P100/T4 **không hỗ trợ bf16**).
- Metric: Acc=EM (chuẩn hoá lower/strip), macro P/R/F1 over gold answer class, FPR/FNR, Hits@1 (ghi Note). ROC-AUC/PR-AUC: N/A (không rerank). Train Time = wall-clock, Params = RGAT + adapter + LoRA, Steps = 32.

## 6. Điểm paper NÓI RÕ
- Kiến trúc 5 thành phần + công thức Eq 1–Eq 8 (visual adapter, RGAT, knowledge adapter, triplet alignment, generative + tổng loss).
- 2-stage training (pretrain trên MMKG-grounded dataset → fine-tune downstream). LLM + visual encoder **freeze** toàn bộ.
- **MMKG-grounded dataset xây từ Visual Genome**: object entity → "image of" (crop bbox), attribute → "attribute of", chỉ dùng Region-based QA, 18,448 instance (§A.3, Fig 6).
- Retrieval: cosine similarity trên CLIP space → Top-n triple → 1-hop neighbour → Top-N (§A.6). Hop=1, Top-N ∈ {10, 20}.
- Visual encoder = CLIP **ViT-L/32**; RGAT init bằng CLIP; backbones = FLAN-T5-3B/11B, FLAN-UL2-19B, LLaMA-2-7B.
- Hyperparam (Appendix A.7 + Table 9):
  - ScienceQA: 3 epoch, lr 4e-5, max token 512, batch 1, AdamW, wd 0.01. Multimodal-CoT prompting.
  - MARS: MarKG pretrain 3 epoch lr 2e-5 batch 8 seq 96; MARS fine-tune 3 epoch lr 5e-6 batch 4 seq 128.
  - MMKG-grounded pretrain: 2 epoch, lr 5e-5, in/out 512/128, batch 2, AdamW, wd 0.01.
- Trainable param ~2.25% LLM (FLAN-T5-3B: 77M; FLAN-T5-11B: 248M; FLAN-UL2-19B: 248M).
- Metric: ScienceQA → Accuracy; MARS → Hits@k, MRR.
- Hardware: 8×A800-80GB.
- Kết quả: ScienceQA 92.78% (T5-11B), 93.63% (UL2-19B); MARS Hits@1 0.405 (+10.4%).
- Ablation: +KG +5.66%, +MMKG (text→multi) +0.47% (full) / +1.41% (image subset), +Alignment +0.15% / +0.54%, +Pretrain +0.42%. KGE compare: GNN 92.23 / GAT 91.94 / RGAT 92.78 → RGAT tốt nhất. Retrieval: Text-only > Text+Image > Image-only.

## 7. Điểm paper KHÔNG NÓI RÕ / CHƯA TRÍCH ĐƯỢC
- **Số layer RGAT cụ thể** — paper chỉ khảo sát đồ thị (Fig 4) và nói "appropriate stacking … can positively affect", không đưa con số chính xác. → ASSUMPTION: 2 layer.
- **Hidden dim của RGAT / adapter** — không nói. → ASSUMPTION: 256.
- **λ (trade-off loss `L_g` vs `L_a`)** — không đưa giá trị. → ASSUMPTION: 0.5.
- **α (margin trong triplet loss)** — chỉ gọi là "constant used to ensure a certain margin". → ASSUMPTION: 0.5 (giá trị FaceNet cổ điển).
- **Cách chọn `Q` trong Eq 5** (`H_T` hay `H_I`) — paper nói "based on the specific scenario at hand", không fix quy tắc. → ASSUMPTION: dùng `H_T` (text query) cho simplification.
- **Cách sample image entity cho alignment** — "selecting a set of image entities from G at random" — số lượng `M` mỗi step không nói. → ASSUMPTION: 4–8 entity/batch.
- **Top-n (retrieval bước 1) khác Top-N (bước cuối)** — paper phân biệt `n` vs `N` nhưng không đưa giá trị `n` cụ thể, chỉ nói `N ∈ {10, 20}`. → ASSUMPTION: n=20.
- **Multimodal-CoT prompting chi tiết** — paper引用 Zhang 2023c (MM-CoT) nhưng không reproduce prompt; nhóm cần tự lấy từ paper MM-CoT gốc (ASSUMPTION: thay bằng zero-shot CoT đơn giản cho Visual CoT–GQA).
- **Toàn bộ phần adapt lên Qwen2.5-VL-3B + Visual CoT–GQA (§3)** là đề xuất của tôi, paper không làm. Đặc biệt: Qwen2.5-VL có sẵn visual encoder → bỏ visual adapter (B); scene graph phải lấy từ GQA release (Visual CoT–GQA không có field scene graph — đã verify notebook).

> → Code dùng default hợp lý + comment `ASSUMPTION:` cho các điểm này.

## 8. Checklist đem đi code
- [x] Đã xác định paradigm: RGAT over MMKG + knowledge adapter + cross-modal triplet alignment + frozen-LLM PEFT (2-stage).
- [x] Đã nắm recipe build MMKG từ scene graph (§A.3): image-of + attribute-of.
- [x] Đã xác nhận Visual CoT–GQA KHÔNG có scene graph → phải load riêng từ GQA release theo image id.
- [x] Adapt sang Qwen2.5-VL-3B + QLoRA r=8: train RGAT + knowledge adapter + LoRA; bỏ visual adapter (Qwen đã có).
- [x] Điểm chưa rõ đã đánh dấu (layer RGAT, λ, α, dim, Q, M, Top-n).
- [ ] Code Kaggle (`code/14_MR-MKG_kaggle.py`) — cấu trúc: Config / Load / Preprocess / Model / Train / Eval / Save.
- [ ] Chạy → metrics → điền row 14. Ghi Note: `Qwen2.5-VL-3B 4bit QLoRA r=8; Visual CoT–GQA; n=200; RGAT-MMKG from GQA scene graphs + knowledge adapter + triplet align; Hits@1=N/A (VQA); adaptation, not full MR-MKG reproduction.`

## 9. Rủi ro compute (Kaggle P100/T4 16GB, fp16, không bf16)
- **Lớn nhất: load GQA scene graph (~600MB JSON)** + build MMKG mỗi sample → cache dict image_id→graph vào RAM/numpy lúc preload. Nếu OOM: lọc trước top-200 train + top-200 eval image id, chỉ giữ scene graph của những id đó.
- RGAT nhẹ (2 layer, hidden 256, ~1–2M param) + Qwen2.5-VL-3B 4-bit (~2GB) + LoRA r=8 + adapter nhỏ → vừa T4 16GB nếu batch=1. Soft token 4–8 → không tăng sequence length nhiều.
- Triple alignment triplet loss cần ≥2 entity/sample → đảm bảo sampler lấy sample có ≥2 image-entity (GQA scene graph trung bình nhiều object → OK).
- 32 step train không đủ hội tụ cho RGAT từ scratch → cân nhắc **freeze RGAT ở init CLIP** (không train RGAT, chỉ train LoRA + adapter) làm baseline nhanh; hoặc train RGAT từ kết quả warm-up của pretrain stage (paper chia 2 stage, ta có thể giảm pretrain xuống vài chục step).
- Nếu RGAT implement quá rủi ro → fallback verbalized scene graph (xem §3), vẫn báo cáo được Acc/P/R/F1; Note rõ "RGAT replaced by verbalization".
- fp16 обязатель trên P100/T4; gradient checkpointing cho Qwen2.5-VL-3B để tiết kiệm VRAM.
