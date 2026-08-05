# KẾ HOẠCH IMPLEMENT 5 PAPER (Phúc) — JOUR-2026, sheet "Implement", dòng 11–15

> Mục tiêu: áp dụng ý tưởng chính của mỗi paper lên setup thống nhất của team, chạy Kaggle, lấy metrics điền vào sheet. Ghi chú: đây là **adaptation** (giống Khoa/Thịnh đã làm), KHÔNG phải reproduction đầy đủ.

---

## 0. PROTOCOL THỐNG NHẤT (rút ra từ các dòng Khoa/Thịnh đã làm — rows 7–17)

| Hạng mục | Giá trị chuẩn |
|---|---|
| Base model | **Qwen2.5-VL-3B** (inference-only thì dùng 4-bit) |
| Dataset | **Visual CoT–GQA** (subset GQA của Visual CoT) |
| Eval subset | **n = 200** (một số paper n=500/444) |
| Training (nếu có) | train **256**, eval **200**, **QLoRA r=8**, ~32 steps |
| Cách tiếp cận | **Adaptation, not reproduction** — áp ý tưởng paper lên setup chung |
| Metrics bắt buộc | Acc, Prec, Recall, F1, ROC-AUC, PR-AUC, FPR, FNR, Train Time, Params, Training Steps |
| Cột Note mẫu | `Qwen2.5-VL-3B; Visual CoT–GQA; n=200; <chi tiết method>; adaptation, not full <X> reproduction.` |
| Định nghĩa metric | Acc = EM (exact match). Prec/Recall/F1 = **macro P/R/F1 over gold answer classes** (như row 8). ROC-AUC/PR-AUC = candidate-level khi có rerank. |
| Metric đặc thù | ghi thêm ở cột Note (vd mIoU, Recall@K, candidate recall@4...) |

**File tham chiếu phải xin team:** notebook Kaggle `test_dataset_visual_cot` của KhoaNgoo — đây chính là template load Visual CoT–GQA + chạy Qwen2.5-VL-3B. (Hiện notebook đang có error ở version mới → lấy version run thành công `run/335352904`.)

---

## 1. BẢNG TỔNG QUAN 5 PAPER

| # | Paper | Task gốc | Benchmark/metric gốc | Là VQA? | Khớp protocol GQA? | Khó |
|---|---|---|---|---|---|---|
| 11 | **COCO-Tree** (EMNLP'25) | Compositional reasoning: concept tree + LLM reasoner + beam search | Winoground/EqBench/ColorSwap/SugarCrepe → accuracy | KHÔNG (image-text matching) | Cần adapt | 2 |
| 12 | **LLaVA-CoT** (ICCV'25) | VLM reasoning 4 giai đoạn (Summary/Caption/Reasoning/Conclusion) + SWIRES | reasoning benchmarks (MMStar, MathVista…) → Acc | CÓ | Khớp tốt nhất | 2 |
| 13 | **Pix2Graph / PGSG** (CVPR'24) | Open-vocab Scene Graph Generation (image→graph sequence) | PSG/OpenImages/VG → Recall@K | KHÔNG (SGG) | Cần adapt | 3 |
| 14 | **MR-MKG** (ACL'24) | Multimodal reasoning + MMKG (RGAT + adapter + cross-modal align) | ScienceQA + analogy → Acc, Hits@1 | CÓ | Khớp | 2 |
| 15 | **LLM4SGG** (CVPR'24) | Weakly-supervised SGG (LLM CoT+few-shot extract triplet) | VG/GQA → Recall@K | KHÔNG (SGG) | Cần adapt | 3 |

> **Repo code gốc** (dùng tham khảo, không copy verbatim):
> - COCO-Tree: github.com/sanchit97/compositionality-low-res-vlm
> - LLaVA-CoT: github.com/PKU-YuanGroup/LLaVA-CoT
> - Pix2Graph: github.com/SHTUPLUS/Pix2Grp_CVPR2024
> - MR-MKG: (paper không public link rõ — kiểm tra lại; repo thường tên MR-MKG)
> - LLM4SGG: github.com/rlqja1107/torch-LLM4SGG

---

## 2. VẤN ĐỀ CỐT LÕI CẦN QUYẾT (quan trọng)

**3/5 paper KHÔNG phải là VQA** (#11 COCO-Tree, #13 Pix2Graph, #15 LLM4SGG) → output/metric gốc không khớp bảng Implement (đang là Acc/Prec/Recall/F1/ROC-AUC...).

**Hai hướng xử lý (cần Phúc chốt với lead):**

- **Hướng A — Adapt sang "graph/concept-augmented VQA" (KHUYẾN NGHỊ, đồng nhất với team):**
  - Dùng method của paper để sinh **scene graph / concept tree** cho ảnh Visual CoT–GQA.
  - Đưa graph/tree đó vào prompt của Qwen2.5-VL-3B như context → trả lời VQA.
  - Báo cáo Acc/Prec/Recall/F1/... như các dòng khác (so sánh công bằng). Ghi thêm metric gốc (Recall@K) ở Note.
- **Hướng B — Báo cáo metric gốc của paper (Recall@K cho SGG):**
  - Để cột Acc..FNR trống/N-A, chỉ ghi metric gốc ở Note → MẤT khả năng so sánh chéo với team.

> Mặc định đề xuất: **Hướng A** cho #13, #15; #11 làm "concept-tree augmentation" inference-only.

---

## 3. CÁCH ADAPT TỪNG PAPER (đề xuất chi tiết)

### #12 LLaVA-CoT — KHỚP NHẤT, làm trước
- **Ý tưởng:** SFT Qwen2.5-VL-3B (QLoRA) để sinh reasoning 4 giai đoạn có tag `<SUMMARY>/<CAPTION>/<REASONING>/<CONCLUSION>`. (Optional: SWIRES test-time search.)
- **Data train:** Visual CoT–GQA **đã có CoT annotation + bbox** → dựng thành 4 stage (caption = region caption từ bbox; summary/reasoning/conclusion từ CoT steps).
- **Setup:** train=256, eval=200, QLoRA r=8.
- **Metric:** Acc=EM của phần `<CONCLUSION>`; macro P/R/F1; + (optional) SWIRES best-of-N → ROC-AUC/PR-AUC candidate-level.

### #14 MR-MKG — KHỚP
- **Ý tưởng:** Freeze LLM+visual encoder, train adapter (~2.25% param). RGAT encode MMKG (từ scene graph ảnh) → knowledge adapter → inject vào prompt; + cross-modal alignment loss.
- **Data:** MMKG từ scene graph của ảnh GQA (GQA **có sẵn scene graph** trong release gốc — cần xác nhận có trong Visual CoT không, nếu không thì build từ GQA scene graph bằng id ảnh).
- **Setup:** train adapters/LoRA, eval=200.
- **Metric:** Acc, macro P/R/F1, Hits@1 (ghi Note), train time, params (chỉ adapter).

### #11 COCO-Tree — adapt inference-only
- **Ý tưởng:** LLM reasoner phân rã câu hỏi VQA thành **concept tree** (entities/attributes/relations) → query Qwen2.5-VL-3B theo từng concept → **beam search** trên cây → augment đáp án cuối.
- **Setup:** inference-only (Qwen2.5-VL-3B + 1 LLM reasoner).
- **Metric:** Acc, macro P/R/F1, FPR/FNR. Không train → Train Time = inference wall-clock, Steps=0.

### #13 Pix2Graph (PGSG) — adapt graph-augmented VQA (Hướng A)
- **Ý tưởng:** PGSG = image→graph sequence (relation-aware tokens) bằng VLM. Dùng nó **sinh scene graph** cho ảnh → đưa graph vào prompt → Qwen2.5-VL-3B trả lời VQA.
- **Setup:** inference-only (sinh graph) + VQA. Nếu muốn đúng paper: QLoRA PGSG-style image-to-sequence trên GQA (train=256).
- **Metric:** Acc VQA (graph-augmented) + Recall@K graph (Note).

### #15 LLM4SGG — adapt graph-augmented VQA (Hướng A)
- **Ý tưởng:** LLM (CoT + few-shot) trích triplet từ **caption ảnh** → align class → scene graph → đưa vào prompt → Qwen2.5-VL-3B trả lời VQA.
- **Setup:** inference-only. (GQA có caption qua image, hoặc lấy caption do VLM tự sinh.)
- **Metric:** Acc VQA (graph-augmented) + Recall@K (Note).

---

## 4. WORKFLOW 3 BƯỚC (lead yêu cầu)

Theo quy trình lead:
1. **GPT Prompt 1** — đọc paper → trích kiến trúc/workflow (cấm bịa, ghi "paper không nêu rõ").
2. **GPT Prompt 2** — verify lại, sửa suy diễn → bản tóm tắt sạch.
3. **Claude (tôi)** — nhận (bản tóm tắt đã verify + paper) → viết code Python Kaggle, tách Config/Load/Preprocess/Model/Train/Eval, in đủ metrics, xuất CSV/DataFrame.

**Lưu ý:** tôi (Claude) có thể đóng luôn bước 1+2 (tự đọc paper + tự verify) nếu Phúc muốn nhanh — vẫn giữ nguyên yêu cầu "không bịa, ghi rõ chỗ paper không nêu".

---

## 5. THỨ TỰ + CHECKLIST

**Thứ tự đề xuất (từ dễ/sạch → khó):**
1. **LLaVA-CoT (#12)** — khớp protocol nhất, data sẵn.
2. **MR-MKG (#14)** — VQA, cần scene graph GQA.
3. **COCO-Tree (#11)** — inference-only.
4. **Pix2Graph (#13)** — SGG→VQA adapt.
5. **LLM4SGG (#15)** — SGG→VQA adapt.

**Checklist mỗi paper:**
- [ ] Bước 1+2: tóm tắt kiến trúc đã verify (GPT hoặc Claude)
- [ ] Bảng module/chức năng/input/output
- [ ] Danh sách "paper nói rõ" / "paper không nêu rõ"
- [ ] Code Kaggle (Config/Load/Preprocess/Model/Train/Eval/Save)
- [ ] Chạy lấy metrics (Acc,Prec,Recall,F1,ROC-AUC,PR-AUC,FPR,FNR,Train Time,Params,Steps)
- [ ] Điền sheet "Implement" đúng dòng + cột Note theo mẫu
