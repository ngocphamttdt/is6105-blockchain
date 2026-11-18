# Thay đổi trong báo cáo midterm sau khi tune lại B1

File: `docs/[NỘP MIDTERM V2] Blockchain-Group01-C1-OnchainBitcoinForecast-Midterm-REVIEW2.docx`
Ngày sửa: 13/11/2026 · Branch: `vi/b1-lr-tuning` · Theo góp ý 3 của thầy (tuning chỉ dùng train/validation)

## Vì sao phải sửa

Bản tune B1 cũ dùng solver `saga`. Trong sklearn 1.9, `saga` dừng sau 1 vòng khi mọi hệ số bằng 0 dù intercept chưa hội tụ, nên kết quả phụ thuộc `random_state`: chỉ đổi seed mà accuracy dao động 51,0–53,5%. B1 nay được giải tất định bằng L-BFGS-B, cùng hàm mục tiêu với sklearn (`ElasticNetLR` trong `pipeline/src/b1t_tuning.py`). Kết quả khớp `lbfgs` của sklearn (sai khác p̂ ≤ 3·10⁻⁷), và chạy lại cho file giống hệt.

B1 được tune lại cho cả hai cấu hình E1 (v1, 33 đặc trưng On-chain) theo cùng một quy trình:

- Lưới gồm 300 tổ hợp: 5 tập đặc trưng × 10 mức C × l1_ratio {0; 0,5; 1} × class_weight {không; balanced}.
- Chọn theo log-loss trên validation.
- Ở cấu hình đầy đủ, mỗi tập ứng viên được thêm 33 đặc trưng On-chain.

Tái lập:

```bash
python pipeline/src/b1t_tuning.py --stage run      # so sánh trước/sau tuning (Bảng C.2)
python pipeline/src/b1t_tuning.py --stage tables
python pipeline/src/b1t_tuning.py --stage e1       # B1 tune cho E1 -> results/b1t/e1/
```

Các file dự báo chính thức `pipeline/results/predictions/b1_*.csv` **chưa bị ghi đè**. Các số mới của B1 trong báo cáo lấy từ `results/b1t/e1/{ohlcv,ohlcv_onchain}/predictions/`.

## Kết luận thay đổi

| | Trước | Sau |
|---|---|---|
| B1 DA, OHLCV → +On-chain | 51,11 → 51,89 | **53,20 → 51,19** |
| ΔDA B1 [CI 95%], p McNemar | +0,78 [−1,27; +2,92], 0,49 | **−2,01 [−4,23; +0,12], 0,075** |
| Mô hình tốt nhất (log-loss, cấu hình đầy đủ) | B2 (0,6957) | **B1 (0,6944)** |
| H1 | không chấp nhận | không chấp nhận (On-chain còn có xu hướng làm B1 kém đi) |
| H3 | không chấp nhận | không chấp nhận |

## Các bảng

| Bảng | Dòng | Trước → Sau |
|---|---|---|
| 5.1 | B1, Thiết lập | "Grid C" → "Grid tập đặc trưng, C, l1_ratio, class_weight (Phụ lục C)" |
| 7.1 | E1, Trạng thái | thêm "; B1 tune lại theo góp ý 3 (Phụ lục C)" |
| 8.1 | B1 | DA 51,11 → 51,89 ⇒ **53,20 → 51,19** · F1 0,509 → 0,517 ⇒ **0,532 → 0,512** · AUC 0,510 → 0,513 ⇒ **0,530 → 0,521** · Log-loss 0,6953 → 0,7006 ⇒ **0,6926 → 0,6944** · % ngày dự báo tăng 56,4 → 54,7 ⇒ **49,0 → 51,4** |
| 8.2 | B1 | ΔDA +0,78 [−1,27; +2,92] ⇒ **−2,01 [−4,23; +0,12]** · McNemar 327/346, 0,49 (1,00) ⇒ **389/340, 0,075 (0,38)** · Δlog-loss +0,0054 [+0,0012; +0,0102] ⇒ **+0,0019 [−0,0010; +0,0047]** · DM −2,23, 0,026 (0,13) ⇒ **−1,27, 0,20 (0,82)** |
| 8.2 | B3 | p DM (Holm) 0,098 (0,39) ⇒ **0,098 (0,49)** |
| 8.2 | B4 | p McNemar (Holm) 0,15 (0,73) ⇒ **0,15 (0,58)** |
| 8.3 | B1 | Lợi nhuận +19% → +245% ⇒ **+263% → +179%** · Sharpe 0,31 → 0,63 ⇒ **0,66 → 0,57** · MDD −86,6% → −65,4% ⇒ **−70,0% → −80,2%** · Số lần đổi vị thế 843 → 857 ⇒ **1.273 → 968** |
| B.1 | B1 Bull | ΔDA +0,08 [−2,48; +2,64] ⇒ **−2,88 [−5,68; −0,08]** · Δlog-loss +0,0063 [−0,0005; +0,0135] ⇒ **+0,0021 [−0,0017; +0,0059]** |
| B.1 | B1 Sideways | ΔDA +1,66 [−2,00; +5,10] ⇒ **−1,00 [−4,77; +2,66]** · Δlog-loss +0,0062 [−0,0020; +0,0141] ⇒ **+0,0005 [−0,0036; +0,0047]** |
| B.1 | B1 Bear | ΔDA +1,06 [−4,61; +6,74] ⇒ **−1,42 [−7,80; +4,61]** · Δlog-loss −0,0010 [−0,0104; +0,0084] ⇒ **+0,0052 [−0,0040; +0,0152]** |
| B.2 | B1 | Δ_Bull − Δ_Bear −0,98 ⇒ **−1,46** · khoảng 95% [−6,85; +6,97] ⇒ **[−7,39; +6,61]** · p 0,80 ⇒ **0,70** · p (Holm) 1,00 (không đổi) |
| C.1 | B1 | Grid "C ∈ {0,01; 0,1; 1; 10}" ⇒ lưới mới gồm tập đặc trưng, C (10 mức), l1_ratio, class_weight · Tham số cố định "max_iter 1000" ⇒ "winsorize 1%/99%; solver L-BFGS-B tất định (Phụ lục C)" |
| **C.2 (mới)** | — | B1 trước và sau tuning: trước tuning 51,48%, lưới C mở rộng 50,16%, tune theo log-loss 53,20%, tune theo accuracy 52,67%, luật đảo chiều 53,25% |

B3 và B4 trong Bảng 8.2 chỉ đổi cột Holm, vì Holm hiệu chỉnh gộp chung 5 mô hình.

## Caption

- **Bảng 8.1:** thêm "B1 tune đầy đủ trên train/validation cho từng cấu hình (Phụ lục C)".
- **Bảng B.1:** thêm "; B1 là bản tune ở Phụ lục C".
- **Bảng C.2:** ghi chú rằng hai dòng đầu dùng winsorize nên khác bản B1 chưa tune ở mục 8.2 (51,11%).

## Hình

Hai hình được vẽ lại theo style cũ. Chỉ B1 dùng số mới; B2–B5 giữ đúng số trong bảng.

- **Hình 8.1:** ΔDA và Δlog-loss theo mô hình.
- **Hình 8.3:** ΔDA theo regime.

## Đoạn văn

| Vị trí | Thay đổi |
|---|---|
| 8.2, bullet 1 | Mô hình tốt nhất B2 ⇒ **B1 sau tuning** (log-loss 0,6944); ΔDA −2,01, p = 0,075 (Holm 0,38); không chấp nhận H1 |
| 8.2, bullet 2 | Khoảng ΔDA "−1,48 đến +0,78" ⇒ "−2,01 đến +0,16"; thêm "thấp nhất là B1, p = 0,075" |
| 8.2, bullet 3 | "Không mô hình nào thắng Majority" ⇒ "chỉ B1 OHLCV-only (0,6926) thấp hơn Majority (0,6938)"; mọi CI của Δlog-loss chứa 0; DM của B1 p = 0,20 |
| 8.2, bullet 4 | Thêm mốc ΔDA −2,01 khi tune đầy đủ và lý do (B1 OHLCV-only chủ yếu dùng return hôm nay) |
| 8.4, bullet 1 | "Mọi CI ΔDA đều chứa 0" ⇒ "chỉ B1 ở Bull có CI không chứa 0", cùng với B4 ở Bear (log-loss) là hai kết quả đơn lẻ trong 30 phép so sánh |
| 8.4, bullet 2 | Placebo B1 −0,98 ⇒ −1,46 |
| 9, "Phạm vi thực nghiệm" | "B1, B2, B4 là bản chạy thử" ⇒ "B1 đã tune đầy đủ (Phụ lục C); B2, B4 là bản chạy thử" |
| 10, kết luận | Thêm: với B1 đã tune, On-chain có xu hướng làm kém đi (ΔDA −2,01 điểm, p = 0,075); B1 OHLCV-only chỉ ngang luật đảo chiều (~53%) |
| 10, hướng tiếp theo | "hoàn tất bản chính thức B1, B2, B4" ⇒ "B2, B4" |

## Phần mới: Phụ lục C, "Tuning đầy đủ cho B1 Logistic Regression (góp ý 3)"

Tiểu mục mới đặt sau Bảng C.1. Phần tuning không thành mục riêng trong mục 8, vì các baseline khác cũng sẽ được tune và có thể thêm vào cùng chỗ này (Bảng C.3, C.4, …). Tiểu mục gồm:

- một đoạn mô tả quy trình tuning;
- Bảng C.2;
- 6 nhận xét:
  1. Tuning có tác dụng trên validation: thắng dự báo hằng số 14/14 lần.
  2. Trên test, B1 tăng +1,73 điểm so với trước tuning, nhưng p = 0,14.
  3. Tham số chọn chủ yếu là return hôm nay, nên B1 sau tuning gần như là luật đảo chiều.
  4. Chọn theo accuracy không tốt hơn chọn theo log-loss; C = 10 chạm biên lưới 5/14 lần.
  5. Ổn định số: lỗi `saga` và cách sửa.
  6. Bảng 8.1–8.3, Hình 8.1 và 8.3, Phụ lục B dùng B1 tune theo quy trình này.

## Không sửa

- Mục 1–4 và mục 6; Phụ lục A.
- Phần chữ của mục 8.3.
- Số liệu của B0, B2, B4, B5, và các ô của B3 không nêu ở trên.

## Lưu ý cho nhóm

- **Phụ lục B:** script tạo số liệu B2–B5 không có trong repo. Dòng B1 được tính bằng `pipeline/src/experiments.py` (block bootstrap trong regime, khối 15 ngày, 2.000 mẫu). Ước lượng điểm khớp với số cũ, còn CI có thể lệch nhẹ vì là một lần rút bootstrap khác.
- **CI của B5 trong Bảng 8.2** (−2,38; +2,10) khác bản tính lại từ repo (−2,55; +2,18). B5 không liên quan đến thay đổi này nên được giữ nguyên.
