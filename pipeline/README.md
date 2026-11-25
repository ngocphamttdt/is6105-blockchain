# pipeline: walk-forward, đặc trưng On-chain, mô hình, chạy thí nghiệm

Code và kết quả cho task **1.15, 1.16, 2.8, 2.11, 3.8** (Ngọc). Đây là pipeline đo:

```
Δ = M(OHLCV + On-chain) − M(OHLCV-only)
```

Hai vế dùng cùng mô hình, cùng 82 fold walk-forward và cùng ngân sách tuning; chỉ khác bộ đặc trưng.

## Trạng thái task

| Task | Nội dung | Trạng thái | File |
|---|---|---|---|
| 1.15 | Walk-forward chống rò rỉ: train 730 ngày, gap 1 ngày, test 30 ngày, 82 fold (01/2020 – 08/2026) | ✅ Xong | `src/walkforward.py` |
| 1.16 | Biến đổi On-chain: 3 đặc trưng mỗi chỉ số; phương án Flows A có 11 chỉ số, 33 đặc trưng | ✅ Xong | `src/features.py` |
| 2.8 | Interface mô hình và pipeline chạy bằng một lệnh | ✅ Xong | `src/models.py`, `src/run_all.py` |
| 2.11, 3.8 | B0 (`b0pers`, `b0maj`) và B3 (`b3`, SVM RBF), đủ 82 fold × 2 cấu hình | ✅ Xong | `src/models.py`, `results/` |
| (thử) | B2 (`b2`, Random Forest) và B4 (`b4`, XGBoost), đủ 82 fold × 2 cấu hình. Bản chạy thử để có bảng Δ sớm; bản chính thức vẫn do Trâm (B2) và Khoa (B4) làm | ✅ Chạy thử | `src/models.py`, `configs/models/b2.yaml`, `b4.yaml` |
| 2.12, 3.10 | B5 (`b5`, LSTM của Ngân) tích hợp vào `run_all.py`: 82 fold × 2 cấu hình × 3 seed | ✅ Xong | `regime/model_lstm.py`, `configs/models/b5.yaml` |
| 2.13 | Ghép regime HMM của Ngân vào mọi file predictions | ✅ Xong | `regime/regime_map.csv`, `--stage regime` |
| 2.9, 2.10 | E0 sanity check: xáo nhãn, chia 80/20 | ⏳ Chưa có | (`--stage e0`) |

## Cấu trúc

```
pipeline/
├── README.md
├── src/                      mỗi file ứng với một bước; chạy file nào cũng test trước
│   ├── data.py               (0) đọc data freeze, kiểm sha256, tên cột, chọn cột theo dictionary A/B
│   ├── features.py           (1) nhãn y, đặc trưng OHLCV (lấy từ features/) và On-chain
│   ├── walkforward.py        (2) chia fold, scaler + tune trong train, chạy mô hình qua mọi fold
│   ├── models.py             (3) interface mô hình; B0, B3; bản thử B2, B4
│   └── run_all.py            (4) chạy thí nghiệm: --stage data | e1 | regime; B5 và ghép regime
├── configs/
│   ├── walkforward.yaml      tham số protocol
│   ├── features_onchain.yaml flows_set: A, shift_days: 1, ffill_limit: 3, signed_cols
│   └── models/{lr,b0maj,b2,b3,b4,b5}.yaml   tham số cố định và grid để tune (b5: seed và tham số LSTM)
└── results/
    ├── predictions/{model}_{config}.csv
    ├── predictions/b5_seeds/b5s{seed}_{config}.csv   B5 từng seed; b5_{config}.csv là p̂ trung bình
    ├── tables/e1_delta_draft.csv
    ├── tables/e1_by_regime_draft.csv                 DA, log-loss theo regime (sơ bộ, chưa có CI)
    ├── tables/e1_by_year.csv                         DA, log-loss từng mô hình × cấu hình × năm test
    ├── tables/e1_vs_reversal.csv                     mỗi mô hình so với luật đảo chiều b0rev (ΔDA gộp, McNemar, từng năm)
    └── figures/walkforward.png
```

## Dữ liệu đầu vào

Chỉ đọc `data/processed/` (data freeze v1.0 của Khoa và Vi), không đọc `data/raw/`. Không có code nào ghi vào thư mục này.

| File | Dùng cho |
|---|---|
| `master_daily.csv` | Giá ngày (`ohlcv_*`) cho nhãn và đặc trưng OHLCV; chỉ số On-chain (`cm_*`, `bc_*`, `puell_multiple`, `nvt_ratio_bc`) |
| `ohlcv_1h_clean.csv` | Giá 1h, chỉ để tính `rv_7d`, `rv_30d` |
| `data_dictionary_A.csv` | Danh sách chỉ số On-chain đưa vào mô hình: `status = retained`, `role` là `candidate` hoặc `backup` |
| `MANIFEST.json` | sha256 của mọi file. Mỗi lần đọc đều so; lệch thì dừng |

Ngoài ra đọc `regime/regime_map.csv` (date, fold, regime) để điền cột `regime`.

## Chạy

Cần Python 3.10+ với `pandas numpy scikit-learn pyyaml matplotlib`, và thư mục `features/src/` (đặc trưng OHLCV). `b4` cần thêm `xgboost` (trên macOS phải `brew install libomp` trước); `b5` cần `torch`. Chạy từ thư mục gốc repo:

```bash
python pipeline/src/run_all.py --test                                   # chỉ chạy test (~10 giây)
python pipeline/src/run_all.py --stage data                             # kiểm freeze, đặc trưng, fold
python pipeline/src/run_all.py --stage e1 --models b1 b0pers b0maj b2 b3 b4 b5   # E1 đủ (~10 phút)
python pipeline/src/run_all.py --stage e1 --models b3 --every 10        # chạy thử 1/10 số fold -> every10/
python pipeline/src/run_all.py --stage regime                           # ghép lại regime khi regime_map.csv đổi
python pipeline/src/run_all.py --stage tables                           # tính lại mọi bảng từ predictions đã có
python pipeline/src/walkforward.py                                      # vẽ lại results/figures/walkforward.png
```

**Ghi vào thư mục chỉ định và bỏ nhóm On-chain.** `--out DIR` (tính từ gốc repo) ghi `DIR/predictions/`, `DIR/tables/` thay cho `results/`. Mỗi nhóm On-chain có một cờ riêng để bỏ nhóm đó khỏi cấu hình `ohlcv_onchain` cho mọi mô hình; các cờ dùng riêng hoặc kết hợp được (không được bỏ cả 4 nhóm):

| Cờ | Nhóm | Đặc trưng bị bỏ |
|---|---|---|
| `--no-flows` | Flows | Flows A: 6 (`cm_SplyExNtv`, `cm_NetflowExNtv`) |
| `--no-activity` | Activity | 9 (`cm_AdrActCnt`, `cm_TxCnt`, `bc_est_tx_vol_usd`) |
| `--no-mining` | Mining | 9 (`cm_HashRate`, `cm_FeeTotNtv`, `bc_difficulty`) |
| `--no-valuation` | Valuation | 9 (`cm_CapMVRVCur`, `puell_multiple`, `nvt_ratio_bc`) |

Vd. `--no-flows --no-activity` còn 33 − 6 − 9 = 18 On-chain. Bộ `ohlcv` giữ nguyên. Các cờ này bắt buộc đi kèm `--out` để không đè kết quả E1 chính. Mỗi thư mục có `feature_sets.json` ghi bộ đặc trưng đã dùng (`drop_groups`); chạy tiếp vào cùng thư mục với bộ khác (vd. lần bỏ Flows, lần bỏ Flows + Activity) sẽ báo lỗi.

```bash
python pipeline/src/run_all.py --stage e1 --models b1 b2 b3 b4 b5 --no-flows --out pipeline/results/no_flows
python pipeline/src/run_all.py --stage tables --out pipeline/results/no_flows
# Như rerun B0–B3 (10 đặc trưng OHLCV), gồm cả b1t:
python pipeline/src/rerun_e1.py --out pipeline/results/no_flows_rerun --no-flows \
    --models b0maj b0pers b0rev b1 b1fs b1t b2 b2t b3 b3t b4 b4t b5
python pipeline/src/rerun_tables.py --dir pipeline/results/no_flows_rerun --old pipeline/results/rerun_2026-11-19
# Bỏ Flows và Activity (tương tự với --no-mining, --no-valuation):
python pipeline/src/rerun_e1.py --out pipeline/results/no_flows_activity --no-flows --no-activity \
    --models b0maj b0pers b0rev b1 b1fs b1t b2 b2t b3 b3t b4 b4t b5
python pipeline/src/rerun_tables.py --dir pipeline/results/no_flows_activity --old pipeline/results/rerun_2026-11-19
```

Các thí nghiệm theo góp ý 3–6 của thầy nằm trong `src/experiments.py` (đánh giá đầy đủ: `docs/feedback_review/report.md`):

```
python pipeline/src/experiments.py --stage tuning [--models b1 b3]      # góp ý 3: tham số chọn mỗi lần tune, trước/sau tuning
python pipeline/src/experiments.py --stage tuning-tables                # -> results/tuning/tuning_{params,before_after}.csv
python pipeline/src/experiments.py --stage e2 [--models b2] [--configs no_flows]   # góp ý 4, 5: ablation trên bộ đặc trưng v2
python pipeline/src/experiments.py --stage e2-tables                    # -> results/e2/tables/
python pipeline/src/experiments.py --stage e3                           # góp ý 6: Δ theo regime HMM và theo xu hướng -> results/e3/
python pipeline/src/gate.py --stage gate                                # góp ý 4a: cổng baseline OHLCV, lưới mở rộng -> results/gate/baseline/
python pipeline/src/gate.py --stage stepwise                            # góp ý 4b: thêm nhóm On-chain tiến dần theo inner CV -> results/gate/stepwise/
python pipeline/src/gate.py --stage tables                              # -> results/gate/tables/, figures/fig_gate_by_year.png
python pipeline/src/rules30.py                                          # luật 30 ngày (mom30, ma30, maj30) + accuracy/F1 cả chuỗi -> results/rules30/
python pipeline/src/hybrid.py --stage a                                 # luật làm đặc trưng (dấu return hôm nay) cho B1–B4
python pipeline/src/hybrid.py --stage tables                            # + luật mặc định/mô hình ghi đè, luật On-chain cực trị -> results/hybrid/
```

- `src/gate.py`: tiêu chí cổng chốt trong code trước khi chạy. G1: log-loss inner CV thấp hơn B0 (tỷ lệ ngày tăng) ở ≥ 10/14 lần tune. G2: không tham số nào chạm biên lưới quá 7/14 lần. Kết quả: không mô hình nào qua cổng; stepwise không chọn nhóm On-chain nào ở 9–13/14 lần tune.

- Bộ đặc trưng v2 (`configs/features_onchain_v2.yaml`) sửa v1 theo `eda_pre_training`: sai phân 7 ngày cho cột có mùa vụ tuần, bỏ MVRV, NVT, SplyExNtv, difficulty, Puell logdiff (20 đặc trưng On-chain). v1 giữ nguyên cho E1.
- E2 thêm winsorize 1%/99% (`walkforward.QuantileClipper`, học trên train của fold) cho mọi cấu hình, kể cả chỉ OHLCV.
- `src/stats.py`: block bootstrap, McNemar, Diebold–Mariano, Holm, xáo đợt regime cho placebo.

- Mỗi lần chạy stage đều chạy test của mọi file trước.
- Có thể chạy riêng một file, ví dụ `python pipeline/src/features.py --test`.
- Thời gian chạy E1 đủ: `b2` khoảng 7 phút, `b5` khoảng 1 phút (3 seed chạy song song), các mô hình khác dưới 1 phút.

`--stage data` phải in:

```
Data freeze v1.0: sha256 khớp 4 file; flows set A; 10 OHLCV + 33 On-chain; 82 fold, train fold 0 từ 2017-12-31; NaN từ ngày đó: 0
```

Mỗi người có thể chạy mô hình của mình riêng. Bảng Δ luôn tính lại từ mọi file trong `results/predictions/`, nên không mất dòng của mô hình chạy trước. Mỗi file predictions được điền `regime` ngay khi ghi; ngày nào không có trong `regime_map.csv` thì báo lỗi, không tự gán.

**Tái lập:** chạy lại cho `p_hat` giống hệt với `b1`, `b0*`, `b3`, `b4`, `b5`. `b2` lệch tối đa khoảng 1e-15, do Random Forest cộng xác suất của các cây song song (`n_jobs: -1`); metric không đổi.

### Hai lưu ý kỹ thuật

- **torch và xgboost không chạy chung tiến trình.** Trên macOS, hai thư viện mang hai bản OpenMP khác nhau; nếu xgboost đã chạy trước, torch bị treo khi tạo tensor. Vì vậy `run_all.py` chạy B5 trong tiến trình con (`spawn`), mỗi seed một tiến trình, và không import torch ở tiến trình chính.
- **Tuning hoà điểm.** Grid của B2 có những tổ hợp cho cùng mô hình (vd. `max_depth: null` và `6` khi `min_samples_leaf: 50`), điểm chỉ lệch ~1e-16 do cộng song song. `tune()` làm tròn điểm đến 10 chữ số và lấy tổ hợp đứng trước khi hoà, để lần chạy lại chọn đúng tổ hợp cũ.

## Quy tắc chống rò rỉ, và test nào kiểm

| Rủi ro | Cách chặn | Test |
|---|---|---|
| Đặc trưng dùng dữ liệu tương lai | Chỉ dùng `diff` và `rolling` lùi. `ffill` chỉ lấy quá khứ, tối đa 3 ngày | `features.py` (2), (6): sửa dữ liệu sau mốc cut thì đặc trưng đến cut không đổi |
| On-chain công bố trễ | `shift_days: 1`: dòng t mang On-chain của ngày t−1. OHLCV vẫn dùng đến hết ngày t | `features.py` (3) |
| Nhãn ngày train cuối dùng giá đóng cửa của ngày test đầu | Gap 1 ngày giữa train và test, cả trong inner CV. Với dự báo 1 ngày, gap là biên an toàn thêm chứ không bắt buộc: dự báo cho ngày test đầu t₀ được đưa ra lúc cuối ngày t₀, khi giá đóng cửa ngày t₀ (và do đó nhãn của ngày t₀−1) đã biết. Gap chỉ bỏ 1 ngày train mỗi fold, không làm sai kết quả | `walkforward.py` (1), (4) |
| Scaler hoặc tuning thấy test | StandardScaler nằm trong Pipeline; GridSearchCV nhận Pipeline | `walkforward.py` (3): nhân dữ liệu test lên 1.000 lần, dự báo vẫn khớp pipeline chỉ fit trên train |
| Persistence nhìn ngày mai | p̂_t = 1[close_t > close_{t−1}] | `models.py` (3) |
| Dùng nhầm phiên bản dữ liệu | So sha256 với `MANIFEST.json` | `data.py` (1), (3) |
| Ghép nhầm regime | Ghép theo `date` + `fold`, `validate="one_to_one"`; thiếu regime thì báo lỗi | `run_all.py` (4) |

Đã thử cố tình cài 4 lỗi vào một bản sao, và `--test` bắt được cả 4:
- z-score dùng `rolling(center=True)`, tức nhìn tương lai;
- bỏ gap giữa train và test;
- persistence dùng nhãn của chính ngày t;
- scaler fit trên cả tập test.

## Đặc trưng

| Nhóm | Đặc trưng |
|---|---|
| OHLCV (10 cột, `build_ohlcv_features` trong `features/src/features_ohlcv.py`, task 1.13) | `logret_lag1` … `logret_lag5` (`logret_lag1` = log return của chính ngày t), `hl_range` = (H−L)/C, `dlog_volume`, `rsi_14`, `rv_7d`, `rv_30d` (từ return 1h). Công thức chi tiết: README của `features/` |
| On-chain, cột dương | `{col}_logdiff` = log x_t − log x_{t−1}, `{col}_ma7_30` = MA7 / MA30, `{col}_z90` = z-score 90 ngày của log x |
| On-chain, cột có dấu (`cm_NetflowExNtv`) | `{col}_diff` = x_t − x_{t−1}, `{col}_ma7m30` = MA7 − MA30, `{col}_z90` = z-score 90 ngày của x. Netflow âm ở khoảng một nửa số ngày nên không lấy log được |

Số đặc trưng On-chain theo nhóm (phương án A, dùng cho ablation E2 qua `feature_groups`): Activity 9, Flows 6, Mining 9, Valuation 9.

## Mô hình

| Tên | Mô hình | Ghi chú |
|---|---|---|
| `b1` | B1 Logistic Regression, grid C | Bản cho pipeline mỏng; Trâm sở hữu B1 |
| `b0maj` | B0 majority class: p̂ = tỷ lệ ngày tăng trong train | Mức sàn cho log-loss và Brier |
| `b0pers` | B0 persistence: ngày mai cùng hướng hôm nay | Mức sàn cho DA. p̂ chỉ là 0/1, nên log-loss và Brier để trống trong bảng |
| `b0rev` | B0 đảo chiều: ngày mai ngược hướng hôm nay; chiều chọn lại trên từng cửa sổ train | Đề xuất 1 của `baseline_check`. Mốc DA mạnh nhất (53,25%); mọi mô hình có học phải so với luật này (`e1_vs_reversal.csv`). p̂ chỉ là 0/1 như `b0pers` |
| `b1fs` | B1 + chọn đặc trưng tiến dần: bắt đầu từ `logret_lag1`, thêm cột làm log-loss inner CV giảm; grid C | Đề xuất 2 của `baseline_check`. Thường giữ 1–3 đặc trưng OHLCV |
| `b3` | B3 SVM RBF, grid C × gamma | Xác suất hiệu chỉnh sigmoid trên điểm ngoài mẫu theo thời gian (TimeSeriesSplit, gap 1); SVM cuối fit trên toàn cửa sổ train |
| `b2` | B2 Random Forest, grid số cây × độ sâu × `min_samples_leaf` (12 tổ hợp) | Bản chạy thử; Trâm sở hữu B2 |
| `b4` | B4 XGBoost, grid learning rate × độ sâu | Bản chạy thử; Khoa sở hữu B4. Số cây chọn bằng early stopping trên 20% cuối cửa sổ train (bỏ 1 ngày gap), rồi fit lại trên toàn cửa sổ |
| `b5` | B5 LSTM: 1 lớp, hidden 32, cửa sổ 30 ngày, BCE, Adam | Code của Ngân (`model_lstm.py`), không qua GridSearchCV. Early stopping trên 20% cuối cửa sổ train; seed 42, 43, 44; `b5_{config}.csv` là p̂ trung bình 3 seed |

B0 không dùng đặc trưng, nên hai cấu hình cho cùng dự báo (Δ = 0).

Thêm mô hình mới:
1. Gọi `register("rf", RandomForestClassifier)` trong `src/models.py`.
2. Tạo `configs/models/rf.yaml` gồm `class`, `params`, `param_grid`.
3. Chạy `--stage e1 --models rf`.

## Kết quả E1 nháp (chạy lại 30/10/2026; B0–B3 kiểm lại 19/11/2026)

Phương án A, 10 đặc trưng OHLCV của Trâm, 2.434 ngày test. DA và AUC: cao hơn là tốt hơn. Log-loss và Brier: thấp hơn là tốt hơn.

Các dòng "tune lại" (`b1t`, `b2t`, `b3t`; hậu tố `t`, xem [`configs/models/README.md`](configs/models/README.md)) lấy từ dự báo đã lưu trong `results/b1t/e1/` và `results/predictions/`, cùng 82 fold và cùng 10 đặc trưng OHLCV. `b4t` chưa chạy qua pipeline nên chưa có dòng.

| Mô hình | DA: OHLCV → +On-chain | ΔDA | Log-loss: OHLCV → +On-chain | Δlog-loss |
|---|---|---|---|---|
| `b0maj` | 0,5008 | 0 | 0,6938 | 0 |
| `b0pers` | 0,4651 | 0 | không dùng | 0 |
| `b0rev` | 0,5325 | 0 | không dùng | 0 |
| `b1fs` | 0,5111 → 0,5156 | +0,0045 | 0,6933 → 0,6968 | +0,0036 |
| `b1` | 0,5111 → 0,5189 | +0,0078 | 0,6953 → 0,7006 | +0,0054 |
| `b1t` (tune lại) | 0,5320 → 0,5119 | −0,0201 | 0,6926 → 0,6944 | +0,0019 |
| `b2` (thử) | 0,5123 → 0,5140 | +0,0016 | 0,6961 → 0,6957 | −0,0005 |
| `b2t` (tune lại) | 0,5115 → 0,4922 | −0,0193 | 0,6947 → 0,6951 | +0,0004 |
| `b3` | 0,5062 → 0,5029 | −0,0033 | 0,7009 → 0,6959 | −0,0050 |
| `b3t` (tune lại) | 0,5066 → 0,4901 | −0,0164 | 0,6959 → 0,6966 | +0,0007 |
| `b4` (thử) | 0,5045 → 0,4897 | −0,0148 | 0,6973 → 0,6958 | −0,0015 |
| `b5` (3 seed) | 0,5074 → 0,5058 | −0,0016 | 0,6950 → 0,6959 | +0,0009 |

- Mọi mô hình có học chỉ đoán đúng khoảng 49–53% số ngày. Cao nhất là `b1t` chỉ OHLCV (0,5320), gần bằng `b0rev` vì nó gần như tự học lại luật đảo chiều (xem mục "Kiểm lại B0–B3").
- Ở cấu hình OHLCV + On-chain, không mô hình có học nào có log-loss thấp hơn mức sàn `b0maj` (0,6938). Ở cấu hình chỉ OHLCV, chỉ `b1t` (0,6926) và `b1fs` (0,6933) thấp hơn, và chỉ rất ít. Xác suất của mô hình gần như chưa tốt hơn việc luôn đoán tỷ lệ ngày tăng của tập train.
- Với 2.434 ngày, sai số chuẩn của DA khoảng 1 điểm % (ước lượng thô), nên các Δ ở trên nằm trong mức nhiễu. Riêng `b4` có ΔDA −1,5 điểm, hơi vượt mức đó, nhưng Δlog-loss của nó lại âm (On-chain tốt hơn chút), nên chưa đủ để nói On-chain làm `b4` tệ đi. Kiểm định cho B1–B3: xem mục "Kiểm lại B0–B3" bên dưới.
- Δ đổi dấu giữa các mô hình: ΔDA dương ở `b1`, `b1fs`, `b2`, âm ở `b3`, `b4`, `b5`. Ba bản tune lại (`b1t`, `b2t`, `b3t`) đều có ΔDA âm, từ −1,6 đến −2,0 điểm: tune kỹ hơn không làm On-chain có ích hơn.
- Theo quy tắc ghi sẵn (mô hình tốt nhất là mô hình có log-loss thấp nhất ở cấu hình OHLCV + On-chain), trong các bản grid gốc, mô hình tốt nhất là `b2` (0,6957), sát với `b4` (0,6958), `b3` và `b5` (0,6959). ΔDA của `b2` chỉ +0,16 điểm. Nếu tính cả bản tune lại, thấp nhất là `b1t` (0,6944) rồi `b2t` (0,6951), nhưng cả hai đều có DA giảm khi thêm On-chain.
- Δ rất nhạy với bộ OHLCV: với 7 đặc trưng tạm (bản 28/10), ΔDA của `b1` là −0,0045; đổi sang 10 đặc trưng thì thành +0,0078.
- `b0pers` dưới 50%: trong giai đoạn 2020–2026, hướng giá ngày mai hay ngược với hôm nay.
- `b3` cấu hình OHLCV-only có log-loss tệ hơn mức sàn `b0maj`. Có những ngày mô hình quá tự tin (p̂ tới 0,987); cần xem lại theo từng fold.
- Tỷ lệ ngày tăng đổi mạnh theo năm, từ khoảng 46% (2022) đến 58% (2020) ([`eda_05_label_balance.png`](../features/reports/figures/eda_05_label_balance.png)). Vì vậy DA của từng fold nên so với tỷ lệ tăng của chính giai đoạn đó, không so với 0,5.

### Kiểm lại B0–B3 (19/11/2026)

Chạy lại toàn bộ E1 cho 10 mô hình B0–B3 (`b0maj`, `b0pers`, `b0rev`, `b1`, `b1fs`, `b1t`, `b2`, `b2t`, `b3`, `b3t`) bằng đúng code hiện có, 10 đặc trưng OHLCV + 33 On-chain v1, rồi kiểm định Δ bằng `stats.py`. Kết quả nằm riêng trong [`results/rerun_2026-11-19/`](results/rerun_2026-11-19/report.md), không đè `results/predictions/`. Chạy lại bằng `src/rerun_e1.py`, `src/rerun_tables.py`, `src/rerun_figures.py`.

- **Tái lập được.** 16/20 file dự báo khớp bản cũ (lệch p̂ < 1e-7). `b2`, `b3` lệch nhỏ (p̂ tối đa 0,0017 và 0,0029), nhiều khả năng do khác phiên bản thư viện; chỉ 1 ngày đổi quyết định (`b2` +On-chain: DA 0,5140 → 0,5144). Bảng trên giữ số của lần chạy 30/10.
- Bảng dưới là số của lần chạy 19/11. CI 95% block bootstrap (khối 15 ngày); p Holm hiệu chỉnh trên 7 mô hình (Δ On-chain) và trên 14 phép so (so với `b0rev`, không ghi trong bảng). ΔDA tính bằng điểm %.

| Mô hình | ΔDA On-chain (điểm %) [CI 95%] | p Holm | Δlog-loss [CI 95%] | p Holm | ΔDA so với `b0rev`: OHLCV / +On-chain |
|---|---|---|---|---|---|
| `b1` | +0,78 [−1,27; +2,92] | 1,000 | +0,0054 [+0,0012; +0,0102] | 0,179 | −2,14 / −1,36 |
| `b1fs` | +0,45 [−1,73; +2,63] | 1,000 | +0,0036 [−0,0004; +0,0078] | 0,584 | −2,14 / −1,68 |
| `b1t` | −2,01 [−4,23; +0,12] | 0,452 | +0,0019 [−0,0010; +0,0047] | 0,819 | −0,04 / −2,05 |
| `b2` | +0,21 [−1,77; +2,05] | 1,000 | −0,0005 [−0,0034; +0,0025] | 1,000 | −2,01 / −1,81 |
| `b2t` | −1,93 [−3,70; −0,16] | 0,406 | +0,0004 [−0,0024; +0,0032] | 1,000 | −2,10 / −4,03 |
| `b3` | −0,33 [−2,05; +1,40] | 1,000 | −0,0050 [−0,0116; +0,0010] | 0,584 | −2,63 / −2,96 |
| `b3t` | −1,64 [−3,41; +0,12] | 0,475 | +0,0007 [−0,0034; +0,0042] | 1,000 | −2,59 / −4,23 |

- **Không Δ On-chain nào có ý nghĩa sau hiệu chỉnh Holm.** Hai Δ có CI không chứa 0 trước hiệu chỉnh đều theo chiều On-chain làm tệ đi: ΔDA của `b2t` và Δlog-loss của `b1`.
- **Chưa mô hình nào vượt luật đảo chiều.** `b1t` chỉ OHLCV đạt 0,5320, gần bằng `b0rev` (0,5325), vì nó gần như tự học lại luật: ở 14 lần tune, nó chọn tập `lag1`, `sign` hoặc `lag1+sign` 12 lần, và chỉ bất đồng với `b0rev` 99/2.434 ngày.
- **Thêm On-chain làm `b2t` và `b3t` thua luật rõ rệt:** ΔDA so với `b0rev` là −4,03 và −4,23 điểm, vẫn có ý nghĩa sau Holm (p 0,024 và 0,040).
- **Luật đảo chiều không ổn định theo thời gian:** DA từ 0,510 đến 0,585 trong 2020–2025, nhưng chỉ 0,438 năm 2026 (242 ngày).

### Backtest

Bằng `features/src/backtest.py` (long nếu p̂ > 0,5, phí 0,1%/lượt, 2020-01-02 → 2026-08-31). Bảng đầy đủ: `features/reports/backtest_e1_draft.md`; equity curve là task 3.7 của Trâm.

| Mô hình | Lợi nhuận tích luỹ: OHLCV → +On-chain | Sharpe | MDD | % ngày long |
|---|---|---|---|---|
| buy-and-hold | +990% | 0,90 | −77% | 100% |
| `b0maj` | +72% | 0,43 | −82% | 63% |
| `b0pers` | −71% | −0,23 | −79% | 51% |
| `b1` | +19% → +245% | 0,31 → 0,63 | −87% → −65% | 56% → 55% |
| `b2` (thử) | +241% → +14% | 0,62 → 0,29 | −80% → −76% | 55% → 54% |
| `b3` | +286% → +63% | 0,65 → 0,41 | −72% → −80% | 73% → 65% |
| `b4` (thử) | −43% → −65% | 0,11 → −0,05 | −86% → −93% | 63% → 58% |
| `b5` (3 seed) | +99% → +192% | 0,46 → 0,61 | −85% → −56% | 58% → 33% |

- Không mô hình nào thắng buy-and-hold. `b4` còn lỗ ở cả hai cấu hình.
- Xếp hạng theo lợi nhuận khác xếp hạng theo DA, vì lời lỗ phụ thuộc vào việc đoán đúng những ngày giá biến động mạnh. Ví dụ `b2` có DA gần như không đổi giữa hai cấu hình (0,512 → 0,514) nhưng lợi nhuận giảm từ +241% còn +14%.
- `b5` + On-chain chỉ giữ BTC 33% số ngày mà vẫn lời +192% với MDD −56%, thấp nhất trong các mô hình. DA của nó không đổi (0,507 → 0,506), nên đây có thể là may mắn về thời điểm hơn là tín hiệu.
- `b3` OHLCV-only ở trạng thái long 73% số ngày, trong khi chỉ 51% số ngày test là ngày tăng: mô hình lệch hẳn về phía "tăng".
- `b0pers` cho −71% ở cả `backtest.py` của Trâm và kết quả ở đây, dù hai bên viết code độc lập. Code của Trâm còn kiểm `y_true` = 1[close_{t+1} > close_t] trước khi chạy. Hai điều này xác nhận quy ước ngày của dự báo là đúng.

### Theo regime (sơ bộ cho E3, chưa có khoảng tin cậy)

Regime từ HMM của Ngân (`regime_map.csv`): Bull 1.250 ngày, Sideways 902 ngày, Bear 282 ngày. Bảng đầy đủ (DA, log-loss từng regime): `results/tables/e1_by_regime_draft.csv`.

| Mô hình | ΔDA Bull | ΔDA Sideways | ΔDA Bear |
|---|---|---|---|
| `b1` | +0,0008 | +0,0166 | +0,0106 |
| `b2` (thử) | −0,0120 | +0,0166 | +0,0142 |
| `b3` | −0,0064 | −0,0089 | +0,0284 |
| `b4` (thử) | −0,0112 | −0,0288 | +0,0142 |
| `b5` | −0,0208 | +0,0155 | +0,0284 |

- ΔDA ở Bear dương với cả 5 mô hình có học (+1,1 đến +2,8 điểm), còn ở Bull âm với 4/5 mô hình. Chiều này **ngược với H3** (dự kiến Δ ở Bull lớn hơn ở Bear).
- Nhưng Bear chỉ có 282 ngày, sai số chuẩn của DA khoảng 3 điểm, nên mọi ΔDA ở Bear đều nằm trong mức nhiễu. Cần CI bootstrap và placebo (xáo nhãn regime) của E3.
- Tỷ lệ ngày tăng gần như bằng nhau giữa ba regime (Bull 51,6%, Sideways 49,7%, Bear 51,8%). Regime HMM gắn với mức return và biến động của ngày t, chưa chắc phân biệt được hướng giá ngày t+1.

**Chưa dùng các con số này để kết luận RQ1, RQ3:** chưa có kiểm định (Vi, 2.3), `b2` và `b4` mới là bản chạy thử.

## Ablation + SHAP (task 5b, 25/11/2026)

Phần phụ, làm sau phép so sánh chính không Flows (`results/no_flows/`, mốc "full"). Thiết kế: [`docs/superpowers/specs/2026-11-24-ablation-shap-design.md`](../docs/superpowers/specs/2026-11-24-ablation-shap-design.md). Không sửa code pipeline; dùng lại `rerun_e1.py`, `rerun_tables.py` với cờ `--no-flows --no-<nhóm>`.

```bash
# 6 thư mục ablation, mỗi thư mục 13 model (phải ghi --models: mặc định của rerun_e1 thiếu b4, b4t, b5)
M="b0maj b0pers b0rev b1 b1fs b1t b2 b2t b3 b3t b4 b4t b5"
for x in "plus_activity --no-mining --no-valuation" "plus_mining --no-activity --no-valuation" \
         "plus_valuation --no-activity --no-mining" "drop_activity --no-activity" \
         "drop_mining --no-mining" "drop_valuation --no-valuation"; do
  set -- $x; d=pipeline/results/ablation/$1; shift
  python pipeline/src/rerun_e1.py --out $d --no-flows "$@" --models $M
  python pipeline/src/rerun_tables.py --dir $d
done
python pipeline/src/ablation_tables.py      # -> results/ablation/ablation_{summary,delta,reproduce,rank}.csv
python pipeline/src/shap_analysis.py        # LR, RF, XGBoost; 82 fold -> results/shap/ (cần cài thư viện shap)
```

- **Bảng gọn để đưa vào báo cáo:** [`results/ablation/ablation_summary.csv`](results/ablation/ablation_summary.csv). Mỗi dòng một model; mỗi cấu hình có 2 cột Δ log-loss (×1000) và Δ DA (điểm %); `*` = p sau Holm < 0,05; `†` = cột full của b4/b4t chạy trên máy khác. Số chi tiết (khoảng tin cậy, p-value) ở `ablation_delta.csv`.
- `plus_<g>` = OHLCV + nhóm g; `drop_<g>` = 10 OHLCV + 27 on-chain bỏ nhóm g. Mọi Δ = (có on-chain) − (OHLCV), cùng ngày, fold, tuning.
- Xếp hạng chính: Δ log-loss của `plus_*` (nhỏ = tốt). Phụ: log-loss(drop_g) − log-loss(full), chỉ cho model tái lập được mốc full.
- **Tái lập mốc của Khoa (max |Δp̂| ≤ 1e-3):** b1–b3t, b5 khớp (b5: 9,4e-6). XGBoost b4/b4t lệch 0,17–0,22 trên máy khác → không có `rank_drop`, bỏ khỏi SHAP.
- **b5 trên macOS:** tiến trình con của LSTM bị segfault (`BrokenProcessPool`) khi sklearn/xgboost nạp OpenMP trước torch. Cách chạy không cần sửa code: tạo một thư mục riêng chứa `sitecustomize.py` có dòng `import torch` (nếu môi trường đã có `sitecustomize.py` riêng, như Anaconda, thì `exec` file đó ngay sau dòng này), rồi chạy `PYTHONPATH=<thư mục đó> python pipeline/src/rerun_e1.py ... --models b5` và sau đó `rerun_tables.py`.

**Δ log-loss (×1000) khi thêm một nhóm vào OHLCV** (âm = tốt hơn; trong bảng này không ô nào có p_DM sau Holm < 0,05, nhỏ nhất 0,057):

| Model | Activity | Mining | Valuation | Full (27) |
|---|---|---|---|---|
| b1 | 3,55 | 2,23 | 0,49 | 5,15 |
| b1fs | 1,92 | 1,01 | 1,50 | 3,32 |
| b1t | 1,14 | 2,02 | 1,34 | 2,27 |
| b2 | −0,45 | −1,46 | 0,38 | −0,91 |
| b2t | 0,38 | −0,37 | 0,25 | −0,34 |
| b3 | 15,16 | −3,70 | 14,12 | −6,60 |
| b3t | 17,55 | 1,06 | 19,36 | −0,02 |
| b4 | 0,87 | −2,02 | −0,90 | 2,54 |
| b4t | 1,19 | −0,62 | 0,33 | 2,22 |
| b5 | 0,91 | 1,94 | 2,73 | 0,20 |

Cột Full (27) lấy từ `results/no_flows` (máy của Khoa). Với b4/b4t, XGBoost không tái lập được trên máy khác, nên cột này không so trực tiếp được với các cột plus_* của hai model đó.

- Chỉ 7/30 ô `plus_*` có log-loss tốt hơn OHLCV. Mining đứng đầu `rank_plus` ở 7/10 model, nhưng ở b1fs và b3t đó chỉ là nhóm ít làm kém nhất; 5 model cải thiện log-loss khi thêm Mining, lớn nhất −0,0037 (b3), không ô nào có ý nghĩa sau Holm.
- Δ DA của `plus_*` nằm trong [−2,5; +0,9] điểm %.

**Δ log-loss (×1000) khi bỏ một nhóm khỏi 27 on-chain** (so với OHLCV, cùng quy ước):

| Model | Bỏ Activity | Bỏ Mining | Bỏ Valuation |
|---|---|---|---|
| b1 | 2,21 | 3,53 | **5,62** |
| b1fs | 1,72 | 2,36 | 3,02 |
| b1t | 1,57 | 2,34 | 2,14 |
| b2 | −0,36 | −0,52 | −1,45 |
| b2t | 0,06 | −0,25 | −0,62 |
| b3 | −4,65 | −4,61 | −6,07 |
| b3t | 0,11 | −0,95 | 1,09 |
| b4 | 0,17 | −0,43 | −1,76 |
| b4t | 1,67 | 0,17 | 0,40 |
| b5 | 1,96 | 0,54 | 0,32 |

- Trong 30 ô `drop_*`, chỉ **drop_valuation / b1** có p_DM sau Holm < 0,05 (0,014), theo chiều on-chain (Activity + Mining) làm log-loss **kém đi**. Không cấu hình nào cho thấy on-chain cải thiện có ý nghĩa.

**SHAP** (`results/shap/shap_groups.csv`; tổng mean|SHAP| theo nhóm, đơn vị log-odds cho LR, xác suất cho RF):

| Model | OHLCV (10; b1t thêm `up_lag1`) | Activity (9) | Mining (9) | Valuation (9) | Tỷ trọng on-chain |
|---|---|---|---|---|---|
| b1 | 0,257 | 0,191 | 0,204 | 0,224 | 0,71 |
| b1fs | 0,126 | 0,064 | 0,052 | 0,063 | 0,59 |
| b1t | 0,090 | 0,054 | 0,054 | 0,065 | 0,66 |
| b2 | 0,039 | 0,026 | 0,025 | 0,026 | 0,66 |
| b2t | 0,029 | 0,018 | 0,014 | 0,017 | 0,63 |

- Model dùng on-chain nhiều (59–71% tổng |SHAP|, một phần do on-chain có 27 cột so với 10 cột OHLCV), nhưng điều đó không làm dự báo ngoài mẫu tốt hơn (bảng Δ ở trên).
- Thứ hạng nhóm theo SHAP chỉ khớp ablation ở b1 (`shap_vs_ablation.csv`); 4 model còn lại không khớp.

## Điểm cần nhóm chốt

1. **Phương án Flows (họp 1.19):** đang dùng A (`cm_NetflowExNtv` + `cm_SplyExNtv`). Nếu chốt B, đổi `flows_set: B` trong `configs/features_onchain.yaml` rồi chạy lại các cấu hình OHLCV + On-chain.
2. **Cửa sổ train 1.095 ngày (task 4.5):** fold đầu cần train từ 31/12/2016, nhưng OHLCV chỉ có từ 17/08/2017. Đề xuất: với cửa sổ này, bắt đầu test từ khoảng 09/2020, rồi so độ nhạy trên phần ngày test chung của cả ba cửa sổ.
3. **Fold cuối chỉ có 4 ngày test** (27–30/08/2026). Ghi chú trong mục Method.
4. **Hai điểm Trâm nêu, ảnh hưởng trực tiếp tới đặc trưng của pipeline** (chi tiết ở mục "Điểm cần nhóm chốt" trong README của `features/` và mục "Câu hỏi thường gặp" trong `data/processed/README.md`):
   - Ngày thiếu nến 2018-02-08 và 2018-02-09 (`dlog_volume` có z = −7,9 và +5,7) nằm trong cửa sổ train của các fold đầu. Có lọc ngày `ohlcv_n_bars_1h < 24` không?
   - Đặc trưng trùng thông tin ([`eda_04_feature_corr.png`](../features/reports/figures/eda_04_feature_corr.png)): `cm_CapMVRVCur` gần như trùng return của ngày t (ρ = 0,98), `nvt_ratio_bc` với `bc_est_tx_vol_usd` (ρ = −1,00), `cm_HashRate` với `puell_multiple` (ρ = 0,95). Cần lưu ý khi diễn giải Δ của nhóm Valuation ở E2.
5. **Regime HMM chưa tái lập được từ repo.** `regime_map.csv` là file Ngân xuất sẵn; repo chưa có đoạn code tạo `log_return` và `realized_volatility` làm đầu vào cho `regime_hmm.py`. Thử dựng lại bằng 14 cách tính RV (rv_7d, rv_30d, độ lệch chuẩn 5–30 ngày, |return|, hl_range; giá Binance hoặc CoinMetrics) chỉ khớp 39–59% số ngày. Cần Ngân thêm script tạo `regime_map.csv`.
