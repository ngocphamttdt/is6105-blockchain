# Trước và sau tuning của mọi baseline (B0–B5)

Tạo bởi `analyses/tuning_summary/src/tuning_summary.py`. Chỉ đọc file dự báo đã có, không chạy lại mô hình. Mọi dòng dùng cùng 2.434 ngày test walk-forward (01/01/2020 → 30/08/2026, 82 fold) và cùng cách tính metric.

- **Mức tune:** chưa tune (luật, tham số mặc định hoặc cố định) · grid gốc (Phụ lục C, dự báo E1) · tune lại.
- **Δ so với chưa tune:** dòng hiện tại trừ dòng ở cột *so với* (bản chưa tune cùng code). ΔDA tính bằng điểm %, CI 95% block bootstrap 15 ngày. Δlog-loss < 0 là tốt hơn.
- **ΔDA vs đảo chiều:** so với luật `b0rev` (53,25%), mốc mà chưa mô hình có học nào vượt được.
- Bảng đầy đủ (p-value McNemar, DM, Brier, nguồn từng dòng): `results/tuning_summary.csv`.

## OHLCV-only

| Baseline | Mức | Mô hình | DA (%) | Log-loss | AUC | % đoán tăng | ΔDA so với chưa tune | Δlog-loss so với chưa tune | ΔDA vs đảo chiều |
|---|---|---|---:|---:|---:|---:|---|---|---|
| B0 | chưa tune | `b0maj` | 50,08 | 0,6938 | – | 63,0 |  |  | -3,16 [-5,75; -0,62] |
| B0 | chưa tune | `b0pers` | 46,51 | – | 0,465 | 50,9 |  |  | -6,74 [-10,72; -2,79] |
| B0 | chưa tune | `b0rev` | 53,25 | – | 0,533 | 49,4 |  |  |  |
| B1 | chưa tune | `b1 tham số mặc định` | 50,70 | 0,6991 | 0,511 | 52,5 |  |  | -2,55 [-4,60; -0,53] |
| B1 | grid gốc | `b1` | 51,11 | 0,6953 | 0,510 | 56,4 | +0,41 [-0,78; +1,60] | -0,0038 [-0,0061; -0,0017] | -2,14 [-4,23; +0,00] |
| B1 | tune lại | `b1t` | 53,20 | 0,6926 | 0,530 | 49,0 | +2,51 [+0,45; +4,56] | -0,0065 [-0,0105; -0,0026] | -0,04 [-0,86; +0,78] |
| B1 | tune lại | `b1 grid mở rộng (gate)` | 50,16 | 0,6953 | 0,499 | 59,2 | -0,53 [-2,22; +1,19] | -0,0037 [-0,0075; -0,0002] | -3,08 [-5,46; -0,66] |
| B2 | chưa tune | `b2 tham số mặc định` | 51,36 | 0,7178 | 0,511 | 52,4 |  |  | -1,89 [-4,60; +0,90] |
| B2 | grid gốc | `b2` | 51,23 | 0,6961 | 0,514 | 54,7 | -0,12 [-2,22; +1,97] | -0,0217 [-0,0299; -0,0132] | -2,01 [-4,27; +0,21] |
| B2 | tune lại | `b2t` | 51,15 | 0,6947 | 0,517 | 54,5 | -0,21 [-2,71; +2,26] | -0,0231 [-0,0319; -0,0140] | -2,10 [-4,19; +0,00] |
| B2 | tune lại | `b2 grid mở rộng (gate)` | 51,64 | 0,6951 | 0,514 | 57,0 | +0,29 [-1,97; +2,63] | -0,0227 [-0,0315; -0,0138] | -1,60 [-3,78; +0,62] |
| B3 | chưa tune | `b3 tham số mặc định` | 49,88 | 0,6951 | 0,501 | 64,9 |  |  | -3,37 [-5,96; -0,62] |
| B3 | grid gốc | `b3` | 50,62 | 0,7009 | 0,503 | 72,8 | +0,74 [-0,70; +2,18] | +0,0058 [+0,0003; +0,0123] | -2,63 [-5,34; +0,21] |
| B3 | tune lại | `b3 grid 5×6` | 50,66 | 0,6959 | 0,507 | 75,4 | +0,78 [-0,62; +2,18] | +0,0008 [-0,0020; +0,0046] | -2,59 [-5,34; +0,25] |
| B3 | tune lại | `b3 grid mở rộng (gate)` | 50,66 | 0,7290 | 0,503 | 71,0 | +0,78 [-0,62; +2,22] | +0,0339 [+0,0089; +0,0690] | -2,59 [-5,26; +0,29] |
| B4 | chưa tune | `b4 tham số mặc định` | 50,37 | 0,6969 | 0,510 | 60,1 |  |  | -2,88 [-5,09; -0,45] |
| B4 | grid gốc | `b4` | 50,45 | 0,6973 | 0,502 | 62,7 | +0,08 [-1,52; +1,60] | +0,0005 [-0,0023; +0,0029] | -2,79 [-5,10; -0,41] |
| B4 | tune lại | `b4 grid mở rộng (gate)` | 50,62 | 0,6955 | 0,509 | 62,8 | +0,25 [-1,23; +1,64] | -0,0014 [-0,0046; +0,0016] | -2,63 [-4,89; -0,16] |
| B5 | chưa tune | `b5` | 50,74 | 0,6950 | 0,500 | 57,5 |  |  | -2,51 [-5,26; +0,25] |
| B5 | chưa tune | `b5 lstm_tuning trước tune` | 50,00 | 0,6947 | 0,503 | 59,6 |  |  | -3,25 [-5,63; -0,70] |
| B5 | tune lại | `b5 lstm_tuning sau tune` | 50,41 | 0,6933 | 0,507 | 55,0 | +0,41 [-0,90; +1,73] | -0,0013 [-0,0030; +0,0002] | -2,83 [-5,55; -0,12] |

## OHLCV + On-chain

| Baseline | Mức | Mô hình | DA (%) | Log-loss | AUC | % đoán tăng | ΔDA so với chưa tune | Δlog-loss so với chưa tune | ΔDA vs đảo chiều |
|---|---|---|---:|---:|---:|---:|---|---|---|
| B0 | chưa tune | `b0maj` | 50,08 | 0,6938 | – | 63,0 |  |  | -3,16 [-5,75; -0,62] |
| B0 | chưa tune | `b0pers` | 46,51 | – | 0,465 | 50,9 |  |  | -6,74 [-10,72; -2,79] |
| B0 | chưa tune | `b0rev` | 53,25 | – | 0,533 | 49,4 |  |  |  |
| B1 | chưa tune | `b1 tham số mặc định` | 50,08 | 0,7413 | 0,511 | 53,0 |  |  | -3,16 [-5,55; -0,62] |
| B1 | grid gốc | `b1` | 51,89 | 0,7006 | 0,513 | 54,7 | +1,81 [+0,00; +3,53] | -0,0407 [-0,0665; -0,0234] | -1,36 [-3,90; +1,11] |
| B1 | tune lại | `b1t` | 51,19 | 0,6944 | 0,521 | 51,4 | +1,11 [-1,11; +3,21] | -0,0469 [-0,0752; -0,0274] | -2,05 [-4,27; +0,04] |
| B2 | chưa tune | `b2 tham số mặc định` | 50,74 | 0,7082 | 0,509 | 51,7 |  |  | -2,51 [-5,18; +0,04] |
| B2 | grid gốc | `b2` | 51,40 | 0,6957 | 0,515 | 54,1 | +0,66 [-1,40; +2,71] | -0,0126 [-0,0184; -0,0067] | -1,85 [-4,52; +0,86] |
| B2 | tune lại | `b2t` | 49,22 | 0,6951 | 0,506 | 49,2 | -1,52 [-3,58; +0,58] | -0,0132 [-0,0195; -0,0069] | -4,03 [-6,25; -1,64] |
| B3 | chưa tune | `b3 tham số mặc định` | 49,47 | 0,6959 | 0,495 | 61,4 |  |  | -3,78 [-6,37; -1,11] |
| B3 | grid gốc | `b3` | 50,29 | 0,6959 | 0,498 | 64,7 | +0,82 [-0,70; +2,30] | +0,0000 [-0,0017; +0,0020] | -2,96 [-5,46; -0,45] |
| B3 | tune lại | `b3 grid 5×6` | 49,01 | 0,6966 | 0,494 | 64,0 | -0,45 [-1,97; +1,03] | +0,0007 [-0,0015; +0,0028] | -4,23 [-6,78; -1,60] |
| B4 | chưa tune | `b4 tham số mặc định` | 51,44 | 0,6948 | 0,517 | 55,3 |  |  | -1,81 [-4,35; +0,90] |
| B4 | grid gốc | `b4` | 48,97 | 0,6958 | 0,499 | 58,0 | -2,47 [-4,07; -0,99] | +0,0010 [-0,0015; +0,0036] | -4,27 [-6,78; -1,73] |
| B5 | chưa tune | `b5` | 50,58 | 0,6959 | 0,509 | 33,4 |  |  | -2,67 [-5,42; -0,04] |

## Nhận xét (tính tự động từ bảng)

- Có 19 lần so một bản đã tune với bản chưa tune cùng code. Log-loss tốt lên có ý nghĩa (CI nằm hẳn dưới 0) ở 10 lần: `b1` ohlcv, `b1t` ohlcv, `b1 grid mở rộng (gate)` ohlcv, `b1` ohlcv_onchain, `b1t` ohlcv_onchain, `b2` ohlcv, `b2t` ohlcv, `b2 grid mở rộng (gate)` ohlcv, `b2` ohlcv_onchain, `b2t` ohlcv_onchain; xấu đi có ý nghĩa ở 2 lần: `b3` ohlcv, `b3 grid mở rộng (gate)` ohlcv.
- DA cao nhất trong các mô hình có học: 53,20% (`b1t`, ohlcv); luật đảo chiều đúng 53,25%. Số mô hình có học có DA cao hơn đảo chiều với CI không chứa 0: 0.
- Bản có log-loss thấp nhất của từng baseline: B1 ohlcv: `b1t` (tune lại, 0,6926); B1 ohlcv_onchain: `b1t` (tune lại, 0,6944); B2 ohlcv: `b2t` (tune lại, 0,6947); B2 ohlcv_onchain: `b2t` (tune lại, 0,6951); B3 ohlcv: `b3 tham số mặc định` (chưa tune, 0,6951); B3 ohlcv_onchain: `b3 tham số mặc định` (chưa tune, 0,6959); B4 ohlcv: `b4 grid mở rộng (gate)` (tune lại, 0,6955); B4 ohlcv_onchain: `b4 tham số mặc định` (chưa tune, 0,6948); B5 ohlcv: `b5 lstm_tuning sau tune` (tune lại, 0,6933); B5 ohlcv_onchain: `b5` (chưa tune, 0,6959).

## Ghi chú về nguồn

- `b0maj`: p̂ = tỷ lệ ngày tăng của cửa sổ train. Nguồn (ví dụ cấu hình ohlcv): `pipeline/results/predictions/b0maj_ohlcv.csv`
- `b0pers`: ngày mai cùng hướng hôm nay. Nguồn (ví dụ cấu hình ohlcv): `pipeline/results/predictions/b0pers_ohlcv.csv`
- `b0rev`: ngày mai ngược hướng hôm nay; chiều chọn trên train. Nguồn (ví dụ cấu hình ohlcv): `pipeline/results/predictions/b0rev_ohlcv.csv`
- `b1 tham số mặc định`: tham số trong grid để ở mặc định của sklearn (experiments.py --stage tuning). Nguồn (ví dụ cấu hình ohlcv): `pipeline/results/tuning/default/b1_ohlcv.csv`
- `b1`: dự báo E1. Nguồn (ví dụ cấu hình ohlcv): `pipeline/results/predictions/b1_ohlcv.csv`
- `b1t`: tune cả tập đặc trưng, C, L1/L2, class_weight (b1t_tuning.py --stage e1). Nguồn (ví dụ cấu hình ohlcv): `pipeline/results/b1t/e1/ohlcv/predictions/b1t_ohlcv.csv`
- `b1 grid mở rộng (gate)`: grid mở rộng của gate.py; 10 OHLCV + winsorize 1%/99% (như E2); chỉ OHLCV. Nguồn (ví dụ cấu hình ohlcv): `pipeline/results/gate/baseline/b1.csv`
- `b2 tham số mặc định`: tham số trong grid để ở mặc định của sklearn (experiments.py --stage tuning). Nguồn (ví dụ cấu hình ohlcv): `pipeline/results/tuning/default/b2_ohlcv.csv`
- `b2`: dự báo E1. Nguồn (ví dụ cấu hình ohlcv): `pipeline/results/predictions/b2_ohlcv.csv`
- `b2t`: grid thiết kế từ dữ liệu trước 2020 (analyses/b2_tuning). Nguồn (ví dụ cấu hình ohlcv): `pipeline/results/predictions/b2t_ohlcv.csv`
- `b2 grid mở rộng (gate)`: grid mở rộng của gate.py; 10 OHLCV + winsorize 1%/99% (như E2); chỉ OHLCV. Nguồn (ví dụ cấu hình ohlcv): `pipeline/results/gate/baseline/b2.csv`
- `b3 tham số mặc định`: tham số trong grid để ở mặc định của sklearn (experiments.py --stage tuning). Nguồn (ví dụ cấu hình ohlcv): `pipeline/results/tuning/default/b3_ohlcv.csv`
- `b3`: dự báo E1. Nguồn (ví dụ cấu hình ohlcv): `pipeline/results/predictions/b3_ohlcv.csv`
- `b3 grid 5×6`: C × gamma 5 × 6 (analyses/baselines, Trâm). Nguồn (ví dụ cấu hình ohlcv): `analyses/baselines/results/predictions/b3g30_ohlcv.csv`
- `b3 grid mở rộng (gate)`: grid mở rộng của gate.py; 10 OHLCV + winsorize 1%/99% (như E2); chỉ OHLCV. Nguồn (ví dụ cấu hình ohlcv): `pipeline/results/gate/baseline/b3.csv`
- `b4 tham số mặc định`: tham số trong grid để ở mặc định của sklearn (experiments.py --stage tuning). Nguồn (ví dụ cấu hình ohlcv): `pipeline/results/tuning/default/b4_ohlcv.csv`
- `b4`: dự báo E1. Bản chạy lại cùng grid trong pipeline/results/tuning/tuned/ lệch 312–359 ngày quyết định: XGBoost không tái lập được giữa hai lần chạy (khác máy / phiên bản). Nguồn (ví dụ cấu hình ohlcv): `pipeline/results/predictions/b4_ohlcv.csv`
- `b4 grid mở rộng (gate)`: grid mở rộng của gate.py; 10 OHLCV + winsorize 1%/99% (như E2); chỉ OHLCV. Nguồn (ví dụ cấu hình ohlcv): `pipeline/results/gate/baseline/b4.csv`
- `b5`: dự báo E1, tham số cố định, trung bình 3 seed. Nguồn (ví dụ cấu hình ohlcv): `pipeline/results/predictions/b5_ohlcv.csv`
- `b5 lstm_tuning trước tune`: code analyses/lstm_tuning (Ngọc), khác regime/model_lstm.py của B5 trong E1; chỉ OHLCV. Nguồn (ví dụ cấu hình ohlcv): `analyses/lstm_tuning/reports/predictions_walkforward.csv [cột B5 LSTM trước tune]`
- `b5 lstm_tuning sau tune`: code analyses/lstm_tuning (Ngọc), khác regime/model_lstm.py của B5 trong E1; chỉ OHLCV; grid 16 cấu hình chọn trên validation 2022–2023. Nguồn (ví dụ cấu hình ohlcv): `analyses/lstm_tuning/reports/predictions_walkforward.csv [cột B5 LSTM sau tune]`
