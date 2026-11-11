# eda — Kiểm tra dữ liệu và EDA lại từ đầu (góp ý 1 của thầy)

Chỉ đọc `data/processed/` (data freeze v1.0, kiểm sha256 với `MANIFEST.json`). Không dùng kết quả hay code của thư mục nào khác.
Theo quy tắc của data freeze, không có phép tính nào giữa chỉ số On-chain và nhãn.

```
pip install pandas numpy scipy matplotlib tabulate
python eda/src/eda.py
```

Kết quả: `reports/eda.md` (đọc phần Tóm tắt trước), hình trong `reports/figures/`, bảng trong `reports/tables/`.
