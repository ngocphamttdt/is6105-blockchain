"""
data.py  –  Bước 0 (Ngọc): đọc data freeze trong data/processed/ và tên cột dùng chung.

- Chỉ đọc, không bao giờ ghi vào data/processed/. Thấy lỗi dữ liệu thì báo Khoa hoặc Vi.
- load_master() so sha256 của mọi file với MANIFEST.json trước khi đọc.
- onchain_columns("A" | "B") lấy chỉ số On-chain đưa vào mô hình từ data_dictionary_{A|B}.csv.
- Tên cột giữ nguyên như freeze (ohlcv_*, cm_*, bc_*). Mọi file khác chỉ lấy tên cột từ đây.

python data.py         -> chạy test rồi kiểm tra freeze thật
python data.py --test  -> chỉ chạy test
"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]                   # thư mục gốc repo
PROCESSED = ROOT / "data/processed"
MASTER_CSV = PROCESSED / "master_daily.csv"
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
RESULTS = Path(__file__).resolve().parents[1] / "results"

# ---- Tên cột ----
DATE, TARGET = "date", "y"                                   # y_t = 1 nếu close_{t+1} > close_t
OHLCV_COLS = ["ohlcv_open", "ohlcv_high", "ohlcv_low", "ohlcv_close", "ohlcv_volume"]
CLOSE = "ohlcv_close"
NETFLOW = "cm_NetflowExNtv"                                  # = FlowIn − FlowOut, âm ở ~48% số ngày
# Mọi chỉ số On-chain ứng viên của cả hai phương án; cột nào vào mô hình do data dictionary quyết định
ONCHAIN_GROUPS = {
    "Activity": ["cm_AdrActCnt", "cm_TxCnt", "bc_est_tx_vol_usd"],
    "Flows": ["cm_NetflowExNtv", "cm_FlowInExNtv", "cm_FlowOutExNtv", "cm_SplyExNtv"],
    "Mining": ["cm_HashRate", "cm_FeeTotNtv", "bc_difficulty"],
    "Valuation": ["cm_CapMVRVCur", "puell_multiple", "nvt_ratio_bc"],
}
COL_GROUP = {c: g for g, cols in ONCHAIN_GROUPS.items() for c in cols}
FLOWS_SETS = ("A", "B")                                      # A: Netflow + Sply; B: In + Out + Sply (chờ 1.19)
PREDICTION_COLS = ["date", "fold", "y_true", "p_hat", "regime"]


def load_config(name) -> dict:
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


def read_csv_dated(path) -> pd.DataFrame:
    """Đọc CSV có cột `date`; trả cột `date` kiểu datetime64[ns]. round_trip: số thực đọc lại đúng từng bit
    như lúc ghi (parser mặc định của pandas 3 có thể lệch 1 ulp)."""
    df = pd.read_csv(path, parse_dates=[DATE], float_precision="round_trip")
    df[DATE] = df[DATE].astype("datetime64[ns]")
    return df


def write_csv(df: pd.DataFrame, path) -> Path:
    """Ghi CSV không index, ngày dạng YYYY-MM-DD, số thực giữ đủ chữ số để đọc lại y nguyên."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, date_format="%Y-%m-%d")
    return path


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_freeze(directory=PROCESSED) -> dict:
    """So sha256 của mọi file trong MANIFEST.json; lệch hoặc thiếu file thì báo lỗi. Trả manifest."""
    directory = Path(directory)
    manifest = json.loads((directory / "MANIFEST.json").read_text())
    bad = [name for name, meta in manifest["files"].items()
           if not (directory / name).exists() or _sha256(directory / name) != meta["sha256"]]
    if bad:
        raise ValueError(f"File không khớp MANIFEST {manifest['version']} (thiếu hoặc sai sha256): {bad}")
    return manifest


def load_master(path=MASTER_CSV, verify=True) -> pd.DataFrame:
    """Bảng chính của freeze: một dòng mỗi ngày UTC, tên cột giữ nguyên."""
    path = Path(path)
    if verify:
        verify_freeze(path.parent)
    df = read_csv_dated(path)
    missing = set(OHLCV_COLS + list(COL_GROUP)) - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột: {sorted(missing)}")
    if (df[DATE].diff().iloc[1:] != pd.Timedelta(days=1)).any():
        raise ValueError("Cột date phải tăng dần, không trùng và đủ mọi ngày.")
    return df


def onchain_columns(flows_set="A", directory=PROCESSED) -> list[str]:
    """Chỉ số đưa vào mô hình: dòng status = retained, role là candidate hoặc backup (README freeze)."""
    if flows_set not in FLOWS_SETS:
        raise ValueError(f"flows_set phải là một trong {FLOWS_SETS}, nhận '{flows_set}'.")
    d = pd.read_csv(Path(directory) / f"data_dictionary_{flows_set}.csv")
    cols = d.loc[(d.status == "retained") & d.role.isin(["candidate", "backup"]), "column"].tolist()
    unknown = [c for c in cols if c not in COL_GROUP]
    if unknown:
        raise ValueError(f"Chỉ số chưa có nhóm trong COL_GROUP: {unknown}")
    return cols


def make_synthetic_master(start="2017-09-01", end="2026-08-31", seed=0) -> pd.DataFrame:
    """Dữ liệu giả đúng schema freeze, dùng cho test. Giá theo GBM; On-chain dương, random walk trên log;
    HashRate tăng đơn điệu; FlowIn và FlowOut quanh cùng một mức nên Netflow đổi dấu như dữ liệu thật."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    close = 4000 * np.exp(np.cumsum(rng.normal(0.0005, 0.035, n)))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.003, n))
    spread = np.abs(rng.normal(0, 0.02, n))
    df = pd.DataFrame({DATE: dates, "ohlcv_open": open_,
                       "ohlcv_high": np.maximum(open_, close) * (1 + spread),
                       "ohlcv_low": np.minimum(open_, close) * (1 - spread),
                       CLOSE: close, "ohlcv_volume": np.exp(rng.normal(10, 0.5, n))})
    flow_level = 3e4 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    for col in COL_GROUP:
        if col == "cm_HashRate":
            df[col] = 1e7 * np.exp(np.cumsum(np.abs(rng.normal(0.002, 0.01, n))))
        elif col in ("cm_FlowInExNtv", "cm_FlowOutExNtv"):
            df[col] = flow_level * np.exp(rng.normal(0, 0.1, n))
        elif col != NETFLOW:
            df[col] = rng.uniform(1, 1e6) * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    df[NETFLOW] = df["cm_FlowInExNtv"] - df["cm_FlowOutExNtv"]
    return df


def make_synthetic_close_1h(master: pd.DataFrame, seed=0) -> pd.Series:
    """Giá 1h giả (index UTC) cho rv_7d, rv_30d khi test: 24 nến mỗi ngày quanh close ngày đó,
    nến cuối ngày bằng đúng close ngày."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(master[DATE].iloc[0], periods=24 * len(master), freq="h", tz="UTC")
    noise = rng.normal(0, 0.005, len(idx))
    noise[23::24] = 0
    return pd.Series(np.repeat(master[CLOSE].to_numpy(), 24) * np.exp(noise), index=idx, name="close")


# ----------------------------------------------------------------------------
# Test:  python data.py --test
# ----------------------------------------------------------------------------
def _raises(fn, text=""):
    try:
        fn()
    except ValueError as e:
        assert text in str(e), f"Lỗi đúng loại nhưng sai nội dung: {e}"
        return
    raise AssertionError("Phải báo ValueError")


def _tests():
    # 1) verify_freeze phát hiện file bị sửa hoặc bị xoá
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        files = {"a.csv": b"date,x\n2020-01-01,1\n", "b.csv": b"date,y\n2020-01-01,2\n"}
        for name, content in files.items():
            (tmp / name).write_bytes(content)
        sums = {n: {"sha256": hashlib.sha256(c).hexdigest()} for n, c in files.items()}
        (tmp / "MANIFEST.json").write_text(json.dumps({"version": "vtest", "files": sums}))
        assert verify_freeze(tmp)["version"] == "vtest"
        (tmp / "a.csv").write_bytes(b"date,x\n2020-01-01,7\n")
        _raises(lambda: verify_freeze(tmp), "a.csv")
        (tmp / "a.csv").write_bytes(files["a.csv"])
        (tmp / "b.csv").unlink()
        _raises(lambda: verify_freeze(tmp), "b.csv")

    # 2) dữ liệu giả đúng schema; Netflow đổi dấu khoảng một nửa số ngày
    m = make_synthetic_master(seed=42)
    assert set(OHLCV_COLS) | set(COL_GROUP) <= set(m.columns)
    assert 0.3 < (m[NETFLOW] < 0).mean() < 0.7

    # 3) freeze thật (nếu có): sha256 khớp; cột chọn theo dictionary đúng bảng trong README của freeze
    if MASTER_CSV.exists():
        verify_freeze()
        others = ["cm_AdrActCnt", "cm_TxCnt", "bc_est_tx_vol_usd", "cm_HashRate", "cm_FeeTotNtv",
                  "bc_difficulty", "cm_CapMVRVCur", "puell_multiple", "nvt_ratio_bc"]
        assert sorted(onchain_columns("A")) == sorted(others + ["cm_NetflowExNtv", "cm_SplyExNtv"])
        assert sorted(onchain_columns("B")) == sorted(others + ["cm_FlowInExNtv", "cm_FlowOutExNtv",
                                                                "cm_SplyExNtv"])
        _raises(lambda: onchain_columns("C"), "flows_set")
    print("Tất cả test data đều PASS")


if __name__ == "__main__":
    _tests()
    if "--test" not in sys.argv:
        man = verify_freeze()
        print(f"Data freeze {man['version']}: sha256 khớp {len(man['files'])} file; "
              f"flows_set_chosen = {man.get('flows_set_chosen')}")
