"""Task 1.7 + 1.19 (Khoa, Vi): đóng băng dữ liệu (data freeze) ra data/processed/.

Chạy từ bất kỳ đâu: /opt/miniconda3/bin/python src/data_freeze.py [--test]
- Chạy test trên dữ liệu giả trước, rồi mới ghi dữ liệu thật. Thêm --test để chỉ chạy test.
- master_daily.csv lấy từ Khoa (1.7); giữ nguyên từng ký tự của các cột cũ, chỉ thêm cm_NetflowExNtv.
- ohlcv_1h_clean.csv lấy từ Vi (1.9): bản đã làm tròn 43 nến lệch +28 phút sau sự cố 02/2018.
- data_dictionary_A.csv, data_dictionary_B.csv: hai phương án Flows chờ nhóm chốt (1.19). Cả hai thêm dòng
  cm_NetflowExNtv; cột Flows nào không phải đặc trưng của phương án thì role=input (cùng quy ước với
  cm_IssTotUSD của Khoa: vẫn có trong master nhưng không đưa vào mô hình). Master và 1h dùng chung.
- MANIFEST.json: sha256, số dòng, số cột của từng file để mọi người kiểm tra đúng phiên bản.
Sau freeze mọi sửa đổi ra phiên bản mới (v1.1, ...) kèm CHANGELOG.md, không sửa đè.
"""
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # thư mục gốc repo
SRC_MASTER = ROOT / "data/raw/khoa-btc-onchain-ohlcv-daily-2016-2026/master_daily.csv"
SRC_DICT = ROOT / "data/raw/khoa-btc-onchain-ohlcv-daily-2016-2026/data_dictionary.csv"
SRC_1H = ROOT / "data/raw/vi_ohlcv_flows/data/clean/ohlcv_1h_clean.csv"
PROCESSED = ROOT / "data/processed"
VERSION = "v1.0"

# Quyết định 1.19: Netflow = In − Out nên không dùng cả ba cùng lúc (đa cộng tuyến hoàn hảo).
# Bộ A (đề xuất trong qa_report_ohlcv_flows.md): Netflow + Sply. Bộ B: In + Out + Sply.
FLOWS_SET_A = ["cm_NetflowExNtv", "cm_SplyExNtv"]
FLOWS_SET_B = ["cm_FlowInExNtv", "cm_FlowOutExNtv", "cm_SplyExNtv"]
FLOWS_SETS = {"A": FLOWS_SET_A, "B": FLOWS_SET_B}
FLOWS_COLUMNS = ["cm_FlowInExNtv", "cm_FlowOutExNtv", "cm_NetflowExNtv", "cm_SplyExNtv"]


def add_netflow(master: pd.DataFrame) -> pd.DataFrame:
    """master đọc ở dạng chuỗi (dtype=str) để không đổi ký tự nào của cột cũ.
    Thêm cm_NetflowExNtv = cm_FlowInExNtv − cm_FlowOutExNtv ngay sau cm_SplyExNtv; ô thiếu để trống."""
    flow_in = pd.to_numeric(master["cm_FlowInExNtv"].replace("", np.nan))
    flow_out = pd.to_numeric(master["cm_FlowOutExNtv"].replace("", np.nan))
    netflow = (flow_in - flow_out).map(lambda v: "" if pd.isna(v) else repr(float(v)))
    out = master.copy()
    out.insert(out.columns.get_loc("cm_SplyExNtv") + 1, "cm_NetflowExNtv", netflow)
    return out


def update_dictionary(dictionary: pd.DataFrame, set_name: str) -> pd.DataFrame:
    """Thêm dòng cm_NetflowExNtv (ngay sau cm_SplyExNtv) và ghi phương án Flows (A hoặc B) vào role/reason
    của các cột Flows. Cột Flows ngoài phương án: role=input, status vẫn retained (còn trong master)."""
    feature_set = FLOWS_SETS[set_name]
    label = f"flows set {set_name}: {' + '.join(feature_set)}"
    d = dictionary.copy()
    template = d.loc[d["column"] == "cm_FlowInExNtv"].iloc[0].copy()
    netflow_row = template.copy()
    netflow_row.update({
        "column": "cm_NetflowExNtv", "source": "derived", "api_code": "",
        "transform": "cm_FlowInExNtv − cm_FlowOutExNtv (same day)",
        "reason": "Exchange netflow, derived in data freeze; inherits the flash/attribution caveat of In and Out",
    })
    pos = d.index[d["column"] == "cm_SplyExNtv"][0] + 1
    d = pd.concat([d.iloc[:pos], netflow_row.to_frame().T, d.iloc[pos:]], ignore_index=True)

    for col in FLOWS_COLUMNS:
        i = d.index[d["column"] == col][0]
        d.loc[i, "status"] = "retained"
        if col in feature_set:
            d.loc[i, "role"] = "candidate"
            d.loc[i, "reason"] += f"; [1.19] model feature ({label})"
        else:
            d.loc[i, "role"] = "input"
            d.loc[i, "reason"] += f"; [1.19] not a model feature: Netflow = In − Out is perfect collinearity ({label})"
    return d


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _shape(path: Path) -> tuple[int, int]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return len(df), df.shape[1]


def freeze(out_dir: Path = PROCESSED, version: str = VERSION,
           master_path: Path = SRC_MASTER, dict_path: Path = SRC_DICT, h1_path: Path = SRC_1H) -> dict:
    """Ghi master_daily.csv, ohlcv_1h_clean.csv, data_dictionary_A.csv, data_dictionary_B.csv
    và MANIFEST.json vào out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(master_path, dtype=str, keep_default_na=False)
    add_netflow(master).to_csv(out_dir / "master_daily.csv", index=False)
    shutil.copyfile(h1_path, out_dir / "ohlcv_1h_clean.csv")
    dictionary = pd.read_csv(dict_path, dtype=str, keep_default_na=False)
    sources = {"master_daily.csv": master_path, "ohlcv_1h_clean.csv": h1_path}
    for name in FLOWS_SETS:
        update_dictionary(dictionary, name).to_csv(out_dir / f"data_dictionary_{name}.csv", index=False)
        sources[f"data_dictionary_{name}.csv"] = dict_path

    files = {}
    for name, src in sources.items():
        rows, cols = _shape(out_dir / name)
        files[name] = {"sha256": file_sha256(out_dir / name), "rows": rows, "cols": cols,
                       "source": str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src)}
    manifest = {"version": version, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "flows_sets": FLOWS_SETS, "flows_set_chosen": None, "files": files}
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


# ---------------------------------------------------------------- test trên dữ liệu giả
def _fake_inputs(tmp: Path) -> tuple[Path, Path, Path]:
    master = pd.DataFrame({
        "date": ["2016-01-01", "2016-01-02", "2016-01-03"],
        "ohlcv_close": ["", "", "430.10000000000002"],
        "cm_FlowInExNtv": ["32032.874302050001", "100.5", "7"],
        "cm_FlowOutExNtv": ["33163.545528030001", "", "2.25"],
        "cm_SplyExNtv": ["1471426.7652219399", "1471000", "1470000"],
        "cm_HashRate": ["697129.16640580597", "1", "2"],
    })
    dictionary = pd.DataFrame({
        "column": ["date", "cm_FlowInExNtv", "cm_FlowOutExNtv", "cm_SplyExNtv", "cm_HashRate"],
        "source": ["key"] + ["coinmetrics_community"] * 4,
        "api_code": ["", "FlowInExNtv", "FlowOutExNtv", "SplyExNtv", "HashRate"],
        "group": ["key", "flows", "flows", "flows", "mining"],
        "role": ["key", "candidate_pending_go_no_go", "candidate_pending_go_no_go",
                 "candidate_pending_go_no_go", "candidate"],
        "status": ["retained"] * 5,
        "reason": ["key", "Exchange inflow", "Exchange outflow", "Exchange supply", "Hash rate"],
    })
    h1 = pd.DataFrame({"datetime": ["2018-02-09 01:00:00+00:00"], "close": ["8000.0"], "flag_ts_shifted": ["True"]})
    paths = tmp / "master.csv", tmp / "dict.csv", tmp / "h1.csv"
    for df, p in zip((master, dictionary, h1), paths):
        df.to_csv(p, index=False)
    return paths


def _tests():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        master_p, dict_p, h1_p = _fake_inputs(tmp)
        src_master = pd.read_csv(master_p, dtype=str, keep_default_na=False)

        # 1. Netflow = In − Out, đặt ngay sau Sply, ô thiếu để trống
        out = add_netflow(src_master)
        cols = list(out.columns)
        assert cols.index("cm_NetflowExNtv") == cols.index("cm_SplyExNtv") + 1
        assert float(out.loc[0, "cm_NetflowExNtv"]) == 32032.874302050001 - 33163.545528030001
        assert out.loc[1, "cm_NetflowExNtv"] == "" and float(out.loc[2, "cm_NetflowExNtv"]) == 4.75

        # 2. Cột cũ giữ nguyên từng ký tự (kể cả chuỗi số dài và ô trống)
        m = freeze(tmp / "out", master_path=master_p, dict_path=dict_p, h1_path=h1_p)
        frozen = pd.read_csv(tmp / "out/master_daily.csv", dtype=str, keep_default_na=False)
        assert frozen.drop(columns="cm_NetflowExNtv").equals(src_master)

        # 3. File 1h được chép nguyên byte
        assert file_sha256(tmp / "out/ohlcv_1h_clean.csv") == file_sha256(h1_p)

        # 4. Manifest khớp file thật
        assert set(m["files"]) == {"master_daily.csv", "ohlcv_1h_clean.csv",
                                   "data_dictionary_A.csv", "data_dictionary_B.csv"}
        for name, info in m["files"].items():
            assert info["sha256"] == file_sha256(tmp / "out" / name)
        assert m["files"]["master_daily.csv"]["rows"] == 3 and m["files"]["master_daily.csv"]["cols"] == 7
        saved = json.loads((tmp / "out/MANIFEST.json").read_text())
        assert saved["flows_sets"] == FLOWS_SETS and saved["flows_set_chosen"] is None

        def read_dict(name):
            path = tmp / f"out/data_dictionary_{name}.csv"
            return pd.read_csv(path, dtype=str, keep_default_na=False).set_index("column")
        d_a, d_b = read_dict("A"), read_dict("B")

        # 5. Bộ A → Netflow, Sply là đặc trưng; In, Out là input. Bộ B → ngược lại. Không còn pending
        assert list(d_a.index).index("cm_NetflowExNtv") == list(d_a.index).index("cm_SplyExNtv") + 1
        assert d_a.loc["cm_NetflowExNtv", "source"] == "derived"
        assert d_a.loc[FLOWS_SET_A, "role"].eq("candidate").all()
        assert d_a.loc[["cm_FlowInExNtv", "cm_FlowOutExNtv"], "role"].eq("input").all()
        assert d_b.loc[FLOWS_SET_B, "role"].eq("candidate").all() and d_b.loc["cm_NetflowExNtv", "role"] == "input"
        assert not pd.concat([d_a, d_b])["role"].str.contains("pending").any()

        # 6. Hai dictionary chỉ khác nhau ở role/reason của các cột Flows
        assert list(d_a.index) == list(d_b.index)
        others = [c for c in d_a.index if c not in FLOWS_COLUMNS]
        assert d_a.loc[others].equals(d_b.loc[others])
        assert d_a.drop(columns=["role", "reason"]).equals(d_b.drop(columns=["role", "reason"]))
    print("data_freeze: 6/6 test qua")


if __name__ == "__main__":
    _tests()
    if "--test" not in sys.argv:
        manifest = freeze()
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
