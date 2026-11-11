"""
eda.py  –  Kiểm tra dữ liệu và EDA lại từ đầu (góp ý 1 của thầy).

Chỉ đọc data/processed/ (data freeze v1.0). Không dùng kết quả, đặc trưng hay code của thư mục nào khác.
Mọi kiểm tra là assert: báo cáo chỉ được ghi khi tất cả đều đúng.

Quy tắc của data freeze (data/processed/README.md, mục 6): KHÔNG tính tương quan giữa chỉ số On-chain và nhãn
trước khi chạy thí nghiệm. Vì vậy:
  - Phần nhãn chỉ dùng giá (OHLCV).
  - Phần On-chain chỉ xem chất lượng và đặc tính của từng chuỗi (thiếu, bước nhảy, xu hướng, mùa vụ tuần,
    tương quan giữa các chỉ số On-chain với nhau). Không có bảng hay hình nào đặt On-chain cạnh nhãn.

Chạy từ thư mục gốc repo:
  python eda/src/eda.py
Ghi: eda/reports/eda.md, eda/reports/figures/*.png, eda/reports/tables/*.csv
"""
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "eda" / "reports"
FIG, TAB = OUT / "figures", OUT / "tables"

STUDY_START = pd.Timestamp("2018-01-01")      # giai đoạn nghiên cứu theo proposal
STUDY_END = pd.Timestamp("2026-08-31")
HALVINGS = [pd.Timestamp("2020-05-11"), pd.Timestamp("2024-04-20")]
CRASH = pd.Timestamp("2020-03-12")
OHLCV = ["ohlcv_open", "ohlcv_high", "ohlcv_low", "ohlcv_close", "ohlcv_volume"]
WEEKDAYS = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]
PERS = "tỷ lệ cùng hướng hôm trước"

report: list[str] = []
K: dict = {}                                   # số liệu chính, dùng cho phần Tóm tắt


def say(*lines):
    report.extend(lines)


def md(df, floatfmt=".4f", index=False):
    """Bảng markdown; cột số nguyên (năm, số ngày) giữ dạng nguyên thay vì bị in 2018.0000."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].astype(str)
    return df.to_markdown(index=index, floatfmt=floatfmt)


def num(x, d=0):
    """Số kiểu Việt trong phần chữ: 3.164 ngày, 0,125."""
    s = f"{x:,.{d}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x, d=2):
    return num(100 * x, d) + "%"


def pval(p):
    return "< 0,0001" if p < 1e-4 else num(p, 4)


def save_fig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=120)
    plt.close(fig)
    return f"figures/{name}"


def save_tab(df, name):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / name, index=False)


# ============================================================================
# 0. Đọc dữ liệu, kiểm phiên bản
# ============================================================================
def load():
    man = json.loads((DATA / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, info in man["files"].items():
        sha = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        assert sha == info["sha256"], f"{name}: sha256 không khớp MANIFEST"
    m = pd.read_csv(DATA / "master_daily.csv", parse_dates=["date"], float_precision="round_trip")
    h = pd.read_csv(DATA / "ohlcv_1h_clean.csv", float_precision="round_trip")
    h["datetime"] = pd.to_datetime(h["datetime"], utc=True)
    dicts = {k: pd.read_csv(DATA / f"data_dictionary_{k}.csv") for k in ("A", "B")}
    for name, df in (("master_daily.csv", m), ("ohlcv_1h_clean.csv", h)):
        assert df.shape == (man["files"][name]["rows"], man["files"][name]["cols"]), name
    return man, m, h, dicts


def onchain_columns(d):
    """Cột On-chain đưa vào mô hình theo dictionary: status retained, role candidate hoặc backup."""
    sel = d[(d.status == "retained") & d.role.isin(["candidate", "backup"])]
    return dict(zip(sel.column, sel.group))


# ============================================================================
# 1. Kiểm tra dữ liệu
# ============================================================================
def check_data(man, m, h, dicts):
    say("## Phần 1. Kiểm tra dữ liệu", "")

    # 1a. Lịch ngày
    dates = m["date"]
    assert dates.is_unique and dates.is_monotonic_increasing
    assert (dates.diff().iloc[1:] == pd.Timedelta(days=1)).all(), "master thiếu ngày"
    first_px = m.loc[m.ohlcv_close.notna(), "date"].min()
    assert m.loc[m.date >= first_px, OHLCV].notna().all().all(), "OHLCV thiếu ngày sau ngày niêm yết"
    say("### 1.1 Lịch ngày và ô trống", "",
        f"`master_daily.csv`: {num(len(m))} ngày liên tục từ {dates.min().date()} đến {dates.max().date()}, "
        f"không trùng, không thiếu ngày (sha256 của 4 file khớp `MANIFEST.json` {man['version']}). "
        f"Giá Binance có từ {first_px.date()} và không thiếu ngày nào sau đó.", "")

    on_a, on_b = onchain_columns(dicts["A"]), onchain_columns(dicts["B"])
    cols = list(dict.fromkeys(OHLCV + list(on_a) + list(on_b)))
    study = m[(m.date >= STUDY_START) & (m.date <= STUDY_END)]
    miss = pd.DataFrame({
        "cột": cols,
        "nhóm": [on_a.get(c, on_b.get(c, "ohlcv")) for c in cols],
        "dùng trong A": ["có" if c in on_a else "" for c in cols],
        "dùng trong B": ["có" if c in on_b else "" for c in cols],
        "ngày đầu có số": [m.loc[m[c].notna(), "date"].min().date() for c in cols],
        "ô trống (cả file)": [int(m[c].isna().sum()) for c in cols],
        "ô trống từ 2018": [int(study[c].isna().sum()) for c in cols],
    })
    save_tab(miss, "missing.csv")
    gaps = {c: study.loc[study[c].isna(), "date"].dt.date.astype(str).tolist() for c in cols if study[c].isna().any()}
    K["gaps"] = gaps
    say(md(miss), "",
        "Ô trống từ 2018 (giai đoạn nghiên cứu): "
        + "; ".join(f"`{c}`: {', '.join(v)}" for c, v in gaps.items()) + ". "
        "Ô trống còn lại đều trước 2018 (giá Binance trước niêm yết, Puell cần 365 ngày lịch sử).", "")

    # 1b. OHLC hợp lệ
    px = m.dropna(subset=OHLCV)
    bad = px[(px.ohlcv_low > px[["ohlcv_open", "ohlcv_close"]].min(axis=1)) |
             (px.ohlcv_high < px[["ohlcv_open", "ohlcv_close"]].max(axis=1)) |
             (px[OHLCV] <= 0).any(axis=1)]
    assert bad.empty, f"OHLC không hợp lệ: {bad.date.tolist()}"
    say("### 1.2 Giá ngày khớp nến 1 giờ", "",
        f"Mọi ngày có giá: low ≤ open, close ≤ high và mọi giá, volume > 0 ({num(len(px))} ngày).", "")

    # 1c. Gộp lại nến 1h
    assert h["datetime"].is_unique and h["datetime"].is_monotonic_increasing
    assert (h["datetime"].dt.minute == 0).all(), "nến 1h không bắt đầu đúng đầu giờ"
    h = h.assign(day=h["datetime"].dt.tz_localize(None).dt.normalize())
    g = h.groupby("day").agg(o=("open", "first"), hi=("high", "max"), lo=("low", "min"), c=("close", "last"),
                             v=("volume", "sum"), n=("close", "size"))
    j = m.set_index("date").join(g, how="inner")
    diffs = {a: float((j[a] - j[b]).abs().max()) for a, b in
             [("ohlcv_open", "o"), ("ohlcv_high", "hi"), ("ohlcv_low", "lo"), ("ohlcv_close", "c"),
              ("ohlcv_volume", "v"), ("ohlcv_n_bars_1h", "n")]}
    assert max(diffs.values()) < 1e-6, diffs
    only_master = m[(m.date >= first_px) & (m.date < g.index.min())]
    short = m[m.ohlcv_n_bars_1h < 24][["date", "ohlcv_n_bars_1h"]]
    last_hour = h.groupby("day")["datetime"].max().dt.hour
    close_not_23 = last_hour[last_hour != 23]
    K.update(n_days_1h=len(j), n_short=len(short), n_short_study=int((short.date >= STUDY_START).sum()),
             close_not_23=[str(d.date()) for d in close_not_23.index])
    say(f"Gộp {num(len(h))} nến 1h theo ngày UTC (open đầu, high max, low min, close cuối, volume tổng) rồi so với "
        f"master trên {num(len(j))} ngày chung: lệch lớn nhất "
        + ", ".join(f"{k.removeprefix('ohlcv_')} {v:.1e}" for k, v in diffs.items()) + ". "
        f"{len(only_master)} ngày đầu ({only_master.date.min().date()} → {only_master.date.max().date()}) chỉ có "
        f"trong master vì file 1h bắt đầu {g.index.min().date()}; các ngày này trước giai đoạn nghiên cứu.", "",
        f"**Ngày thiếu nến:** {len(short)} ngày có dưới 24 nến, {K['n_short_study']} ngày trong "
        f"giai đoạn nghiên cứu. Giá đóng cửa của ngày là giá của nến cuối cùng trong ngày; ngày mà nến cuối không phải "
        f"nến 23:00 UTC: {', '.join(K['close_not_23'])} "
        f"(nến cuối lúc {', '.join(f'{x:02d}:00' for x in close_not_23)}). "
        f"Ở những ngày này, \"giá đóng cửa\" không phải giá cuối ngày, nên nhãn của ngày đó và ngày trước bị lệch "
        f"nghĩa. {int(h.flag_ts_shifted.sum())} nến có `flag_ts_shifted` (đã làm tròn về đầu giờ).", "")

    # 1d. Đối chiếu nguồn giá độc lập: CoinMetrics PriceUSD
    s = m.dropna(subset=["ohlcv_close", "cm_PriceUSD"]).set_index("date")
    rel = (s.ohlcv_close / s.cm_PriceUSD - 1)
    rel_s = rel[rel.index >= STUDY_START]
    top = rel_s.abs().nlargest(5)
    in_study = s.index >= STUDY_START
    agree = (np.sign(s.ohlcv_close.diff()) == np.sign(s.cm_PriceUSD.diff()))[in_study]
    sign_agree = agree.mean()
    move = np.log(s.ohlcv_close).diff().abs()[in_study]
    small_dis = (move[~agree] < 0.005).mean()           # ngày hai nguồn ngược hướng: bao nhiêu % là ngày đi < 0,5%
    small_all = (move < 0.005).mean()
    assert small_dis > small_all
    K.update(px_rel_median=rel_s.abs().median(), px_sign_agree=sign_agree)
    say("### 1.3 Đối chiếu với nguồn giá thứ hai", "",
        "`cm_PriceUSD` là giá tham chiếu của CoinMetrics (tổng hợp nhiều sàn), độc lập với Binance. "
        f"Từ 2018, giá đóng cửa Binance lệch với nó trung vị {pct(rel_s.abs().median(), 3)}, "
        f"95% số ngày dưới {pct(rel_s.abs().quantile(0.95), 3)}, lớn nhất {pct(top.iloc[0])} "
        f"(các ngày lệch nhiều nhất: {', '.join(f'{d.date()} ({pct(v)})' for d, v in top.items())}). "
        "Hai nguồn chốt giá ở thời điểm khác nhau, nên ngày biến động mạnh thì lệch nhiều. "
        "Không có dấu hiệu sai đơn vị hay lệch ngày.", "",
        f"Hướng giá ngày (tăng/giảm) theo hai nguồn trùng nhau ở {pct(sign_agree)} số ngày từ 2018. "
        f"Trong {num((~agree).sum())} ngày hai nguồn ngược hướng, {pct(small_dis, 0)} là ngày giá Binance đi dưới "
        f"0,5% (so với {pct(small_all, 0)} trên mọi ngày): hướng của ngày gần như đứng yên phụ thuộc vào nguồn và "
        "giờ chốt giá. Đây là một mức nhiễu của nhãn mà không mô hình nào vượt qua được.", "")
    assert rel_s.abs().median() < 0.01 and sign_agree > 0.8

    # 1e. Cột tự tính
    netflow = (m.cm_FlowInExNtv - m.cm_FlowOutExNtv - m.cm_NetflowExNtv).abs().max()
    puell = m.cm_IssTotUSD / m.cm_IssTotUSD.rolling(365).mean()
    puell_err = ((puell - m.puell_multiple).abs() / m.puell_multiple).max()
    nvt = m.cm_CapMrktCurUSD / m.bc_est_tx_vol_usd
    nvt_err = ((nvt - m.nvt_ratio_bc).abs() / m.nvt_ratio_bc).max()
    assert netflow < 1e-6 and puell_err < 1e-9 and nvt_err < 1e-9, (netflow, puell_err, nvt_err)
    say("### 1.4 Cột tự tính", "",
        f"Tính lại từ cột gốc: `cm_NetflowExNtv` = FlowIn − FlowOut (lệch tối đa {netflow:.1e} BTC); "
        f"`puell_multiple` = IssTotUSD / trung bình 365 ngày (lệch tương đối {puell_err:.1e}); "
        f"`nvt_ratio_bc` = CapMrktCurUSD / bc_est_tx_vol_usd (lệch tương đối {nvt_err:.1e}).", "")


# ============================================================================
# 2. Nhãn (chỉ dùng giá)
# ============================================================================
def label_frame(m):
    c = m[["date", "ohlcv_close"]].dropna().set_index("date").ohlcv_close
    nxt = c.shift(-1)
    assert (nxt == c).sum() == 0, "có ngày giá đóng cửa bằng nhau"
    lab = pd.DataFrame({"close": c, "close_next": nxt,
                        "y": (nxt > c).astype(float).where(nxt.notna()),
                        "ret_next": np.log(nxt / c)})          # return mà nhãn ngày t mô tả (của ngày t+1)
    lab = lab[(lab.index >= STUDY_START) & lab.y.notna()]
    assert ((lab.ret_next > 0) == (lab.y == 1)).all()
    return lab


def ljung_box(x, lags):
    x = x - x.mean()
    n = len(x)
    rho = [np.sum(x[k:] * x[:-k]) / np.sum(x * x) for k in range(1, lags + 1)]
    q = n * (n + 2) * sum(r ** 2 / (n - k) for k, r in enumerate(rho, 1))
    return q, stats.chi2.sf(q, lags)


def eda_label(m):
    lab = label_frame(m)
    y, r = lab.y, lab.ret_next
    n = len(y)
    K.update(n_label=n, label_start=lab.index.min().date(), label_end=lab.index.max().date())
    say("## Phần 2. Nhãn: hướng giá ngày mai (chỉ dùng giá)", "",
        "y_t = 1 nếu close_{t+1} > close_t. Dòng ngày t mô tả return của ngày t+1. "
        f"Giai đoạn: {lab.index.min().date()} → {lab.index.max().date()} ({num(n)} ngày có nhãn; ngày cuối "
        f"{STUDY_END.date()} không có nhãn vì chưa biết giá hôm sau). Không có ngày nào giá đóng cửa bằng nhau.", "")

    ex = lab.loc[CRASH - pd.Timedelta(days=2): CRASH + pd.Timedelta(days=1), ["close", "close_next", "y"]]
    ex = ex.reset_index().assign(date=lambda d: d.date.dt.date, y=lambda d: d.y.astype(int))
    say("### 2.1 Ví dụ quanh ngày sập 12/03/2020", "",
        md(ex.rename(columns={"close": "close_t", "close_next": "close_t+1"}), floatfmt=".2f"), "",
        "Dòng 11/03 có y = 0 vì giá ngày 12/03 giảm. Mô hình bắt được cú sập phải dự báo giảm ở dòng **11/03**, "
        "chỉ với thông tin đến hết ngày 11/03.", "")

    # 2.2 Cân bằng nhãn theo năm
    yr = lab.groupby(lab.index.year).agg(n=("y", "size"), up=("y", "mean"),
                                         ret_mean=("ret_next", "mean"), ret_std=("ret_next", "std"))
    yr["p_binom"] = [stats.binomtest(int(round(a * b)), b, 0.5).pvalue for a, b in zip(yr.up, yr.n)]
    yr = yr.reset_index(names="năm")
    save_tab(yr, "label_by_year.csv")
    p_all = stats.binomtest(int(y.sum()), n, 0.5).pvalue
    K.update(up=y.mean(), p_up=p_all, up_min=yr.up.min(), up_min_yr=int(yr.loc[yr.up.idxmin(), "năm"]),
             up_max=yr.up.max(), up_max_yr=int(yr.loc[yr.up.idxmax(), "năm"]))
    say("### 2.2 Tỷ lệ ngày tăng", "",
        f"Cả giai đoạn: {pct(y.mean())} ngày tăng (p = {num(p_all, 2)} so với 50%). Nhãn gần cân bằng, nên DA 50% là "
        f"mốc đoán bừa và \"luôn đoán tăng\" chỉ được {pct(y.mean())}. Theo năm:", "",
        md(yr.rename(columns={"up": "tỷ lệ tăng", "ret_mean": "return TB", "ret_std": "độ lệch chuẩn return"})), "",
        f"Tỷ lệ tăng thay đổi từ {pct(K['up_min'])} ({K['up_min_yr']}) đến {pct(K['up_max'])} ({K['up_max_yr']}). "
        "Một mô hình chỉ học \"tỷ lệ tăng của 2 năm train\" sẽ đoán lệch khi chuyển năm, và con số gộp cả giai đoạn "
        "che mất điều này.", "")

    # 2.3 Biên độ: bao nhiêu ngày gần như đứng yên
    a = r.abs()
    q = pd.DataFrame({"ngưỡng |return|": ["< 0,1%", "< 0,25%", "< 0,5%", "< 1%"],
                      "tỷ lệ ngày": [(a < t).mean() for t in (0.001, 0.0025, 0.005, 0.01)]})
    K.update(small05=(a < 0.005).mean(), abs_median=a.median())
    say("### 2.3 Biên độ return: nhãn của nhiều ngày gần như là nhiễu", "", md(q), "",
        f"Trung vị |return| là {pct(a.median())}. {pct(K['small05'], 1)} số ngày đi dưới 0,5%: "
        "hướng của những ngày này phụ thuộc vào vài phút giao dịch quanh 00:00 UTC (xem 1.3: hai nguồn giá còn "
        "không đồng ý về hướng ở một phần số ngày). Đúng hay sai ở các ngày này gần như ngẫu nhiên, nên "
        "trần DA thực tế thấp hơn nhiều so với 100%.", "")

    # 2.4 Phụ thuộc theo thời gian (chỉ return quá khứ của chính giá)
    acf = pd.DataFrame({"độ trễ k": range(1, 11),
                        "ACF return": [r.autocorr(k) for k in range(1, 11)],
                        "ACF dấu return": [(2 * y - 1).autocorr(k) for k in range(1, 11)],
                        "ACF |return|": [a.autocorr(k) for k in range(1, 11)]})
    band = 1.96 / np.sqrt(n)
    save_tab(acf, "acf.csv")
    lb = ljung_box(r.to_numpy(), 10)
    same = (y == y.shift(1)).iloc[1:]
    p_pers = stats.binomtest(int(same.sum()), len(same), 0.5).pvalue
    outside = [int(k) for k, v in zip(acf["độ trễ k"], acf["ACF return"]) if abs(v) > band]
    K.update(acf1=acf.iloc[0, 1], acf1_sign=acf.iloc[0, 2], acf1_abs=acf.iloc[0, 3], pers=same.mean(),
             p_pers=p_pers, lb_p=lb[1], band=band)
    say("### 2.4 Hướng giá có lặp lại hay đảo chiều?", "",
        f"Tự tương quan của return ngày, dấu return và |return| (khoảng 95% khi không có phụ thuộc: ±{num(band, 3)}):", "",
        md(acf), "",
        f"- **Hướng giá có xu hướng đảo chiều sau 1 ngày:** ACF bậc 1 của return là {num(K['acf1'], 3)}, "
        f"của dấu là {num(K['acf1_sign'], 3)}. Hướng ngày mai trùng hướng hôm nay chỉ ở {pct(same.mean())} số ngày "
        f"(p {pval(p_pers)} so với 50%). Ljung–Box trên 10 bậc đầu của return: Q = {num(lb[0], 1)}, p = {pval(lb[1])}.",
        f"- Các bậc ACF của return nằm ngoài khoảng nhiễu: {', '.join(map(str, outside)) or 'không có'}.",
        f"- **|return| tự tương quan mạnh và kéo dài** (bậc 1: {num(K['acf1_abs'], 2)}): biến động tụ thành cụm. "
        "Dự báo được *độ lớn* biến động dễ hơn nhiều so với *hướng*.", "")

    grp = same.groupby(same.index.year)
    pers = pd.DataFrame({"năm": grp.size().index, "n": grp.size().to_numpy(), PERS: grp.mean().to_numpy()})
    pers["p_binom"] = [stats.binomtest(int(round(p * k)), k, 0.5).pvalue for p, k in zip(pers[PERS], pers.n)]
    pers["ACF1 return"] = [r[r.index.year == yv].autocorr(1) for yv in pers["năm"]]
    save_tab(pers, "persistence_by_year.csv")
    rev_years = pers.loc[pers[PERS] < 0.5, "năm"].tolist()
    K.update(n_rev=len(rev_years), n_years=len(pers), pers_min_yr=int(pers.loc[pers[PERS].idxmin(), "năm"]),
             pers_min=pers[PERS].min(), not_rev=[int(v) for v in pers.loc[pers[PERS] >= 0.5, "năm"]])
    say("Theo năm:", "", md(pers), "",
        f"Tỷ lệ cùng hướng dưới 50% (nghiêng về đảo chiều) ở {K['n_rev']}/{K['n_years']} năm, thấp nhất "
        f"{K['pers_min_yr']} ({pct(K['pers_min'])}); năm không đảo chiều: "
        f"{', '.join(map(str, K['not_rev'])) or 'không có'}. Đây là quy luật đơn giản nhất có trong dữ liệu, "
        "nên mọi mô hình cần được so với luật \"ngày mai ngược hôm nay\", không chỉ với 50%.", "")

    # 2.5 Ngày trong tuần
    wd = lab.groupby(lab.index.dayofweek).agg(n=("y", "size"), up=("y", "mean"),
                                              abs_ret=("ret_next", lambda s: s.abs().mean()))
    chi = stats.chi2_contingency(pd.crosstab(lab.index.dayofweek, y))
    wd.index = [WEEKDAYS[(i + 1) % 7] for i in wd.index]      # dòng ngày t mô tả ngày t+1
    wd = wd.loc[WEEKDAYS]
    weekend = wd.loc[["T7", "CN"], "abs_ret"].mean() / wd.loc[["T2", "T3", "T4", "T5", "T6"], "abs_ret"].mean()
    K.update(p_weekday=chi[1], weekend=weekend)
    say("### 2.5 Theo ngày trong tuần", "",
        "Dòng ngày t mô tả return của ngày t+1, nên bảng xếp theo **ngày được dự báo**:", "",
        md(wd.reset_index(names="ngày được dự báo").rename(columns={"up": "tỷ lệ tăng", "abs_ret": "|return| TB"})),
        "",
        f"Tỷ lệ tăng giữa các ngày trong tuần: χ² = {num(chi[0], 1)}, p = {num(chi[1], 2)}. "
        f"Biên độ T7 và CN chỉ bằng khoảng {pct(weekend, 0)} ngày thường, tức nhãn cuối tuần nhiễu hơn.", "")

    # 2.6 Phân phối return
    big = r.abs().nlargest(5)
    say("### 2.6 Phân phối return", "",
        f"Return ngày: trung bình {pct(r.mean(), 3)}, độ lệch chuẩn {pct(r.std())}, skew {num(stats.skew(r), 2)}, "
        f"kurtosis dư {num(stats.kurtosis(r), 1)} (phân phối chuẩn: 0). Đuôi dày: 5 ngày biến động lớn nhất là "
        + ", ".join(f"{(d + pd.Timedelta(days=1)).date()} ({pct(r[d], 1)})" for d in big.index) + ". "
        "DA và log-loss tính các ngày này như mọi ngày khác; backtest thì không, nên vài ngày cực đoan có thể "
        "quyết định kết quả backtest.", "")

    plot_label(lab, pers, acf, band)


def plot_label(lab, pers, acf, band):
    fig, ax = plt.subplots(3, 1, figsize=(10, 9))
    ax[0].semilogy(lab.index, lab.close, lw=0.8, color="#333")
    for hv in HALVINGS:
        ax[0].axvline(hv, color="#F58518", ls="--", lw=1)
    ax[0].set_title("Giá đóng cửa BTC/USDT (thang log); đường cam: halving")
    roll = lab.y.rolling(90).mean()
    ax[1].plot(roll.index, roll, color="#4C78A8", lw=1)
    ax[1].axhline(0.5, color="#999", lw=0.8)
    ax[1].set_title("Tỷ lệ ngày tăng, cửa sổ trượt 90 ngày")
    x = np.arange(1, len(acf) + 1)
    ax[2].bar(x - 0.2, acf["ACF return"], 0.4, label="return", color="#4C78A8")
    ax[2].bar(x + 0.2, acf["ACF dấu return"], 0.4, label="dấu return", color="#F58518")
    ax[2].axhspan(-band, band, color="#ddd")
    ax[2].set_xticks(x)
    ax[2].set_title("Tự tương quan theo độ trễ (vùng xám: khoảng 95% khi không có phụ thuộc)")
    ax[2].legend()
    say(f"![Nhãn]({save_fig(fig, 'eda_01_label.png')})", "")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    vals = pers[PERS]
    ax.bar(pers["năm"].astype(str), vals, color=["#E45756" if v < 0.5 else "#4C78A8" for v in vals])
    ax.axhline(0.5, color="#333", lw=0.8)
    ax.set_ylim(0.38, 0.58)
    ax.set_title("Tỷ lệ ngày mai cùng hướng hôm nay, theo năm (đỏ: nghiêng về đảo chiều)")
    say(f"![Persistence theo năm]({save_fig(fig, 'eda_02_persistence_by_year.png')})", "")


# ============================================================================
# 3. On-chain: chỉ đặc tính của chuỗi, không đặt cạnh nhãn
# ============================================================================
def eda_onchain(m, dicts):
    on_a, on_b = onchain_columns(dicts["A"]), onchain_columns(dicts["B"])
    groups = {**on_b, **on_a}
    cols = list(groups)
    s = m[(m.date >= STUDY_START) & (m.date <= STUDY_END)].set_index("date")[cols]
    say("## Phần 3. On-chain: đặc tính của từng chuỗi", "",
        "Phần này không đặt chỉ số On-chain cạnh nhãn (quy tắc chống data snooping của data freeze). "
        f"Gồm {len(cols)} cột của hai phương án Flows A và B; giai đoạn {STUDY_START.date()} → {STUDY_END.date()}.", "")

    signed = [c for c in cols if (s[c] <= 0).any()]
    trans = {c: (s[c].diff() if c in signed else np.log(s[c]).diff()) for c in cols}
    trend = {c: stats.spearmanr(np.arange(len(s)), s[c], nan_policy="omit")[0] for c in cols}
    desc = pd.DataFrame({
        "cột": cols, "nhóm": [groups[c] for c in cols],
        "A/B": ["A, B" if c in on_a and c in on_b else ("A" if c in on_a else "B") for c in cols],
        "trung vị": [s[c].median() for c in cols], "min": [s[c].min() for c in cols], "max": [s[c].max() for c in cols],
        "max/min": [s[c].max() / s[c].min() if c not in signed else np.nan for c in cols],
        "tỷ lệ ngày ≤ 0": [(s[c] <= 0).mean() for c in cols],
        "tương quan hạng với thời gian": [trend[c] for c in cols],
    })
    save_tab(desc, "onchain_describe.csv")
    trending = [c for c in cols if abs(trend[c]) > 0.7]
    K.update(trending=trending, n_onchain=len(cols), signed=signed)
    say("### 3.1 Mô tả", "", md(desc, floatfmt=".3g"), "",
        f"- **Mức (level) của {len(trending)}/{len(cols)} chỉ số có xu hướng mạnh theo thời gian** "
        f"(|tương quan hạng với thời gian| > 0,7): {', '.join(f'`{c}`' for c in trending)}. "
        "Đưa mức thô vào mô hình thì mô hình học \"đang ở năm nào\" thay vì tín hiệu, và giá trị ở giai đoạn test "
        "nằm ngoài khoảng của train. Cần biến đổi (log-diff, tỷ số MA, z-score trượt) trước khi dùng; mọi tham số "
        "biến đổi phải fit trong cửa sổ train.",
        f"- {', '.join(f'`{c}`' for c in signed)} có giá trị ≤ 0 nên không lấy log được; ở các phần sau dùng sai "
        "phân thường thay cho log-diff.", "")

    # 3.2 Bước nhảy: robust z của chuỗi đã biến đổi
    rows = []
    for c in cols:
        x = trans[c].dropna()
        mad = 1.4826 * (x - x.median()).abs().median()
        z = (x - x.median()) / mad
        rows.append({"cột": c, "số ngày |z| > 8": int((z.abs() > 8).sum()),
                     "ngày lớn nhất": z.abs().idxmax().date(), "|z| lớn nhất": float(z.abs().max())})
    jumps = pd.DataFrame(rows)
    save_tab(jumps, "onchain_jumps.csv")
    sply = trans["cm_SplyExNtv"].dropna()
    big = sply.abs().nlargest(6)
    K["sply_jumps"] = [(d.date(), np.exp(sply[d]) - 1) for d in big.index]
    say("### 3.2 Bước nhảy bất thường", "",
        "Robust z = (x − trung vị) / (1,4826 × MAD) trên chuỗi đã sai phân (log-diff, hoặc sai phân thường với cột "
        "có giá trị ≤ 0):", "", md(jumps, floatfmt=".1f"), "",
        "`cm_SplyExNtv` (tổng BTC trên sàn) là chuỗi tồn kho nên lẽ ra thay đổi từ từ, nhưng có các ngày nhảy: "
        + ", ".join(f"{d} ({pct(v)})" for d, v in K["sply_jumps"]) + ". "
        "Bước nhảy cỡ vài % lượng BTC trên sàn trong một ngày thường là do CoinMetrics thêm/bớt ví của một sàn "
        "vào danh sách gắn nhãn, không phải dòng tiền thật. Dictionary ghi Flows là `flash`: lịch sử có thể bị "
        "sửa khi danh sách ví cập nhật. Đây là lý do cần chạy đối chứng bỏ Flows (góp ý 5).", "")

    # 3.3 Mùa vụ tuần (trên chuỗi đã biến đổi)
    pos = [c for c in cols if c not in signed]
    wk = pd.DataFrame({c: trans[c].groupby(trans[c].index.dayofweek).mean() for c in pos}).T
    wk.columns = WEEKDAYS
    kw = {c: stats.kruskal(*[g.dropna() for _, g in trans[c].groupby(trans[c].index.dayofweek)]).pvalue for c in pos}
    wk["p (Kruskal)"] = pd.Series(kw)
    strong = [c for c, p in kw.items() if p < 1e-6]
    K["weekly"] = strong
    say("### 3.3 Mùa vụ theo ngày trong tuần", "",
        "Trung bình log-diff theo ngày trong tuần (ví dụ cột T2 = thay đổi từ CN sang T2):", "",
        md(wk.reset_index(names="cột"), floatfmt=".3f"), "",
        f"Mùa vụ tuần rất rõ (p < 1e-6) ở {len(strong)} chỉ số: {', '.join(f'`{c}`' for c in strong)}. "
        "Log-diff 1 ngày của các chỉ số này vì vậy chủ yếu đo \"hôm nay là thứ mấy\". "
        "Biến đổi trên cửa sổ 7 ngày (MA7, thay đổi so với 7 ngày trước) khử được phần này.", "")

    # 3.4 Tương quan giữa các chỉ số On-chain (không có nhãn)
    tdf = pd.DataFrame(trans).dropna()
    corr = tdf.corr(method="spearman")
    pairs = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().sort_values(key=abs, ascending=False)
    top = pairs.head(8).reset_index()
    top.columns = ["cột 1", "cột 2", "Spearman"]
    save_tab(corr.reset_index(names="cột"), "onchain_corr.csv")
    K["top_pair"] = (top.iloc[0, 0], top.iloc[0, 1], top.iloc[0, 2])
    say("### 3.4 Tương quan giữa các chỉ số On-chain (trên chuỗi đã sai phân)", "", md(top, floatfmt=".2f"), "",
        "Cặp tương quan cao mang thông tin trùng nhau: hệ số của mô hình tuyến tính không ổn định, và ablation theo "
        "nhóm khó tách đóng góp. Netflow = In − Out nên ba cột này không được đưa vào cùng lúc (lý do có hai phương "
        "án Flows A và B).", "")
    plot_onchain(s, cols, groups, corr)


def plot_onchain(s, cols, groups, corr):
    n = len(cols)
    nrow = (n + 2) // 3
    fig, axes = plt.subplots(nrow, 3, figsize=(12, 2.2 * nrow))
    for ax, c in zip(axes.flat, cols):
        x = s[c]
        ax.plot(x.index, x, lw=0.6, color="#4C78A8")
        if (x.dropna() > 0).all():
            ax.set_yscale("log")
        for hv in HALVINGS:
            ax.axvline(hv, color="#F58518", ls="--", lw=0.8)
        ax.set_title(f"{c} ({groups[c]})", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    say(f"![On-chain theo thời gian]({save_fig(fig, 'eda_03_onchain_series.png')})", "")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    fig.colorbar(im)
    ax.set_title("Spearman giữa các chỉ số On-chain đã sai phân")
    say(f"![Tương quan On-chain]({save_fig(fig, 'eda_04_onchain_corr.png')})", "")


# ============================================================================
# Tóm tắt và hệ quả: sinh từ K để số liệu luôn khớp phần chi tiết
# ============================================================================
def summary():
    rows = [
        ("Dữ liệu có đúng không?",
         f"Đúng: lịch ngày đủ; giá ngày khớp nến 1h trên {num(K['n_days_1h'])} ngày; cột tự tính khớp công thức. "
         f"Ô trống từ 2018 chỉ ở {', '.join(f'`{c}`' for c in K['gaps'])}. "
         f"{K['n_short_study']} ngày thiếu nến trong giai đoạn nghiên cứu; close không phải nến 23:00 ở "
         f"{', '.join(K['close_not_23'])}"),
        ("Giá có khớp nguồn thứ hai?",
         f"Có: lệch trung vị {pct(K['px_rel_median'], 2)} so với CoinMetrics; cùng hướng ở "
         f"{pct(K['px_sign_agree'], 1)} số ngày"),
        ("Nhãn có cân bằng?",
         f"Có: {pct(K['up'])} ngày tăng ({num(K['n_label'])} ngày), nhưng theo năm từ {pct(K['up_min'], 1)} "
         f"({K['up_min_yr']}) đến {pct(K['up_max'], 1)} ({K['up_max_yr']})"),
        ("Nhãn có nhiễu?",
         f"Nhiều: {pct(K['small05'], 0)} số ngày đi dưới 0,5%; trung vị |return| {pct(K['abs_median'])}"),
        ("Có quy luật đơn giản nào?",
         f"Đảo chiều sau 1 ngày: ngày mai cùng hướng hôm nay chỉ {pct(K['pers'], 1)} (p {pval(K['p_pers'])}); "
         f"ACF1 = {num(K['acf1'], 3)}. Không ổn định: năm không đảo chiều {', '.join(map(str, K['not_rev'])) or '—'}"),
        ("On-chain cần lưu ý gì?",
         f"{len(K['trending'])}/{K['n_onchain']} chỉ số có xu hướng mạnh ở mức thô; "
         f"{len(K['weekly'])} chỉ số có mùa vụ tuần; `cm_SplyExNtv` có bước nhảy tới "
         f"{pct(max(abs(v) for _, v in K['sply_jumps']), 1)}/ngày (nghi do đổi nhãn ví sàn)"),
    ]
    head = ["## Tóm tắt", "", md(pd.DataFrame(rows, columns=["Câu hỏi", "Kết quả"])), "",
            "## Hệ quả cho bước baseline", "",
            f"1. **Mốc so sánh:** DA 50%; luôn đoán tăng {pct(K['up'])}; luật \"ngày mai ngược hôm nay\" "
            f"(đúng {pct(1 - K['pers'])} trên cả giai đoạn, chưa tính chọn chiều trong train). Một mô hình chưa vượt "
            "luật đảo chiều thì chưa học được gì hơn quy luật đơn giản nhất.",
            "2. **Báo cáo theo năm**, không chỉ con số gộp: tỷ lệ tăng và mức đảo chiều đổi mạnh giữa các năm.",
            f"3. **Sai số của DA** với khoảng {num(K['n_label'])} ngày là khoảng "
            f"{pct(np.sqrt(0.25 / K['n_label']), 1)} (một độ lệch chuẩn), nên chênh lệch dưới ~2 điểm % giữa hai "
            "mô hình cần kiểm định cặp (McNemar) trước khi kết luận.",
            "4. **On-chain:** không dùng mức thô; biến đổi trên cửa sổ 7 ngày để khử mùa vụ tuần; fit mọi biến đổi "
            "trong train; chỉ dùng On-chain đến ngày t−1 (độ trễ công bố).",
            "5. **Flows:** chạy đối chứng bỏ Flows vì `cm_SplyExNtv` có bước nhảy do đổi nhãn ví sàn (góp ý 5).",
            ""]
    i = report.index("<!-- TOM_TAT -->")
    report[i:i + 1] = head


# ============================================================================
def main():
    man, m, h, dicts = load()
    say("# Kiểm tra dữ liệu và EDA (góp ý 1 của thầy)", "",
        "Tạo bởi `eda/src/eda.py`. **Chỉ đọc `data/processed/`** (data freeze, sha256 khớp `MANIFEST.json`); "
        "không dùng kết quả hay code của thư mục nào khác. Mọi kiểm tra là `assert`, báo cáo chỉ được ghi khi "
        "tất cả đều đúng. Theo quy tắc của data freeze, không có phép tính nào giữa chỉ số On-chain và nhãn.", "",
        "<!-- TOM_TAT -->")
    check_data(man, m, h, dicts)
    eda_label(m)
    eda_onchain(m, dicts)
    summary()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eda.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Đã ghi {OUT / 'eda.md'}")


if __name__ == "__main__":
    main()
