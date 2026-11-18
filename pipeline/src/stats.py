"""
stats.py  –  Kiểm định cho bảng Δ (Vi, task 2.3; dùng cho E2, E3 và góp ý 3–6 của thầy).

  bootstrap_delta   Δ = M(B) − M(A) cùng ngày test, CI 95% bằng circular block bootstrap (khối 15 ngày)
  mcnemar           McNemar chính xác cho ΔDA: chỉ đếm ngày hai bên bất đồng
  diebold_mariano   DM cho Δ log-loss, phương sai dài hạn Newey–West
  holm              hiệu chỉnh Holm khi kiểm nhiều phép cùng lúc
  shuffle_runs      xáo thứ tự các đoạn regime, giữ độ dài (placebo cho E3)

python stats.py  -> chạy test
"""
import numpy as np
from scipy import stats as sps

BLOCK, N_BOOT, SEED = 15, 2000, 0
EPS = 1e-6


def row_log_loss(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def row_hit(y, p):
    return ((np.asarray(p, float) > 0.5).astype(int) == np.asarray(y)).astype(float)


ROW_METRIC = {"da": row_hit, "log_loss": row_log_loss}


def block_indices(T, block=BLOCK, n_boot=N_BOOT, seed=SEED):
    """n_boot dãy chỉ số dài T, ghép từ các khối liên tiếp dài `block` (vòng tròn)."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, T, size=(n_boot, int(np.ceil(T / block))))
    return ((starts[:, :, None] + np.arange(block)) % T).reshape(n_boot, -1)[:, :T]


def bootstrap_delta(y, pa, pb, metric, idx=None):
    """(Δ, lo, hi) với Δ = M(pb) − M(pa), metric cộng được theo ngày (da, log_loss)."""
    f = ROW_METRIC[metric]
    d = f(y, pb) - f(y, pa)
    idx = block_indices(len(d)) if idx is None else idx
    boots = d[idx].mean(1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(d.mean()), float(lo), float(hi)


def mcnemar(y, pa, pb):
    """(chỉ A đúng, chỉ B đúng, p) – kiểm định nhị thức chính xác trên các ngày bất đồng."""
    ha, hb = row_hit(y, pa).astype(bool), row_hit(y, pb).astype(bool)
    n_a, n_b = int((ha & ~hb).sum()), int((~ha & hb).sum())
    p = sps.binomtest(n_b, n_a + n_b, 0.5).pvalue if n_a + n_b else 1.0
    return n_a, n_b, float(p)


def diebold_mariano(loss_a, loss_b, lag=None):
    """(stat, p) với d = loss_a − loss_b; stat > 0 nghĩa là B có loss thấp hơn."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    T = len(d)
    lag = int(np.floor(4 * (T / 100) ** (2 / 9))) if lag is None else lag
    dc = d - d.mean()
    lrv = dc @ dc / T + 2 * sum((1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / T for k in range(1, lag + 1))
    if lrv <= 1e-18:
        return np.nan, 1.0
    s = d.mean() / np.sqrt(lrv / T)
    return float(s), float(2 * sps.norm.sf(abs(s)))


def holm(pvals):
    """p đã hiệu chỉnh Holm; NaN giữ nguyên."""
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    run = 0.0
    for i, j in enumerate(ok[np.argsort(p[ok])]):
        run = max(run, min(1.0, (len(ok) - i) * p[j]))
        out[j] = run
    return out


def paired_row(y, pa, pb, with_logloss=True):
    """Một dòng bảng Δ: ΔDA + CI + McNemar; Δ log-loss + CI + DM (nếu with_logloss)."""
    idx = block_indices(len(y))
    r = dict(zip(("d_da", "d_da_lo", "d_da_hi"), bootstrap_delta(y, pa, pb, "da", idx)))
    r["n_only_a"], r["n_only_b"], r["p_mcnemar"] = mcnemar(y, pa, pb)
    if with_logloss:
        r |= dict(zip(("d_logloss", "d_logloss_lo", "d_logloss_hi"), bootstrap_delta(y, pa, pb, "log_loss", idx)))
        r["dm_stat"], r["p_dm"] = diebold_mariano(row_log_loss(y, pa), row_log_loss(y, pb))
    return r


def run_lengths(labels):
    """(nhãn của từng đoạn, độ dài từng đoạn) cho một dãy nhãn liên tiếp."""
    labels = np.asarray(labels)
    cut = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, cut]
    return labels[starts], np.diff(np.r_[starts, len(labels)])


def shuffle_runs(labels, rng):
    """Xáo thứ tự các đoạn, giữ nguyên độ dài từng đoạn."""
    lab, length = run_lengths(labels)
    order = rng.permutation(len(lab))
    return np.repeat(lab[order], length[order])


# ----------------------------------------------------------------------------
# Test:  python stats.py --test
# ----------------------------------------------------------------------------
def _tests():
    rng = np.random.default_rng(0)
    T = 600
    y = rng.integers(0, 2, T)
    pa = np.clip(0.5 + rng.normal(0, 0.05, T), 0.01, 0.99)

    # 1) Hai dự báo giống nhau: Δ = 0, CI = [0, 0], McNemar p = 1, DM không xác định
    assert bootstrap_delta(y, pa, pa, "da") == (0.0, 0.0, 0.0)
    assert mcnemar(y, pa, pa) == (0, 0, 1.0)
    assert np.isnan(diebold_mariano(np.ones(50), np.ones(50))[0])

    # 2) B tốt hơn rõ: ΔDA > 0, CI không chứa 0, McNemar và DM có ý nghĩa, DM > 0
    pb = np.clip(pa + 0.3 * (y - 0.5), 0.01, 0.99)
    r = paired_row(y, pa, pb)
    assert r["d_da"] > 0 and r["d_da_lo"] > 0 and r["p_mcnemar"] < 0.01
    assert r["d_logloss"] < 0 and r["dm_stat"] > 0 and r["p_dm"] < 0.01
    #    ΔDA = (chỉ B đúng − chỉ A đúng) / T
    assert np.isclose(r["d_da"], (r["n_only_b"] - r["n_only_a"]) / T)

    # 3) Holm đúng với ví dụ tay; giữ NaN
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03, np.nan])[:3], [0.03, 0.06, 0.06])
    assert np.isnan(holm([0.01, np.nan])[1])

    # 4) Block bootstrap: đúng kích thước, chỉ số hợp lệ, khối liên tiếp
    idx = block_indices(37, 15, 5, 0)
    assert idx.shape == (5, 37) and idx.min() >= 0 and idx.max() < 37
    assert ((np.diff(idx[0, :15]) % 37) == 1).all()

    # 5) Xáo đoạn giữ số ngày mỗi nhãn và độ dài các đoạn
    reg = np.repeat(np.array(["Bull", "Bear", "Sideways", "Bull"]), [40, 20, 30, 10])
    sh = shuffle_runs(reg, rng)
    assert sorted(sh) == sorted(reg)
    assert sorted(run_lengths(sh)[1]) == sorted(run_lengths(reg)[1]) or len(run_lengths(sh)[1]) < 4
    print("Tất cả test stats đều PASS")


if __name__ == "__main__":
    _tests()
