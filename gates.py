"""Gate validasi. Input: gross R per trade (urut waktu) dan biaya per trade dalam R."""
import numpy as np

CH = 2000


def perm_p(g, c, n_perm=20000, seed=1):
    g, c = np.asarray(g, float), np.asarray(c, float)
    obs, rng, hit = (g - c).mean(), np.random.default_rng(seed), 0
    for _ in range(n_perm // CH):
        s = rng.integers(0, 2, (CH, len(g)), dtype=np.int8) * 2 - 1
        hit += int(((s * g - c).mean(axis=1) >= obs).sum())
    return (hit + 1) / (n_perm // CH * CH + 1)


def run_gates(g, c, n_tried, alpha=0.01, n_perm=20000, folds=4, is_frac=0.6):
    g, c = np.asarray(g, float), np.asarray(c, float)
    a, n = alpha / n_tried, len(g)
    if a < 1 / (n_perm + 1):
        raise ValueError("n_perm terlalu kecil untuk n_tried ini, naikkan n_perm")
    k = int(n * is_frac)
    p_is, p_oos = perm_p(g[:k], c[:k], n_perm), perm_p(g[k:], c[k:], n_perm)
    pos = sum((g[i] - c[i]).sum() > 0 for i in np.array_split(np.arange(n), folds))
    r = {"trade": n, "alpha_adj": a, "p_IS": round(p_is, 5), "p_OOS": round(p_oos, 5),
         "fold_pos": f"{pos}/{folds}", "G1_IS": p_is < a, "G2_OOS": p_oos < a, "G3_folds": pos >= folds - 1}
    r["LOLOS"] = r["G1_IS"] and r["G2_OOS"] and r["G3_folds"]
    return r


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    c = np.full(3000, 0.05)
    print("tanpa edge (harus GUGUR):", run_gates(rng.normal(0, 1, 3000), c, 1))
    print("ada edge (harus LOLOS):", run_gates(rng.normal(0.15, 1, 3000), c, 1))
