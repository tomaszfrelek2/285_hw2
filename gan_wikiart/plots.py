"""Figures for the report."""
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.utils as vutils

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"


def plot_training(logs, path):
    """Losses, discriminator outputs, FID and IS in one row."""
    it, ev = pd.DataFrame(logs["iters"]), pd.DataFrame(logs["evals"])
    x = it["step"] / (it["step"].max() + 1) * it["epoch"].max()  # x axis in epochs
    smooth = lambda s: s.rolling(64, min_periods=1).mean()
    fig, ax = plt.subplots(1, 4, figsize=(14, 3))
    for col, c, label in [("loss_D", BLUE, "D"), ("loss_G", ORANGE, "G")]:
        ax[0].plot(x, it[col], color=c, alpha=0.15, lw=0.4)
        ax[0].plot(x, smooth(it[col]), color=c, label=label)
    ax[0].set_ylim(0, 10)
    for col, c, label in [("D_x", BLUE, "D(x)"), ("D_G_z1", ORANGE, "D(G(z)) before G step"),
                          ("D_G_z2", GREEN, "D(G(z)) after G step")]:
        ax[1].plot(x, smooth(it[col]), color=c, label=label)
    ax[1].set_ylim(0, 1)
    ax[2].plot(ev["epoch"], ev["fid"], "o-", color=BLUE)
    ax[3].plot(ev["epoch"], ev["inception_score"], "o-", color=ORANGE)
    for a, title in zip(ax, ["(a) losses", "(b) discriminator outputs", "(c) FID ↓", "(d) IS ↑"]):
        a.set_title(title); a.set_xlabel("epoch"); a.grid(alpha=0.3)
    ax[0].legend(); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.show()


def save_grid(images, path, nrow=8):
    """images in [-1, 1]."""
    vutils.save_image((images + 1) / 2, path, nrow=nrow, padding=2, pad_value=1.0)


@torch.no_grad()
def nearest_neighbors(samples, X, path, k=5):
    """Each row: a generated sample, then its k nearest training images (pixel L2)."""
    q = ((samples + 1) * 127.5).flatten(1)
    best_d = torch.full((len(q), k), float("inf"), device=X.device)
    best_i = torch.zeros((len(q), k), dtype=torch.long, device=X.device)
    for s in range(0, len(X), 8192):  # chunked so it fits in memory
        x = X[s:s + 8192].float().flatten(1)
        d = torch.cat([best_d, torch.cdist(q, x)], 1)
        i = torch.cat([best_i, torch.arange(s, s + len(x), device=X.device).expand(len(q), -1)], 1)
        best_d, j = d.topk(k, dim=1, largest=False)
        best_i = i.gather(1, j)
    rows = [torch.cat([samples[r:r + 1], X[best_i[r]].float() / 127.5 - 1]) for r in range(len(q))]
    save_grid(torch.cat(rows), path, nrow=k + 1)
