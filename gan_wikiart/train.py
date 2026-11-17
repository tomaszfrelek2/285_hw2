"""DCGAN training loop. Checkpoints every epoch and resumes automatically from out_dir/checkpoint.pt."""
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchvision.utils as vutils

from evaluate import evaluate_generator


def train(G, D, X, out_dir, eval_idx, epochs=50, batch_size=128, lr=2e-4, beta1=0.5, nz=100,
          real_label=0.9, eval_every=5):
    out = Path(out_dir)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    dev = X.device
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
    bce = nn.BCEWithLogitsLoss()
    fixed_noise = torch.randn(64, nz, 1, 1, device=dev, generator=torch.Generator(device=dev).manual_seed(0))
    logs = {"iters": [], "evals": [], "epochs": []}
    start = 1

    ckpt = out / "checkpoint.pt"
    if ckpt.exists():
        ck = torch.load(ckpt, map_location=dev)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"])
        optG.load_state_dict(ck["optG"]); optD.load_state_dict(ck["optD"])
        logs, start = ck["logs"], ck["epoch"] + 1
        print(f"resuming after epoch {ck['epoch']}")

    n_iters = len(X) // batch_size
    for epoch in range(start, epochs + 1):
        t0 = time.time()
        perm = torch.randperm(len(X), device=dev)
        for i in range(n_iters):
            real = X[perm[i * batch_size:(i + 1) * batch_size]].float() / 127.5 - 1
            flip = torch.rand(batch_size, 1, 1, 1, device=dev) < 0.5
            real = torch.where(flip, real.flip(3), real)  # random horizontal flip
            fake = G(torch.randn(batch_size, nz, 1, 1, device=dev))

            # D step: real -> real_label (one-sided label smoothing), fake -> 0
            d_real, d_fake = D(real), D(fake.detach())
            loss_D = bce(d_real, torch.full_like(d_real, real_label)) + bce(d_fake, torch.zeros_like(d_fake))
            optD.zero_grad(); loss_D.backward(); optD.step()

            # G step: non-saturating loss
            d_gen = D(fake)
            loss_G = bce(d_gen, torch.ones_like(d_gen))
            optG.zero_grad(); loss_G.backward(); optG.step()

            logs["iters"].append(dict(epoch=epoch, step=(epoch - 1) * n_iters + i, loss_D=loss_D.item(),
                                      loss_G=loss_G.item(), D_x=d_real.sigmoid().mean().item(),
                                      D_G_z1=d_fake.sigmoid().mean().item(), D_G_z2=d_gen.sigmoid().mean().item()))
        logs["epochs"].append(dict(epoch=epoch, seconds=time.time() - t0))
        r = logs["iters"][-1]
        print(f"epoch {epoch}: loss_D {r['loss_D']:.3f} loss_G {r['loss_G']:.3f} ({time.time() - t0:.0f}s)")

        G.eval()
        with torch.no_grad():
            vutils.save_image((G(fixed_noise) + 1) / 2, out / "samples" / f"epoch_{epoch:03d}.png", nrow=8)
        G.train()

        if epoch % eval_every == 0 or epoch == epochs:
            fid, is_ = evaluate_generator(G, X, eval_idx, nz)
            logs["evals"].append(dict(epoch=epoch, fid=fid, inception_score=is_))
            print(f"  FID {fid:.4f} | IS {is_:.3f}")

        torch.save(dict(G=G.state_dict(), D=D.state_dict(), optG=optG.state_dict(), optD=optD.state_dict(),
                        epoch=epoch, logs=logs), ckpt)
        for k in logs:
            pd.DataFrame(logs[k]).to_csv(out / f"log_{k}.csv", index=False)
    return logs
