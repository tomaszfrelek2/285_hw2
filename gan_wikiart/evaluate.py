"""FID and Inception Score with PyTorch-Ignite, following
https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/"""
import torch
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore
from PIL import Image
from torchvision import transforms

_metrics = None  # (FID, IS), created once so the Inception weights load only once


def interpolate(batch):
    # [-1, 1] -> [0, 1] first (the tutorial skips this, so negative values wrap around), then PIL resize to 299x299
    arr = []
    for img in ((batch + 1) / 2).clamp(0, 1).cpu():
        pil_img = transforms.ToPILImage()(img)
        arr.append(transforms.ToTensor()(pil_img.resize((299, 299), Image.BILINEAR)))
    return torch.stack(arr).to(batch.device)


def fid_is(fake_fn, X, real_idx, batch_size=200):
    """FID/IS of images from fake_fn(n) (range [-1, 1]) against the real images X[real_idx]. Needs > 2048 images."""
    global _metrics
    if _metrics is None:
        _metrics = (FID(device=X.device), InceptionScore(device=X.device, output_transform=lambda x: x[0]))

    def evaluation_step(engine, idx):
        with torch.no_grad():
            real = X[idx].float() / 127.5 - 1
            return interpolate(fake_fn(len(idx))), interpolate(real)

    evaluator = Engine(evaluation_step)
    _metrics[0].attach(evaluator, "fid")
    _metrics[1].attach(evaluator, "is")
    evaluator.run([real_idx[i:i + batch_size] for i in range(0, len(real_idx), batch_size)], max_epochs=1)
    return evaluator.state.metrics["fid"], evaluator.state.metrics["is"]


def evaluate_generator(G, X, real_idx, nz=100, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    G.eval()
    result = fid_is(lambda n: G(torch.randn(n, nz, 1, 1, device=X.device, generator=g)), X, real_idx)
    G.train()
    return result


def evaluate_real(X, idx_a, idx_b, batch_size=200):
    """Reference values: FID between two disjoint real sets, and IS of real images."""
    batches = iter([idx_b[i:i + batch_size] for i in range(0, len(idx_b), batch_size)])
    return fid_is(lambda n: X[next(batches)].float() / 127.5 - 1, X, idx_a, batch_size)
