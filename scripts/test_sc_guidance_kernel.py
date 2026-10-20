"""Gradient unit test for the self-conditioning guidance shift in the chunked head+CE kernel.

With logits = (h - s) W^T + b + singular + mask and the shift s a
stop-gradient target term, the kernel's grads must equal plain autograd of

    CE(F.linear(h, W, b) + singular + mask - s @ W.detach().T)

and must NOT equal the variant where W inside the shift carries gradient. Also checks the no-shift
path is unchanged, that padded head columns get exactly zero gradient, and the no-grad loss value.

Run from the repo root (one GPU): python scripts/test_sc_guidance_kernel.py
"""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from relf.fused_ce import chunked_linear_cross_entropy  # noqa: E402

torch.manual_seed(0)
dev = 'cuda'
n_rows, hidden, V, V_real = 96, 24, 40, 37
chunk_rows = 40  # 3 chunks, last one partial


def make():
    h = (torch.randn(n_rows, hidden, device=dev) * 0.7).bfloat16().requires_grad_(True)
    W = (torch.randn(V, hidden, device=dev) * 0.3)
    W[V_real:] = 0.0  # Padded head rows are exactly zero forever
    W = W.bfloat16().requires_grad_(True)
    b = (torch.randn(V, device=dev) * 0.1)
    b[V_real:] = 0.0
    b = b.bfloat16().requires_grad_(True)
    y = torch.randint(0, V_real, (n_rows,), device=dev)
    rw = (torch.rand(n_rows, device=dev) > 0.2).float()
    s = (torch.randn(n_rows, hidden, device=dev) * 0.5)
    s[::3] = 0.0  # Keep-masked rows carry no shift
    mask = torch.zeros(V, device=dev)
    mask[V_real:] = float('-inf')
    # Singular descriptor: (n_rows, d+1) inputs @ (d+1, V) embed, constant fp32 operands
    d = 5
    sing_in = torch.randn(n_rows, d + 1, device=dev)
    sing_emb = torch.randn(d + 1, V, device=dev)
    return h, W, b, y, rw, s, mask, (sing_in, sing_emb)


def reference(h, W, b, y, rw, s, mask, sing, detach_shift_W):
    logits = F.linear(h.float(), W.float(), b.float()) + sing[0] @ sing[1] + mask
    Ws = W.detach().float() if detach_shift_W else W.float()
    logits = logits - s @ Ws.t()
    ce = F.cross_entropy(logits, y, reduction='none')
    return (ce * rw).sum()


def grads(loss, *params):
    return torch.autograd.grad(loss, params)


def close(a, b, tol):
    return (a.float() - b.float()).abs().max().item() <= tol * (1.0 + b.float().abs().max().item())


h, W, b, y, rw, s, mask, sing = make()
tf32 = torch.backends.cuda.matmul.allow_tf32
torch.backends.cuda.matmul.allow_tf32 = False

# 1. Kernel with shift vs stop-grad-W reference
loss_k = chunked_linear_cross_entropy(h, W, b, y, rw, chunk_rows=chunk_rows, singular=sing, col_mask=mask,
                                      hidden_shift=s)
gk = grads(loss_k, h, W, b)
loss_r = reference(h, W, b, y, rw, s, mask, sing, detach_shift_W=True)
gr = grads(loss_r, h, W, b)
assert close(loss_k, loss_r, 2e-2), (loss_k.item(), loss_r.item())
for name, a, r in zip(('h', 'W', 'b'), gk, gr):
    assert close(a, r, 4e-2), (name, (a.float() - r.float()).abs().max().item())
assert gk[1][V_real:].abs().max().item() == 0.0, 'padded head rows must get exactly zero gradient'
assert gk[2][V_real:].abs().max().item() == 0.0

# 2. The wrong (non-detached shift W) variant must NOT match the kernel's W-gradient
loss_w = reference(h, W, b, y, rw, s, mask, sing, detach_shift_W=False)
gw = grads(loss_w, W)[0]
assert not close(gk[1], gw, 4e-2), 'kernel W-grad matched the non-stop-grad variant: shift leaked into the model'

# 3. No-shift path unchanged (`hidden_shift=None` == zeros shift == old kernel)
loss_0 = chunked_linear_cross_entropy(h, W, b, y, rw, chunk_rows=chunk_rows, singular=sing, col_mask=mask)
loss_z = chunked_linear_cross_entropy(h, W, b, y, rw, chunk_rows=chunk_rows, singular=sing, col_mask=mask,
                                      hidden_shift=torch.zeros_like(s))
assert torch.equal(loss_0, loss_z)
g0, gz = grads(loss_0, h, W, b), grads(loss_z, h, W, b)
assert all(torch.equal(a, c) for a, c in zip(g0, gz))

# 4. No-grad path takes the shift too
with torch.no_grad():
    loss_ng = chunked_linear_cross_entropy(h, W, b, y, rw, chunk_rows=chunk_rows, singular=sing, col_mask=mask,
                                           hidden_shift=s)
assert close(loss_ng, loss_k, 1e-4), (loss_ng.item(), loss_k.item())

torch.backends.cuda.matmul.allow_tf32 = tf32
print('OK: sc_guidance kernel gradients match the stop-grad-shift reference; padded rows zero; no-shift path unchanged')
