"""Row-chunked fused LM head + cross-entropy with the analytic singular logits (fp32 logits per chunk)."""

import torch
import torch.nn.functional as F
from quack.cross_entropy import cross_entropy as fused_cross_entropy

# Rows per chunk of the fused head+CE. 131072 rows (B=128, L=1024) -> 4 chunks;
# each materializes only a (32768, V) logits block (~3.3 GB bf16) instead of the
# full (B*L, V) logits tensor. Fewer chunks also mean fewer read-modify-write
# passes over the fp32 grad_weight accumulator.
CE_CHUNK_ROWS = 32768


def _add_singular_logits(logits, singular, s, e):
    """Add the analytic singular logits of rows `s:e` to the fp32 chunk `logits` in place.

    `singular` is the descriptor `(xs_aug, embed_aug)` built by `ReLF._singular_desc`:
    `logits += xs_aug[s:e] @ embed_aug` with `xs_aug` (n_rows, d+1) = [a * xt, b] and
    `embed_aug` (d+1, V) = [E; ||E||^2]^T, both fp32 and constant w.r.t. parameters.
    """
    logits.addmm_(singular[0][s:e], singular[1])


def _chunked_linear_ce(
    hidden, w, b, targets, row_weight, chunk_rows, need_grads, singular=None, col_mask=None,
    hidden_shift=None,
):
    """Run the fused head+CE over row chunks; per-chunk logits are built, used, and freed.

    `hidden_shift`, if given, is a stop-gradient `(n_rows, hidden)` fp32 tensor subtracted from the
    features BEFORE the head (self-conditioning guidance distillation): the
    CE is taken on `(hidden - shift) W^T + b`, i.e. on the network logits minus a constant logit
    shift `shift W^T`. The gradients are those of the unshifted network --
    `grad_hidden = dlogits W`, `grad_weight = dlogits^T hidden` -- exactly as if the shift had been
    added to the target (ELF's `stop_gradient(v_target + sc_guidance)`), never as if it were part of
    the model.

    When `need_grads` is set, input/weight/bias grads are computed here (Liger-style) so the
    backward never touches logits again; the quack CE kernel provides dlogits via its inplace
    backward.

    `singular`, if given, is a descriptor for `_add_singular_logits` (constant fp32 operands): the
    analytic singular logits are added to each chunk's logits in fp32, since their magnitude
    diverges as t -> 1. dlogits w.r.t. the network logits equals dlogits w.r.t. the total logits, so
    the grad formulas are unchanged.

    `col_mask`, if given, is a `(V,)` additive logit mask (0 on real vocab columns, -inf on
    GEMM-padding columns) applied to every chunk's logits before the CE: exp(-inf) = 0 keeps the
    logsumexp identical to the unpadded head, and the padded columns' dlogits (softmax - onehot) are
    exactly zero, so padded head rows receive exactly zero gradient. Targets are always real tokens,
    so no row is ever fully masked.

    Returns `(loss_sum, grad_hidden, grad_weight, grad_bias)` where `loss_sum` is the
    `row_weight`-weighted sum of per-token CE (fp32).
    """
    n_rows = hidden.shape[0]
    loss_sum = hidden.new_zeros((), dtype=torch.float32)
    grad_hidden = torch.empty_like(hidden) if need_grads else None
    grad_weight = torch.zeros_like(w, dtype=torch.float32) if need_grads else None
    grad_bias = torch.zeros_like(b, dtype=torch.float32) if need_grads and b is not None else None
    # Detach: grads are computed manually below; the per-chunk enable_grad graph
    # must start at the chunk logits, not reach hidden/w/b
    hidden = hidden.detach()
    w16 = w.detach().to(hidden.dtype)
    b16 = b.detach().to(hidden.dtype) if b is not None else None
    cm16 = col_mask.to(hidden.dtype) if col_mask is not None else None
    hs = hidden_shift.detach().float() if hidden_shift is not None else None
    for s in range(0, n_rows, chunk_rows):
        e = min(s + chunk_rows, n_rows)
        h, y, rw = hidden[s:e], targets[s:e], row_weight[s:e]
        # h_in feeds the head; h (unshifted) feeds the weight gradient -- the shift is
        # a stop-gradient target term, not part of the model (see the docstring)
        h_in = h if hs is None else (h.float() - hs[s:e]).to(hidden.dtype)
        if need_grads:
            with torch.enable_grad():
                logits = F.linear(h_in, w16, b16)
                if singular is not None:
                    logits = logits.float()
                    _add_singular_logits(logits, singular, s, e)
                if col_mask is not None:
                    logits.add_(col_mask if logits.dtype == torch.float32 else cm16)
                logits.requires_grad_(True)
                loss = fused_cross_entropy(logits, y, reduction='none', inplace_backward=True)
                (loss.float() * rw).sum().backward()
            dlogits = logits.grad
            loss = loss.detach()
            dl16 = dlogits.to(hidden.dtype)
            grad_hidden[s:e] = dl16 @ w16
            grad_weight += (dl16.t() @ h).float()  # h, NOT h_in (stop-grad shift)
            if grad_bias is not None:
                grad_bias += dlogits.sum(dim=0).float()
        else:
            logits = F.linear(h_in, w16, b16)
            if singular is not None:
                logits = logits.float()
                _add_singular_logits(logits, singular, s, e)
            if col_mask is not None:
                logits.add_(col_mask if logits.dtype == torch.float32 else cm16)
            loss = fused_cross_entropy(logits, y, reduction='none')
        loss_sum += (loss.float() * rw).sum()
    return loss_sum, grad_hidden, grad_weight, grad_bias


class _ChunkedLinearCE(torch.autograd.Function):
    """Autograd wrapper: the forward computes all grads; the backward only scales them."""

    @staticmethod
    def forward(ctx, hidden, w, b, targets, row_weight, chunk_rows, singular, col_mask, hidden_shift):
        loss_sum, gh, gw, gb = _chunked_linear_ce(
            hidden,
            w,
            b,
            targets,
            row_weight,
            chunk_rows,
            need_grads=True,
            singular=singular,
            col_mask=col_mask,
            hidden_shift=hidden_shift,
        )
        ctx.save_for_backward(gh, gw, gb)
        return loss_sum

    @staticmethod
    def backward(ctx, grad_out):
        gh, gw, gb = ctx.saved_tensors
        g = grad_out.float()
        return (
            (gh.float() * g).to(gh.dtype),
            gw * g,
            gb * g if gb is not None else None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def chunked_linear_cross_entropy(
    hidden,
    weight,
    bias,
    targets,
    row_weight,
    chunk_rows=CE_CHUNK_ROWS,
    singular=None,
    col_mask=None,
    hidden_shift=None,
):
    """Compute the fused head projection + cross-entropy over row chunks.

    Never materializes the full `(n_rows, vocab)` logits. `singular` is an optional descriptor of
    constant fp32 tensors (see `_add_singular_logits`) whose analytic singular logits are added
    chunkwise to the network logits. `col_mask` is an optional `(V,)` additive logit mask (-inf on
    the GEMM-padding vocab columns). `hidden_shift` is an optional stop-gradient feature shift
    subtracted before the head (self-conditioning guidance distillation; see `_chunked_linear_ce`).
    Returns the `row_weight`-weighted sum of per-token CE losses as an fp32 scalar (differentiable
    w.r.t. `hidden`, `weight`, and `bias`).
    """
    if torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad):
        return _ChunkedLinearCE.apply(
            hidden, weight, bias, targets, row_weight, chunk_rows, singular, col_mask, hidden_shift
        )
    loss_sum, _, _, _ = _chunked_linear_ce(
        hidden,
        weight,
        bias,
        targets,
        row_weight,
        chunk_rows,
        need_grads=False,
        singular=singular,
        col_mask=col_mask,
        hidden_shift=hidden_shift,
    )
    return loss_sum


# Smallest 1 - t at which the fused head+CE may be trained with singular logits on. The analytic
# term scales like ~16 t / (1 - t)^2 in absolute logit value, and the quack fused cross-entropy
# kernel (`_chunked_linear_ce`) is only accurate while |logits| stays below ~1e6-1e7: measured on
# synthetic rows, CE is wrong from |x| ~ 1e8 and NaN from ~1e9 (a pre-shift by the row max does not
# rescue it), which the singular term reaches at 1 - t ~ 3e-4 / 1e-4 (a t density reaching
# t = 0.99999 produces NaN weights within a few hundred steps). Sequences drawn closer to t = 1 than
# this (possible only under a density whose support reaches past it: uniform up to t_max > 1 - 5e-3,
# logit_normal, or a schedule with t_end that close to 1) bypass the fused kernel: `ReLF.forward`
# routes them through `_exact_singular_ce`, plain fp32 `F.cross_entropy` on dense logits, which
# matched an fp64 reference at every t in the same measurement. The sampler is unaffected (fp32
# torch softmax/argmax are exact there).
SINGULAR_TRAIN_MIN_ONE_MINUS_T = 5e-3


def _slice_singular(singular, rows):
    """Restrict the `_add_singular_logits` descriptor to the rows of the bool mask `rows`."""
    if singular is None:
        return None
    return (singular[0][rows], singular[1])


def _exact_singular_ce(hidden, w, b, targets, row_weight, singular, col_mask, hidden_shift=None):
    """Compute the row-weighted CE for rows whose singular logits exceed the fused kernel's range.

    Dense fp32 logits + torch's `F.cross_entropy` under ordinary autograd. Same logit construction
    as `_chunked_linear_ce` (bf16 head GEMM -> fp32, singular add in fp32 with TF32 off, additive
    `col_mask`) and the same stop-gradient `hidden_shift` semantics (logits from `hidden - shift`,
    weight gradient from the unshifted `hidden`: the shift's head term is detached), so the two
    paths compute the same loss and gradients wherever both are valid; only the CE reduction
    differs. Meant for a handful of sequences per micro-batch (each is an `(L, V)` fp32 block,
    ~200 MB per 1024-row sequence plus its softmax in backward); the caller keeps it rare.
    """
    w16 = w.to(hidden.dtype)
    logits = F.linear(hidden, w16, b.to(hidden.dtype) if b is not None else None)
    if hidden_shift is not None:
        logits = logits - F.linear(hidden_shift.to(hidden.dtype), w16).detach()
    logits = logits.float()
    if singular is not None:
        logits = logits.clone()
        _add_singular_logits(logits, singular, 0, logits.shape[0])
    if col_mask is not None:
        logits = logits + col_mask
    loss = F.cross_entropy(logits, targets, reduction='none')
    return (loss * row_weight).sum()
