# ref: https://github.com/kingsj0405/ciplab-NTIRE-2020/blob/17a8f3d5e0531d5ce0a0a4af07ab048cd94121cb/utils.py
import torch

def Huber(input, target, delta=0.01, reduce=True):
    abs_error = torch.abs(input - target)
    quadratic = torch.clamp(abs_error, max=delta)

    # The following expression is the same in value as
    # tf.maximum(abs_error - delta, 0), but importantly the gradient for the
    # expression when abs_error == delta is 0 (for tf.maximum it would be 1).
    # This is necessary to avoid doubling the gradient, since there is already a
    # nonzero contribution to the gradient from the quadratic term.
    linear = (abs_error - quadratic)
    losses = 0.5 * torch.pow(quadratic, 2) + delta * linear
    
    if reduce:
        return torch.mean(losses)
    else:
        return losses
