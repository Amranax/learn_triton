 

import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')


@triton.jit
def seeded_dropout_kernel(
    x_ptr, output_ptr,
    n_elements,
    p, # fp32 [0,1]
    seed, #int32
    BLOCK_SIZE: tl.constexpr,
):
    PID = tl.program_id(axis=0)
    offsets = PID * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    random = tl.rand(seed, offsets)
    x_keep = random > p

    output = tl.where(x_keep, x / (1-p), 0.0)

    tl.store(output_ptr + offsets, output, mask=mask)

def seeded_dropout(x, p, seed):
    assert x.is_contiguous

    output = torch.empty_like(x)
    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    seeded_dropout_kernel[grid](
        x, output,
        n_elements,
        p, seed,
        BLOCK_SIZE=1024
    )
    return output

if __name__ == "__main__":
    x = torch.randn(size=(8,), device=DEVICE)
    output1 = seeded_dropout(x, p=0.5, seed=123)
    output2 = seeded_dropout(x, p=0.5, seed=123)
    output3 = seeded_dropout(x, p=0.5, seed=321)

    print(x, output1, output2, output3, sep="\n")