import torch
import triton
import triton.language as tl
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        M,N = x.reshape(-1, x.shape[-1]).shape

        y = torch.empty_like(x)
        mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)

        MAX_FUSED_SIZE = 65536 // x.element_size()
        
    @staticmethod
    def backward(

    )

def test_layernorm_kernel(M, N, dtype, eps=1e-5, device=DEVICE):
    x = -2.3 + 0.5 * torch.randn((M,N), dtype=dtype, device=device)
    x.requires_grad_(True)

    weight = torch.rand((N, ), dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand((N, ), dtype=dtype, device=device, requires_grad=True)

    y_tri = layernorm(x, (N, ), weight ,bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, (N, ), weight, bias, eps).to(dtype)

    torch.testing.assert_close(y_tri, y_ref, atol=1e-2, rtol=0)
    print("Passed FWD")

    dLdy = 0.1 * torch.randn_like(x)
    y_tri.backward(dLdy, retain_graph=True)
    dLdx_tri, dLdw_tri, dLdb_tri = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None

    y_ref.backward(dLdy, retain_graph=True)
    dLdx_ref, dLdw_ref, dLdb_ref = [_.grad.clone() for _ in [x, weight, bias]]

    torch.testing.assert_close(dLdx_tri, dLdx_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdw_tri, dLdw_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdb_tri, dLdb_ref, atol=1e-2, rtol=0)
    print("Passed BWD")
    

