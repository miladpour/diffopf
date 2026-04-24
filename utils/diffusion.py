import torch
import torch.nn.functional as F

def linear_beta_schedule(T, start=1e-4, end=0.02):
    return torch.linspace(start, end, T)

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def prepare_diffusion(T=1000):
    betas = linear_beta_schedule(T)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

    return {
        "T": T, 
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1 - alphas_cumprod),
        "sqrt_recip_alphas": torch.sqrt(1 / alphas),
        "posterior_variance": betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod),
    }


def get_index(vals, t, shape):
    batch = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch, *((1,) * (len(shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, diffusion_dict, device="cpu"):
    noise = torch.randn_like(x_0)

    sqrt_alphas_cumprod_t = get_index_from_list(
        diffusion_dict["sqrt_alphas_cumprod"], t, x_0.shape
    )
    sqrt_one_minus_t = get_index_from_list(
        diffusion_dict["sqrt_one_minus_alphas_cumprod"], t, x_0.shape
    )

    x_noisy = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_t * noise

    return x_noisy.to(device), noise.to(device)