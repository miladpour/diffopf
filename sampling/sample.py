import torch
import pandas as pd
import json
from utils.diffusion import prepare_diffusion, get_index
from utils.normalization import denormalize


def sample_timestep(x, t, model, params, data_dict, dataset_test, opf_counter):
    x = x.clone().detach().requires_grad_(True).double()

    betas = data_dict["betas"]
    alphas = data_dict["alphas"]
    alphas_cumprod = data_dict["alphas_cumprod"]
    alphas_cumprod_prev = data_dict["alphas_cumprod_prev"]
    sqrt_one_minus = data_dict["sqrt_one_minus_alphas_cumprod"]
    sqrt_recip = data_dict["sqrt_recip_alphas"]
    sqrt_alpha_bar = data_dict["sqrt_alphas_cumprod"]
    posterior_var = data_dict["posterior_variance"]

    betas_t = get_index(betas, t, x.shape)
    sqrt_one_minus_t = get_index(sqrt_one_minus, t, x.shape)
    sqrt_recip_t = get_index(sqrt_recip, t, x.shape)
    sqrt_alpha_bar_t = get_index(sqrt_alpha_bar, t, x.shape)

    noise_pred = model(x, t)
    x0_hat = (x - sqrt_one_minus_t * noise_pred) / sqrt_alpha_bar_t

    alpha_t = get_index(alphas, t, x.shape)
    alpha_bar_prev = get_index(alphas_cumprod_prev, t, x.shape)
    alpha_bar_t = get_index(alphas_cumprod, t, x.shape)

    sigma = torch.sqrt(get_index(posterior_var, t, x.shape))
    z = torch.randn_like(x)

    n_d = params["n_d"]
    n_g = params["n_g"]
    mask = torch.zeros(2*n_d + 2*n_g, dtype=x.dtype)
    mask[:2*n_d] = 1

    y = dataset_test[opf_counter].to(device=x.device, dtype=torch.double).view(-1)

    y = y * mask
    A = torch.eye(2*n_d + 2*n_g, dtype=torch.double)
    x0_hat = x0_hat.view(-1)
    y = y.view(-1)

    if t <= 900:
        residual = A @ x0_hat - y
        grad = 2 * (A.T @ residual.view(-1))
        #grad = 2 * A.T @ (A @ x0_hat - y)
        x_prime = x0_hat - grad

        x = (
            torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t) * x
            + torch.sqrt(alpha_bar_prev) * betas_t / (1 - alpha_bar_t) * x_prime
            + sigma * z
        )
    else:
        x = (
            torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t) * x
            + torch.sqrt(alpha_bar_prev) * betas_t / (1 - alpha_bar_t) * x0_hat
            + sigma * z
        )

    return x.detach()


def sample(model, params, data_dict, dataset_test, opf_counter, device):
    dim = 2*params["n_d"] + 2*params["n_g"]
    x = torch.randn(dim, device=device, dtype=torch.double)

    for i in reversed(range(1000)):
        t = torch.tensor([i], device=device)
        x = sample_timestep(x, t, model, params, data_dict, dataset_test, opf_counter)

    return x


def run_sampling(n_instances, num_samples, output_path,
                 dataset_test, dataset_original, model, params):

    device = next(model.parameters()).device
    data_dict = prepare_diffusion()

    rows = []

    for opf_counter in range(n_instances):
        for i in range(num_samples):
            x = sample(model, params, data_dict, dataset_test, opf_counter, device)
            x_np = x.cpu().numpy()

            row = {"instance_id": opf_counter+1, "sample_id": i+1}

            n_d, n_g = params["n_d"], params["n_g"]

            vals = list(x_np)
            for idx, val in enumerate(vals):
                row[f"x_{idx}"] = val

            rows.append(row)

    df = pd.DataFrame(rows)
    rows_denorm = []

    feature_cols = [f"x_{i}" for i in range(2*n_d + 2*n_g)]

    # Convert all samples at once (NO loop denorm)
    x_tensor = torch.tensor(
        df[feature_cols].values,
        dtype=torch.double
    )

    x_denorm = denormalize(x_tensor, dataset_original).cpu().numpy()

    for idx in range(len(df)):
        x = x_denorm[idx]

        Pd = x[:n_d]
        Qd = x[n_d:2*n_d]
        Pg = x[2*n_d:2*n_d+n_g]
        Qg = x[2*n_d+n_g:2*n_d+2*n_g]

        row = {
            "instance_id": df.iloc[idx]["instance_id"],
            "sample_id": df.iloc[idx]["sample_id"]
        }

        for k, v in enumerate(Pd, 1):
            row[f"Pd_{k}"] = v
        for k, v in enumerate(Qd, 1):
            row[f"Qd_{k}"] = v
        for k, v in enumerate(Pg, 1):
            row[f"Pg_{k}"] = v
        for k, v in enumerate(Qg, 1):
            row[f"Qg_{k}"] = v

        rows_denorm.append(row)

    diffopf_solutions_denorm_df = pd.DataFrame(rows_denorm)

    import os
    os.makedirs("outputs", exist_ok=True)
    diffopf_solutions_denorm_df.to_csv(output_path, index=False)

    print(f"Saved denormalized results to {output_path}")
