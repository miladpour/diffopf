import argparse
import pandas as pd
import torch
import json
import random
import numpy as np

from models.model import SimpleNN
from utils.normalization import normalize
from sampling.sample import run_sampling


def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_instances", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--output", type=str, default="outputs/DiffOPF_Solution.csv")

    args = parser.parse_args()
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = pd.read_csv("data/IEEE118_Pd_Qd_Pg_Qg_train.csv", header=0)
    test = pd.read_csv("data/IEEE118_Pd_Qd_Pg_Qg_test.csv", header=0)
    # REMOVE metadata column
    train = train.drop(columns=["iteration"])
    test = test.drop(columns=["iteration"])
    train = train.apply(pd.to_numeric, errors="coerce")
    test = test.apply(pd.to_numeric, errors="coerce")
    train = train.dropna().reset_index(drop=True)
    test = test.dropna().reset_index(drop=True)
    ###
    # Load params
    with open("configs/IEEE_118_Parameters.json") as f:
        params = json.load(f)["dims"]

    n_d = params["n_d"]
    n_g = params["n_g"]

    train = train.iloc[:, :-n_g]
    test = test.iloc[:, :-n_g]

    dataset = normalize(train)
    dataset_test = normalize(test)


    dim = 2*params["n_d"] + 2*params["n_g"]

    model = SimpleNN(dim, dim).to(device)
    model.load_state_dict(torch.load("checkpoints/trained_model_DiffOPF_training_IEEE_118_5_1000.pth", map_location=device))
    model.eval()

    run_sampling(
        args.n_instances,
        args.num_samples,
        args.output,
        dataset_test,
        test,
        model,
        params
    )


if __name__ == "__main__":
    main()

