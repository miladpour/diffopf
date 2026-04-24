## Overview

DiffOPF treats solving the Optimal Power Flow (OPF) problem as conditional sampling from the underlying distribution of historical OPF data points.

Given active and reactive load demands $(P_d, Q_d)$, the model learns a conditional distribution over generator dispatches $(P_g, Q_g)$.

The pipeline consists of two main stages:

1. **Data generation (Julia + PowerModels)**  
   Generate AC-OPF data points under randomized load and cost perturbations.

2. **Training + Sampling (Python + Diffusion Model)**  
   Train a diffusion model on OPF solutions and sample generator outputs conditioned on loads.

---

## Repository Structure

```
DiffOPF/
├── main.py                              # Entry point: sampling/inference
├── training/
│   └── train.py                         # Diffusion model training script
├── sampling/
│   └── sample.py                        # Reverse diffusion sampler (DPS-guided)
├── models/
│   └── model.py                         # Score network (SimpleNN + time embedding)
├── utils/
│   ├── diffusion.py                     # Noise schedule + diffusion utilities
│   └── normalization.py                 # Min-max normalization / denormalization
│
├── data/
│   ├── train/                           # Training dataset (generated via Julia script)
│   ├── test/                            # Test dataset (generated via Julia script)
│   └── test_case/                       # Test case inputs 
│
├── configs/
│   └── IEEE_118_Parameters.json         # System parameters (buses, lines, generators)
│
├── checkpoints/
│   └── trained_model.pth                # Trained diffusion model weights
│
├── outputs/
│   └── DiffOPF_Solution.csv             # Generated OPF samples
│
├── data_generation/
│   └── opf_data_generation.jl           # Julia script for AC-OPF dataset generation
│
├── requirements.txt                     # Python dependencies
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/DiffOPF.git
cd DiffOPF
pip install -r requirements.txt
```

### Requirements
- Python ≥ 3.8  
- PyTorch ≥ 1.12  
- Julia ≥ 1.8 (for data generation)  
- PowerModels.jl, JuMP, Ipopt

---

## Data Generation (Julia)

Before training the diffusion model, generate OPF datasets using:

```bash
julia data_generation/opf_data_generation.jl
```

### Notes
- The script solves AC-OPF for randomized load and cost perturbations.
- Inputs are read from:

```
data/test_case/
```

- Output datasets are saved into:

```
data/train/
data/test/
```

### Output format

Each row contains:

```
Pd_1 ... Pd_n, Qd_1 ... Qd_n, Pg_1 ... Pg_n, Qg_1 ... Qg_n
```

---

## Training

Train the diffusion model:

```bash
python training/train.py
```

### What it does
- Loads training dataset from `data/train/`
- Applies Min-Max normalization
- Trains a score-based diffusion model
- Saves:
  - model checkpoint → `checkpoints/`
  - loss curve → `outputs/`

---

## Inference / Sampling

Run conditional sampling:

```bash
python main.py --n_instances 5 --num_samples 2 --output outputs/DiffOPF_Solution.csv
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--n_instances` | 5 | Number of OPF test instances |
| `--num_samples` | 2 | Samples per instance |
| `--output` | outputs/DiffOPF_Solution.csv | Output CSV path |

---

## Output Format

Each generated sample contains:

```
instance_id, sample_id,
Pd_1 ... Pd_n,
Qd_1 ... Qd_n,
Pg_1 ... Pg_n,
Qg_1 ... Qg_n
```

---

## Pretrained Model

A pretrained checkpoint is provided:

```
checkpoints/trained_model.pth
```
---

## Citation

```bibtex
@misc{hoseinpour2026diffopfdiffusionsolveroptimal,
      title={DiffOPF: Diffusion Solver for Optimal Power Flow},
      author={Milad Hoseinpour and Vladimir Dvorkin},
      year={2026},
      eprint={2510.14075},
      archivePrefix={arXiv},
      primaryClass={eess.SY},
      url={https://arxiv.org/abs/2510.14075}
}
```

## License

This project is licensed under the MIT License. See `LICENSE` for details.