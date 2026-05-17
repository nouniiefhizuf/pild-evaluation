# Beyond Pixel-Pushing: Evaluating LLMs as Actionable Physics Simulators

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/XXXX.XXXXX)

> **Official Implementation** of *"Beyond Pixel-Pushing: Evaluating Large Language Models as Actionable Physics Simulators via Prompt Engineering"*  
> Accepted at NeurIPS 2026 (Undergraduate Research Track)  
> Ahmed Soltani, Ryan Chanchah, Skander Darghouth, Khalil Ben Rejeb  
> South Mediterranean University, MedTech Institute

---

## Overview

This repository contains the complete implementation, datasets, prompts, and evaluation framework for our NeurIPS 2026 paper. We introduce **PhysBench-1K**, the first large-scale benchmark for evaluating LLMs as physics simulators, and **Physical Deviation Error (PDE)**, a novel metric that quantifies the divergence between generated trajectories and analytical ground truth.

**Key Results:**
- **1,000** curated physics simulation tasks across 3 domains
- **4** frontier LLMs evaluated (GPT-4o, Llama-3.3-70B, Qwen2.5-72B, DeepSeek-V3)
- **10** graded prompt levels from basic to physics-informed
- **25.9%** Pass@1 for GPT-4o (best) — **74.1% failure rate** even for frontier models
- All differences statistically significant ($p < 0.001$, Cohen's $d$ = 0.43–1.77)

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/medtech-tn/pild-evaluation.git
cd pild-evaluation

# Create conda environment
conda create -n pild python=3.11
conda activate pild

# Install dependencies
pip install -r requirements.txt
```

### Reproduce Main Results (5 minutes)

```bash
# Download pre-computed results and generate all figures/tables
python scripts/reproduce_all.py --results_dir results/ --output_dir analysis/figures/

# Or run full pipeline from scratch (requires API keys / GPU access)
python scripts/run_full_pipeline.py --models gpt-4o llama-3.3-70b qwen2.5-72b deepseek-v3
```

### Evaluate a Single Model on a Single Example

```python
from evaluation.pde_metric import compute_pde
from inference.gpt4o_wrapper import GPT4oWrapper
from physbench import load_example

# Load example
example = load_example("rigid_body", "projectile_001")

# Generate code with GPT-4o
model = GPT4oWrapper(temperature=0.0)
generated_code = model.generate(example.description, prompt_level="L10")

# Compute PDE
pde_score = compute_pde(generated_code, example.ground_truth_fn, 
                        example.t_span, example.y0, example.params)
print(f"PDE: {pde_score:.4f}")
```

---

## Repository Structure

```
pild-evaluation/
├── physbench/              # PhysBench-1K dataset
│   ├── rigid_body/         # 300 examples (projectile, pendulum, collision)
│   ├── fluid_dynamics/     # 400 examples (Navier-Stokes, Bernoulli, vortex)
│   └── multi_object/       # 300 examples (N-body, spring-mass, buoyancy)
├── prompts/                # All 10 prompt templates (L1-L10)
│   ├── L1_basic.txt
│   ├── L2_cot.txt
│   ├── ...
│   └── L10_physics_informed.txt
├── inference/              # Model API wrappers
│   ├── gpt4o_wrapper.py
│   ├── llama_wrapper.py
│   ├── qwen_wrapper.py
│   └── deepseek_wrapper.py
├── evaluation/             # Metrics and scoring
│   ├── pde_metric.py       # Physical Deviation Error
│   ├── execution_rate.py   # Code executability checker
│   ├── semantic_metrics.py # ROUGE-L, BERTScore
│   └── statistical_tests.py # t-tests, effect sizes, heatmaps
├── results/                # Raw model outputs (JSON)
│   ├── gpt-4o/
│   ├── llama-3.3-70b/
│   ├── qwen2.5-72b/
│   └── deepseek-v3/
├── analysis/               # Reproducible analysis
│   ├── notebooks/
│   │   ├── 01_main_results.ipynb
│   │   ├── 02_statistical_tests.ipynb
│   │   ├── 03_prompt_engineering.ipynb
│   │   ├── 04_domain_analysis.ipynb
│   │   └── 05_correlation_analysis.ipynb
│   └── figures/            # All paper figures (PDF + PNG)
├── docs/                   # Additional documentation
│   ├── DATASET.md          # Dataset construction details
│   ├── METRICS.md          # Metric definitions and formulas
│   └── API_SETUP.md        # API key configuration
├── tests/                  # Unit tests
├── scripts/                # Automation scripts
│   ├── reproduce_all.py
│   ├── run_full_pipeline.py
│   └── download_results.py
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

---

## PhysBench-1K Dataset

PhysBench-1K is a curated benchmark of 1,000 physics simulation tasks with human-verified ground truth. Each example includes:

| Field | Description |
|-------|-------------|
| `description` | Natural-language physics problem |
| `domain` | `rigid_body`, `fluid_dynamics`, `multi_object` |
| `difficulty` | `easy`, `medium`, `hard` |
| `initial_conditions` | Dictionary of parameter values |
| `ground_truth_fn` | Callable analytical solution |
| `reference_code` | Verified Python implementation |
| `t_span` | Simulation time range `[t0, tf]` |
| `y0` | Initial state vector |

### Load Dataset

```python
from physbench import load_dataset

# Load full dataset
dataset = load_dataset()
print(f"Total examples: {len(dataset)}")  # 1000

# Load by domain
rigid_body = load_dataset(domain="rigid_body")
fluid = load_dataset(domain="fluid_dynamics", difficulty="hard")
```

### Example Entry

```json
{
  "id": "rigid_body_001",
  "description": "A 2kg ball is thrown from ground level at 30 degrees above horizontal with initial speed 20m/s. Simulate the trajectory under gravity (g=9.81m/s^2), ignoring air resistance.",
  "domain": "rigid_body",
  "difficulty": "easy",
  "initial_conditions": {"m": 2.0, "v0": 20.0, "theta": 30.0, "g": 9.81},
  "t_span": [0.0, 2.5],
  "y0": [17.32, 10.0, 0.0, 0.0],
  "reference_code": "import numpy as np\nfrom scipy.integrate import solve_ivp\n...",
  "ground_truth": {
    "range": 35.31,
    "max_height": 5.10,
    "flight_time": 2.04
  }
}
```

---

## Prompt Engineering Levels

We design 10 graded prompt levels validated by 12 native speakers (Cronbach's $\alpha = 0.87$):

| Level | Name | Physics Relevance | PDE Reduction (GPT-4o) |
|-------|------|-------------------|------------------------|
| L1 | Basic Zero-Shot | 2.1 | Baseline |
| L2 | Zero-Shot CoT | 2.8 | 18.4% |
| L3 | Role: Expert | 3.5 | 31.2% |
| L4 | Few-Shot | 3.2 | 42.7% |
| L5 | Structured CoT | 3.8 | 51.3% |
| L6 | Self-Consistency | 3.6 | 48.9% |
| L7 | Analogical | 3.0 | 29.5% |
| L8 | **Constraint-Explicit** | **4.5** | **78.2%** |
| L9 | Error-Correction | 4.2 | 71.6% |
| L10 | **Physics-Informed** | **4.9** | **93.2%** |

### Use Custom Prompt Level

```python
from prompts import load_prompt

prompt_template = load_prompt("L8")  # Constraint-Explicit
filled_prompt = prompt_template.fill(
    physics_problem="Projectile motion with air resistance",
    initial_conditions={"v0": 25.0, "theta": 45.0, "drag_coeff": 0.47}
)
```

---

## Evaluation Metrics

### Physical Deviation Error (PDE)

Our novel metric quantifies trajectory divergence from analytical ground truth:

$$\text{PDE} = \\frac{1}{N} \\sum_{i=1}^{N} \\sqrt{\\sum_{j=1}^{d} \\left(x_j(t_i) - x_j^*(t_i)\\right)^2}$$

Normalized per-domain to $[0, 1]$. Lower is better.

```python
from evaluation.pde_metric import compute_pde

pde = compute_pde(
    generated_code=code_str,
    ground_truth_fn=analytical_solution,
    t_span=(0, 10),
    y0=np.array([1.0, 0.0]),
    params={"g": 9.81}
)
```

### Other Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Execution Rate** | Proportion of code that runs without errors | $[0, 1]$ |
| **Pass@1** | Binary: executable AND PDE < 0.15 | $\{0, 1\}$ |
| **ROUGE-L** | Longest common subsequence with reference | $[0, 1]$ |
| **BERTScore** | Token-level semantic similarity (CodeBERT) | $[0, 1]$ |

---

## Reproducing Paper Results

### Option 1: Pre-computed Results (Fast, ~5 min)

```bash
# Download our pre-computed model outputs
python scripts/download_results.py

# Generate all figures and tables
python scripts/reproduce_all.py
```

This reproduces:
- Table 1: Main Results
- Table 2: Prompt Validation Survey
- Table 3: Statistical Significance
- Figure 2: Performance Comparison
- Figure 3: P-value Heatmap
- Figure 4: Prompt Engineering Impact
- Figure 5: Domain-Specific Analysis
- Figure 6: Correlation Analysis

### Option 2: Full Reproduction (Slow, ~48 hours, requires API keys / GPUs)

```bash
# 1. Configure API keys
cp docs/API_SETUP.md .env  # Edit with your keys

# 2. Run full inference pipeline
python scripts/run_full_pipeline.py \
    --models gpt-4o llama-3.3-70b qwen2.5-72b deepseek-v3 \
    --prompt_levels L1 L2 L3 L4 L5 L6 L7 L8 L9 L10 \
    --num_examples 1000 \
    --output_dir results/

# 3. Compute metrics
python evaluation/compute_all_metrics.py --results_dir results/

# 4. Generate figures
python analysis/generate_figures.py --metrics_dir results/metrics/
```

### Hardware Requirements

| Model | Hardware | Time (1,000 examples) |
|-------|----------|----------------------|
| GPT-4o | API only | ~4 hours |
| Llama-3.3-70B | 4× A100 80GB | ~8 hours |
| Qwen2.5-72B | 4× A100 80GB | ~9 hours |
| DeepSeek-V3 | API only | ~6 hours |

---

## Model Checkpoints & Inference

### Closed Models (API)

```python
from inference.gpt4o_wrapper import GPT4oWrapper

model = GPT4oWrapper(
    api_key="your-openai-key",  # or set OPENAI_API_KEY env var
    temperature=0.0,
    max_tokens=2048,
    top_p=1.0
)

code = model.generate(
    prompt="Simulate a damped harmonic oscillator",
    system_message="You are a physics simulation expert."
)
```

### Open Models (Local GPU)

```python
from inference.llama_wrapper import LlamaWrapper

model = LlamaWrapper(
    model_path="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=4,  # 4 GPUs
    temperature=0.0,
    max_tokens=2048
)

code = model.generate("Simulate projectile motion with drag")
```

---

## Citation

If you use PhysBench-1K, PDE metric, or this codebase in your research, please cite:

```bibtex
@inproceedings{soltani2026beyond,
  title={Beyond Pixel-Pushing: Evaluating Large Language Models as Actionable Physics Simulators via Prompt Engineering},
  author={Soltani, Ahmed and Chanchah, Ryan and Darghouth, Skander and Ben Rejeb, Khalil},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026},
  volume={39},
  organization={South Mediterranean University, MedTech Institute}
}
```

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

The PhysBench-1K dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Model outputs are subject to the respective licenses of GPT-4o (OpenAI), Llama-3.3 (Meta), Qwen2.5 (Alibaba), and DeepSeek-V3 (DeepSeek-AI).

---

## Acknowledgements

- Prof. Abdeldjalil Labed for guidance on experimental design
- MedTech Institute HPC cluster for GPU compute
- Open-source communities: PyTorch, vLLM, SciPy, Transformers

---

## Contact

For questions, bug reports, or collaboration inquiries:

- Open an [Issue](https://github.com/medtech-tn/pild-evaluation/issues)
- Email: ahmed.soltani@medtech.tn
- Project Page: [medtech-tn.github.io/pild](https://medtech-tn.github.io/pild)

---

**Last Updated:** May 2026 | **Version:** 1.0.0
