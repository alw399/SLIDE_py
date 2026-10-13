# loveslide

**A Python interface to the SLIDE framework for latent factor discovery and statistical inference.**

---

## 📘 Overview

**loveslide** wraps key components of the original [SLIDE R package](https://github.com/jishnu-lab/SLIDE) into a user-friendly Python interface, making it easier to incorporate into machine learning pipelines and bioinformatics workflows.

SLIDE (Statistical Latent Inference for Discovery and Explanation) combines:

* **LOVE**: A latent factor discovery algorithm using model-based overlapping clustering.
* **Knockoffs**: For statistically rigorous identification of significant standalone and interacting latent factors.

This Python implementation retains R underpinnings via `rpy2` and is structured to be modular, extensible, and accessible from both the command line and within Python scripts or notebooks.

---

## 🔗 Related Repositories

* 📦 Original R package: [https://github.com/jishnu-lab/SLIDE](https://github.com/jishnu-lab/SLIDE)
* 🐍 Python wrapper: [https://github.com/alw399/SLIDE\_py](https://github.com/alw399/SLIDE_py)

---

## 🚀 Installation

Set up a compatible Python environment:

```bash
module load anaconda3/2022.10
conda create -n loveslide_env python=3.9 r-base
conda activate loveslide_env
pip install loveslide
```

If needed, clone the environment used during development:

```bash
# On the cluster:
source activate /ix3/djishnu/alw399/envs/rhino
```

---

## ⚡ Quick Start

### 📿 Command Line

```bash
python slide.py \
  --x_path /path/to/your/features.csv \
  --y_path /path/to/your/labels.csv \
  --out_path /path/to/output/
```

Use full paths if not running from the `src/loveslide` directory.

---

### 🧪 In a Notebook

```python
import loveslide

from loveslide import OptimizeSLIDE

input_params = {
    'x_path': '/path/to/features.csv',
    'y_path': '/path/to/labels.csv',
    'fdr': 0.1,
    'thresh_fdr': 0.1,
    'spec': 0.2,
    'y_factor': True,
    'niter': 500,
    'SLIDE_top_feats': 20,
    'rep_CV': 50,
    'pure_homo': True,
    'delta': [0.01],
    'lambda': [0.5, 0.1],
    'out_path': '/path/to/output/'
}

slider = OptimizeSLIDE(input_params)
slider.run_pipeline(verbose=True)
```

Knockoff iterations run serially. Parallel execution was removed because
it did not reproduce serial selections under the same R seed. Remove
`n_workers` from method calls and `input_params`; it is no longer supported.

---

## 🔬 Pipeline Overview

The `run_pipeline()` method follows three key stages:

### 🧩 Stage 1: Latent Factor Discovery

* **LOVE Algorithm**: Identifies overlapping latent factors in the data.
* **Output**: Latent factor matrix (`z_matrix`) and factor loadings.

### 📊 Stage 2: Statistical Inference with Knockoffs

* Identifies significant **standalone** and **interacting** latent factors.
* Controls **False Discovery Rate (FDR)** to maintain statistical rigor.

### 📈 Stage 3: Visualization

* Diagnostic plots
* Top genes/features for each latent factor (loadings > |0.05|)

---

## ⚙️ Parameters

| Name              | Type  | Description                          | Default/Example |
| ----------------- | ----- | ------------------------------------ | --------------- |
| `x_path`          | str   | Path to feature matrix CSV           | Required        |
| `y_path`          | str   | Path to response/labels CSV          | Required        |
| `fdr`             | float | Knockoff FDR threshold               | 0.1             |
| `thresh_fdr`      | float | FDR threshold in LOVE                | 0.1             |
| `spec`            | float | Minimum reproducibility for a factor | 0.2             |
| `y_factor`        | bool  | Treat `y` as categorical             | True            |
| `niter`           | int   | Knockoff iterations per batch        | 500             |
| `SLIDE_top_feats` | int   | Number of top features to plot       | 20              |
| `rep_CV`          | int   | Repeats for cross-validation         | 50              |
| `pure_homo`       | bool  | Use pure variables with loadings = 1 | True            |
| `delta`           | list  | Regularization parameters            | `[0.01]`        |
| `lambda`          | list  | Penalty parameters                   | `[0.5, 0.1]`    |
| `out_path`        | str   | Output directory                     | Required        |

---

## 🏗️ Project Structure

```
SLIDE_py/
├── src/
│   ├── loveslide/             # Main Python & R wrappers
│   │   ├── slide.py           # Main entry point
│   │   ├── love.py
│   │   ├── knockoffs.py
│   │   ├── ...
│   │   ├── LOVE-master/       # (Legacy) Original LOVE code
│   │   └── LOVE-SLIDE/        # Customized LOVE implementation for SLIDE
├── dist/
├── example/
├── ...
```

---

## 🧠 Design Notes

* Core statistical inference is done using **R scripts** via `rpy2`.
* Python acts as an orchestration layer to allow integration into ML workflows.
* Most plotting is done in **R** (e.g., `pheatmap`, `ggplot2`).

---

## 📌 Known Limitations and TODOs

* [x] YAML → dictionary conversion for easier parameter management
* [ ] Extend `y_factor` handling to non-binary variables
* [ ] Parallelization of knockoff inference (e.g., in `select_short_freq`)
* [ ] Correlation networks visualization using `networkx`

---

## 📢 Citation & Contact

If you use `loveslide` in your work, please cite the original R implementation and this repository. For bugs or feature requests, please open an issue on GitHub.

* **Homepage**: [SLIDE\_py on GitHub](https://github.com/alw399/SLIDE_py)
* **Issues**: [Report an Issue](https://github.com/alw399/SLIDE_py/issues)
* **Authors**:

  * Ally Wang (`alw399@pitt.edu`)
  * Swapnil Keshari (`swk25@pitt.edu`)
