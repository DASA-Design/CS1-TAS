# CS-1 TAS: A DASA Evaluation of the Tele Assistance System

Dimensional Analysis for Software Architecture (DASA) evaluation of the **Tele
Assistance System** (TAS), a service-based self-adaptive exemplar for chronic-care home
monitoring with a centralised MAPE-K loop (Weyns and Calinescu, SEAMS 2015).

This repository is a **reproducible deliverable**, not a library: it produces the
metric datasets and figures that ground a DASA case study. It is **Phase 1,
model-only** (analytic + stochastic + dimensional prediction, plus a constructive
search), consuming the [PyDASA](https://github.com/DASA-Design/PyDASA) library.

## Documentation

The `docs/` folder holds the case study in three parts, read in order:

1. [docs/case-study.md](docs/case-study.md) - **what TAS is**: the architecture, the
   analyse-and-act workflow, the ReSeP service catalogue, the quality scenarios, and the
   S1 / S2 adaptation strategies (the source-faithful reconstruction).
2. [docs/procedure.md](docs/procedure.md) - **how the experiment runs**: the DASA method,
   the hypotheses under test, and the idea behind each of the six notebooks.
3. [docs/report.md](docs/report.md) - **what it found**: the DASA modelling derivation,
   the per-method verdicts, the performance-versus-availability trade-off, the constructive
   search, threats to validity, and research-question coverage.

## What it decides

TAS is modelled as a 14-node loss-network queue (13 service stations plus one absorbing
`FAIL` sink); availability is read from the flow leaked to the sink. Two
quality-attribute requirements are decided across four adaptation strategies, framed as
a **performance-versus-availability trade-off**:

| Requirement | Metric | Threshold | Source |
|---|---|---|---|
| **R1** (Availability) | end-to-end failure rate | **<= 1.0 %** | Weyns 2015 |
| **R2** (Performance) | end-to-end response time | **<= 26 ms** | Camara 2023 |

(An original cost requirement, R3, is out of scope for this study.)

| Adaptation | Failure rate | R1 | Response time | R2 |
|---|---|---|---|---|
| baseline (no adaptation) | 14.82 % | FAIL | 25.74 ms | PASS |
| S1 (Retry) | 0.20 % | PASS | 27.72 ms | FAIL |
| S2 (Select-Reliable) | 10.42 % | FAIL | 18.53 ms | PASS |
| **aggregate (S1 & S2)** | **0.07 %** | **PASS** | **19.40 ms** | **PASS** |
| search winner | 0.03 % | PASS | 19.85 ms | PASS |

Only the **aggregate** passes both among the hand-authored adaptations. A
DASA-coefficient-guided search constructs a configuration that beats it and is
cross-validated by an independent scipy optimiser.

## Methods

Three independent predictive pipelines compute the same verdicts, and a fourth notebook
runs the constructive search:

| Notebook | Method | What it does |
|---|---|---|
| [01-analytic.ipynb](01-analytic.ipynb) | analytic | closed-form M/M/c/K Jackson solve (loss network + FAIL sink) |
| [02-stochastic.ipynb](02-stochastic.ipynb) | stochastic | SimPy discrete-event simulation of the same network |
| [03-dimensional.ipynb](03-dimensional.ipynb) | dimensional | PyDASA Pi-groups + dimensionless viable-region bounds |
| [04-yoly.ipynb](04-yoly.ipynb) | dimensional | design-space sweep across the `(mu, c, K)` grid |
| [05-search.ipynb](05-search.ipynb) | search | coefficient-guided descent + scipy cross-validation |
| [06-comparison.ipynb](06-comparison.ipynb) | comparison | cross-method verdict matrix + numerical agreement |

## Setup

Requires Python 3.12+.

```bash
python -m venv venv
source venv/Scripts/activate         # Git Bash on Windows; use venv/bin/activate on Linux/macOS

# 1. Install PyDASA (not on public PyPI) from source:
pip install "git+https://github.com/DASA-Design/PyDASA.git@v0.7.1"

# 2. Install the rest:
pip install -r requirements.txt
```

If the PyDASA tag or location differs, adjust the first command; the pin in
`requirements.txt` (`pydasa==0.7.1`) documents the expected version.

## Running

Each method is a module with a small CLI. Adaptation is one of
`baseline | s1 | s2 | aggregate`:

```bash
python -m src.methods.analytic     --adaptation aggregate
python -m src.methods.stochastic   --adaptation aggregate
python -m src.methods.dimensional  --adaptation aggregate
```

Or run the notebooks end-to-end (this regenerates every dataset and figure):

```bash
jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
```

## Layout

```
├── 01-analytic … 06-comparison.ipynb   # the pipeline, in order
├── src/
│   ├── methods/       # per-method orchestrators (run() + CLI)
│   ├── analytic/      # closed-form queueing-network solvers
│   ├── stochastic/    # SimPy DES processes
│   ├── dimensional/   # PyDASA schema, coefficients, routing, search
│   ├── view/          # plotting helpers
│   ├── io/            # config loaders
│   └── utils/         # shared helpers
├── data/
│   ├── config/        # single source of truth for parameters (PyDASA Variable schema)
│   ├── results/       # per-method, per-adaptation metric JSONs + verdicts
│   └── reference/     # ground-truth thresholds
├── docs/              # case study, procedure, report + figures
└── tests/             # pytest, mirrors src/
```

Figures are written under `data/img/<method>/<adaptation>/` when the notebooks or
methods run; they are not committed (regenerate them from the pipeline).

## Data convention

`data/config/` is the single source of truth. Every config uses the PyDASA
`Variable`-dict schema keyed by LaTeX symbol. Methods read a profile plus an adaptation
and write a single JSON per run to `data/results/<method>/<adaptation>/`, plus a
profile-agnostic `requirements.json` carrying the R1 / R2 verdicts. Files under
`data/results/` are regenerated by the methods; do not hand-edit them.

## Tests

```bash
pytest tests/ -q
```

## License

GPL-3.0. See [LICENSE](LICENSE).
