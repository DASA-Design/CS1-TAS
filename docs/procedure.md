# TAS: Experimental Procedure

This document explains how the CS-1 Tele Assistance System (TAS) experiment is run and the idea behind each of the six notebooks. It is the method companion to two sibling documents: [case-study.md](case-study.md) describes what TAS is (the system, its quality attributes, and its service catalogue), and [report.md](report.md) presents the results together with the full DASA modelling derivation. This procedure keeps the model at a high level and points to [report.md](report.md) wherever the derivation matters.

CS-1 is DASA Phase 1, model-only. It models TAS as a loss-network queue and decides two quality-attribute requirements across four adaptations, then asks a constructive design question. Three independent predictive pipelines (analytic closed-form, stochastic simulation, dimensional coefficient bounds) compute the same verdicts, and a coefficient-guided search constructs a configuration that clears both requirements.

The locked envelope for every run is $\lambda_z = 323$ req/s (the E-QS canonical inflection, $\rho_b \leq 0.65$), $K = 16$, and $c = 1$.

---

## 1. Question and hypotheses

The organising discipline is architectural experimentation in the sense of Pureur and Bittner: a Quality Attribute Requirement (QAR) is not a target to assert and defend, it is a hypothesis about value that has to be tested. If you do not test it, you are betting on it. CS-1 treats its two requirements this way. Rather than asserting that a given adaptation satisfies both requirements, it states each requirement as a falsifiable predicate and runs apparatus that could reject it. The shift from the traditional stance (the architect assumes, decides, defends) to the experimental stance (the architect hypothesises, experiments, falsifies) is the posture of every notebook here.

### 1.1 Atomic, timely, unambiguous

Pureur and Bittner name three properties of an effective experiment. CS-1 instantiates each one deliberately.

- **Atomic** (one question at a time). S1 isolates the **retry** lever and S2 isolates the **selection** lever. The aggregate is the deliberate co-variation of the two. Isolating retry from selection is what lets the experiment attribute the availability change to retry and the response-time change to selection, then show that the aggregate combines both. Bundling them would forfeit that attribution.
- **Timely** (fast feedback). Analytic scoring solves all four adaptations in seconds, and the search's coarse candidate grid finishes in a few seconds, so a design question returns inside a single working session.
- **Unambiguous** (a pre-stated, measurable success criterion). R1 $\leq 1.0\%$ and R2 $\leq 26$ ms are fixed before the runs. Each verdict is a PASS or FAIL bit per cell, not a narrative judgement.

The atomic property is the methodological heart of CS-1. It is why the project models S2 as selection only (no failover), even though the original case-study source bundles sequential failover into its Select-Reliable strategy. Isolating the lever is the point; the aggregate then recombines retry and selection, and that recombination corresponds to the source's bundled strategy.

### 1.2 The four-piece structure mapped onto the notebooks

CS-1 follows the four-piece experimental structure of Hypothesis, Model, Apparatus, and Validation. The six notebooks map onto it as follows.

| Piece | Lives in | CS-1 instance |
|---|---|---|
| **Hypothesis** | this document, section 1.3 | H1, H2, H3 |
| **Model (prediction)** | [01-analytic.ipynb](../01-analytic.ipynb), [03-dimensional.ipynb](../03-dimensional.ipynb) | loss-network M/M/c/K solve with a FAIL sink, re-framed dimensionally |
| **Apparatus (measurement)** | [02-stochastic.ipynb](../02-stochastic.ipynb), [04-yoly.ipynb](../04-yoly.ipynb) | a simulation of the same network, independent of the closed form; a design-space sweep |
| **Validation (decision)** | [06-comparison.ipynb](../06-comparison.ipynb), [05-search.ipynb](../05-search.ipynb) | cross-method verdict matrix and numerical agreement; bound-versus-verdict check |

The key separation the structure demands (apparatus that produces measurements, kept distinct from a thin step that decides) is honoured. Each method writes raw `network`, `nodes`, and `requirements.json` envelopes to disk; the comparison notebook reads them back read-only, and no method recomputes another method's prediction inside the decision step.

### 1.3 Hypotheses and tolerances

Three hypotheses are decided. Tolerances were fixed before the runs and are motivated by measurement noise and the model's approximation budget, not chosen afterward to fit the results.

**H1. Verdict equivalence (categorical).** For every (adaptation, requirement) cell, the three predictive methods return the same PASS or FAIL bit.
Decision rule: accept if and only if all 24 cells (3 methods x 4 adaptations x 2 requirements) are congruent.

**H2. Numerical congruence within simulation noise.** Analytic and dimensional are identical by construction, since the dimensional method reads its end-to-end response time and failure fraction from the same closed-form solve. The testable gap is analytic versus stochastic simulation.
Tolerance: $\lvert \delta W_{e2e} \rvert \leq 5\%$ (the M/M/c/K approximation budget) and $\lvert \delta \varepsilon_{e2e} \rvert \leq 0.1$ pp.
This is a real falsification opportunity: the simulation is an independent solver that simulates the network rather than re-running the analytic formula, so it could have disagreed by more than 5% and rejected the equivalence claim.

**H3. DASA bounds predict the operational verdict (falsifiable).** The dimensionless predicate $\sigma_{\mathrm{arch}} < \sigma_{R2}$ AND $\eta_{\mathrm{arch}} < \eta_{R1}$ should coincide with the analytic R1 AND R2 verdict on every candidate across a region containing both passes and fails.
Decision rule: accept if and only if the confusion matrix is diagonal over a non-empty mixed region.

An experiment whose tolerance is loose enough to absorb any disagreement is not an experiment. The 5% budget on H2 is tight enough that the independent solver could have failed it.

---

## 2. The DASA method

Failures are modelled as routing leaks absorbed at an explicit absorbing node, `FAIL_{1}` (node index 13, giving 14 nodes per scenario). Availability is read directly from flow rather than from a lost-call exposure product. The failed fraction is $\varepsilon_{e2e} = \lambda_{\mathrm{FAIL}} / \lambda_z$, which drives **R1**. The effective successful-completion rate is $\chi_{\mathrm{out}} = \lambda_z - \lambda_{\mathrm{FAIL}}$, and the end-to-end response time is $W_{e2e} = L_{\mathrm{net}} / \chi_{\mathrm{out}}$ (where $L_{\mathrm{net}}$ excludes the FAIL node), which drives **R2**. Failure is applied at the three workflow handlers with a dispatch-weighted pool failure, and retry (present in S1 and the aggregate) is a handler split-loop that shrinks the residual give-up flow multiplicatively. The provenance for the response-time relation is the General Response Time Law (Denning and Buzen, 1978).

The two requirements under test are **R1 (Availability)**, failure rate $\leq 1.0\%$ (Weyns 2015, as the fraction $0.01$), and **R2 (Performance)**, response time $\leq 26$ ms (Camara 2023). A third case-spec requirement, R3 (cost minimisation), is retired in this project; CS-1 is scoped to the R1 and R2 trade-off and does not model the cost axis. The full derivation of the loss-network formulas, the dimensionless re-framing (the $\theta$, $\sigma$, $\eta$, $\phi$ coefficients and the viable-region bounds $\sigma_{R2}$ and $\eta_{R1}$), and the validity envelope is in [report.md](report.md).

---

## 3. The notebooks

The notebooks run in pipeline order and are named with a numbered prefix so they sort that way. Notebooks 01 through 03 are the three predictive methods; each writes its own result envelopes to disk. Notebook 04 sweeps the design space. Notebook 05 runs the constructive search, consuming the dimensional operating points. Notebook 06 reads the persisted outputs of 01 through 03 (and the search winner from 05) and decides the cross-method verdict. Every notebook is thin: the logic lives in `src/`, and the cells orchestrate, display, and save figures.

### 3.1 01-analytic.ipynb (Model)

**Idea.** Solve TAS as an open queueing network in closed form, M/M/c/K per node, across the full adaptation axis, and check the result against the R1 and R2 targets. This is the primary prediction source: the analytic solve is the number every other method is compared against.

**Inputs.** `data/config/profile/dflt.json` and `data/config/profile/opti.json` (the Variable-dict profiles) plus `data/reference/baseline.json` (the R1 and R2 thresholds).

**Outputs.** Per-run metrics at `data/results/analytic/<adaptation>/<profile>.json` (nodes, network, routing, and `lambda_z`), the R1 and R2 verdicts at `data/results/analytic/<adaptation>/requirements.json`, and topology, heatmap, bar, and delta figures under `data/img/analytic/<adaptation>/`.

**Role.** The Model piece. It produces the canonical closed-form prediction for all four adaptations.

### 3.2 02-stochastic.ipynb (Apparatus)

**Idea.** Solve the same queueing network by discrete-event simulation and cross-check against the analytic closed form. The simulation mirrors the model's queue, service, routing, and failure assumptions but solves by simulating arrivals rather than by the closed-form formula, so its measurements are an independent check rather than a restatement of the analytic result.

**Inputs.** The same profile Variable dicts as the analytic method, plus `data/config/method/stochastic.json` (the simulation parameters: seed, replications, horizon and warmup in invocations) and the shared thresholds in `data/reference/baseline.json`.

**Outputs.** Per-run metrics at `data/results/stochastic/<adaptation>/<profile>.json` (nodes carrying `_std` columns, the network aggregate, routing, `lambda_z`, and the embedded method config), the verdicts at `data/results/stochastic/<adaptation>/requirements.json`, and topology, heatmap, diffmap, confidence-interval bar, and delta figures under `data/img/stochastic/<adaptation>/`.

**Role.** The Apparatus piece. Because it is an independent solver, it is the notebook that could have rejected H1 and H2 and did not. Every node's analytic point lands inside the simulation's confidence-interval band, which is the empirical justification for the aggregate agreement.

### 3.3 03-dimensional.ipynb (Model, re-framed)

**Idea.** Characterise TAS dimensionally for every adaptation. For each artifact, PyDASA derives Pi-groups from the relevant variables on the Time / Structure / Data basis, and four operationally meaningful coefficients are built from them: $\theta = L/K$ (Occupancy), $\sigma = W \cdot \lambda / K$ (Stall), $\eta = \chi \cdot K / (\mu \cdot c)$ (Effective-yield), and $\phi = M_{\mathrm{act}} / M_{\mathrm{buf}}$ (Memory-usage). The dimensional method does not produce a different verdict; it re-frames the same threshold check as a viable-region predicate in scale-free dimensionless space, which is what makes designs comparable across architectures.

The persisted dimensional solution carries a seeded 5% white-noise robustness pass. Section 1 runs with `noise_scope="coefficients"`, so the coefficients written to disk (and the figures, and the operating points that 04 and 05 later load) carry the 5% input noise, while the operational layer and the R1 and R2 verdict are solved on clean inputs. That keeps the persisted verdict bit-identical to analytic (which is what lets 06 report an exact analytic-equals-dimensional match) while the dimensionless operating points already reflect realistic input uncertainty. See [report.md](report.md) for how the noise is injected on the independent variables so the dimensional coupling is preserved.

**Inputs.** The profile Variable dicts and the seed from `data/config/method/dimensional.json`.

**Outputs.** Per-run metrics and per-artifact coefficients at `data/results/dimensional/<scenario>/<profile>.json`, verdicts at `data/results/dimensional/<scenario>/requirements.json`, and coefficient-view figures under `data/img/dimensional/<scenario>/`.

**Role.** The Model piece, re-framed dimensionally. It is the promotion of the Bass quality scenarios to Enhanced Quality Scenarios, and it supplies the dimensionless operating points the search navigates by.

### 3.4 04-yoly.ipynb (Apparatus, design space)

**Idea.** Produce a design-space coefficient cloud for the whole TAS architecture. For each `(mu_factor, c, K)` combination in the sweep grid, the external arrival rate is increased up to the first-node saturation point, then samples are drawn from the stable interval; at each sample the routing matrix is applied by Jackson propagation, so every node's arrival rate is consistent with the network topology rather than set in isolation. The saturation rule (first component saturates means the whole network is unstable) is enforced at every sample, so the cloud contains only feasible designs.

Each swept point carries the same seeded 5% white-noise disturbance on the independent variables as the dimensional method, with the queue re-solved per point, so the cloud's spread shows input-uncertainty propagation while the design knobs stay nominal. The four-adaptation overlay traces the baseline to s1 to s2 to aggregate trajectory through coefficient space, with an operating-point marker per adaptation loaded from the robust dimensional solution.

**Inputs.** The sweep grid and seed in `data/config/method/dimensional.json`, plus the profile Variable dicts.

**Outputs.** Per-adaptation clouds and the baseline-versus-aggregate overlay under `data/img/dimensional/yoly/<adaptation>/` and `data/img/dimensional/yoly/cmp/` (3D and 2D projections, per-node and architecture-level).

**Role.** The Apparatus piece extended to the design space. It shows where the hand-authored adaptations sit inside the feasible cloud and how input noise propagates through the coefficients.

### 3.5 05-search.ipynb (Validation, constructive)

**Idea.** The four published adaptations were authored by hand. This notebook asks a constructive question: navigating the design space by DASA's dimensionless coefficients alone, can we find a configuration that clears both requirements at the locked envelope and improves on the published aggregate? Two levers are searched jointly, dispatch weights (a continuous simplex over how traffic splits across a pool of service variants) and retry depth (a discrete choice per workflow handler between retrying and giving up on the first attempt).

**Method.** Stage 1 picks the three lowest-$\varepsilon$ service variants per type from the catalogue. Stage 2 runs a derivative-free coordinate-exchange descent that, at each step, takes the move that most reduces the binding dimensionless coefficient, refining the step size from coarse to fine. The two objectives map directly onto the yoly axes: $\sigma_{\mathrm{arch}}$ tracks R2 and $\eta_{\mathrm{arch}}$ tracks R1, so minimising both moves the operating point toward the origin, into the viable box $\sigma_{\mathrm{arch}} < \sigma_{R2}$ AND $\eta_{\mathrm{arch}} < \eta_{R1}$. The result is cross-validated by re-optimising the same bi-objective with an independent off-the-shelf optimiser (scipy's COBYLA); agreement between the DASA-native descent and scipy shows the method integrates with standard tooling. The search's selection runs on the clean flow model, so the persisted winner is noise-free; a separate step re-evaluates the chosen winner under one seeded input perturbation as a robustness confirmation, without re-searching.

**Inputs.** `data/config/catalogue/tas.json` (the service catalogue), `data/config/profile/opti.json` (the TAS skeleton from the aggregate scenario), and `data/config/method/dimensional.json`.

**Outputs.** The winner and search cloud at `data/results/search/winner/winner.json` and `data/results/search/winner/search.json`, the winner's dimensional envelope under `data/results/dimensional/winner/`, and the Pareto, overlay, and side-by-side figures under `data/img/search/`.

**Role.** The Validation piece in its constructive form, and the test of H3. Across the full candidate cloud the DASA bound predicate matches the analytic R1 AND R2 verdict on every candidate over a region containing both passes and fails. There is no equivalent CLI; this is a research notebook.

### 3.6 06-comparison.ipynb (Validation, decision)

**Idea.** Cross-method R1 and R2 triangulation, then place the search winner alongside the four hand-authored adaptations. This notebook reads the persisted analytic, stochastic, and dimensional envelopes from disk (plus the search winner) and renders the convergence and numerical-agreement bands that close the model-only chapter. It computes nothing new about the model; it decides H1 and H2 from what the methods already wrote.

**Inputs.** The per-method `network` and `nodes` envelopes and `requirements.json` verdicts under `data/results/{analytic,stochastic,dimensional}/<adaptation>/`, the search winner at `data/results/dimensional/winner/` and `data/results/search/winner/winner.json`, and the shared thresholds in `data/reference/baseline.json`.

**Outputs.** The cross-method PASS/FAIL grid at `data/img/comparison/<adaptation>/verdict_grid.{png,svg}`, the grouped metric bars at `data/img/comparison/metrics_bars.{png,svg}` (end-to-end response time and failure fraction across all five solutions, three method bars per group), and inline tables (the verdict matrix, the numerical agreement, the per-node contributor side-by-side, and the winner configuration).

**Role.** The Validation piece as the decision step. It is read-only: re-running the three method modules refreshes its cross-method inputs, and re-running 05 refreshes the winner. This is where the 24-cell congruence of H1 and the residual tolerances of H2 are reported.

---

## 4. Reproducing the runs

Two routes reproduce the runs. Both assume the project virtual environment is active and the PyDASA wheel pinned in `requirements.txt` is installed.

The three predictive methods each expose a command-line interface that writes the same envelopes the notebooks write. Run each method across the four adaptations:

```bash
python -m src.methods.analytic    --adaptation baseline
python -m src.methods.analytic    --adaptation s1
python -m src.methods.analytic    --adaptation s2
python -m src.methods.analytic    --adaptation aggregate

python -m src.methods.stochastic  --adaptation baseline
python -m src.methods.stochastic  --adaptation s1
python -m src.methods.stochastic  --adaptation s2
python -m src.methods.stochastic  --adaptation aggregate

python -m src.methods.dimensional --adaptation baseline
python -m src.methods.dimensional --adaptation s1
python -m src.methods.dimensional --adaptation s2
python -m src.methods.dimensional --adaptation aggregate
```

The `baseline` adaptation reads `dflt.json`; `s1`, `s2`, and `aggregate` read `opti.json` with the matching scenario, so no separate profile flag is required. The search notebook (05) and the comparison notebook (06) have no CLI; run them as notebooks.

To reproduce the full pipeline by executing the notebooks in order:

```bash
jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
```

Executing 01 through 03 refreshes the method envelopes under `data/results/`, 04 refreshes the design-space clouds, 05 refreshes the search winner, and 06 reads all of them back and renders the cross-method decision. Files under `data/results/` are regenerated by the methods and are never hand-edited.

---

## 5. Threats to the procedure

These are caveats of the experimental procedure itself; the full model-and-method threats to validity are in [report.md](report.md) Section 7.

1. **Shared-assumption ceiling.** The three predictive pipelines share the same M/M/c/K loss-network model, so their agreement is internal consistency (mutual confirmation under shared assumptions), not falsification against a live deployed system. The stochastic DES is an independent solver and could have rejected H1 and H2, which is what makes the congruence meaningful, but it does not measure reality.
2. **Model-based apparatus.** The apparatus is a simulation of the model, not an instrumented deployment. The procedure therefore establishes internal rigour and a constructive demonstration, not external falsification. Measuring a running TAS is a separate activity CS-1 does not perform.
3. **Atomic decomposition versus the source strategy.** Modelling S2 as the selection lever in isolation (no failover) is a deliberate atomic-experiment choice that departs from the source's bundled Select-Reliable strategy; the aggregate recombines retry and selection to correspond to the source strategy. The gain in lever attribution is bought at the cost of a literal per-cell match to the source.
4. **Pre-registered tolerances.** The H2 tolerances were fixed before the runs and are motivated by the M/M/c/K approximation budget and simulation noise. The procedure depends on this discipline: a tolerance loosened afterward to absorb any disagreement would strip the experiment of its falsifying force.

---

## References

The methodological sources named in this procedure are used at a high level; their full bibliographic entries live in the sibling documents.

- Pureur and Bittner, on architectural experimentation: the stance that a Quality Attribute Requirement is a falsifiable hypothesis, the three properties of an effective experiment (atomic, timely, unambiguous), and the four-piece Hypothesis / Model / Apparatus / Validation structure.
- Denning and Buzen (1978), the General Response Time Law, provenance for the end-to-end response-time relation used in Section 2; see [report.md](report.md).
- Weyns and Calinescu (2015) and Camara et al. (2023), the sources of the R1 (Availability) and R2 (Performance) thresholds; full entries in [case-study.md](case-study.md) and [report.md](report.md).
