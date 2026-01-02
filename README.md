# The Refusal Dial: Holdout-Validated Steering Vectors for Safety Refusal in Qwen2.5-0.5B (MATS Application)

This repository contains my MATS (ML Alignment & Theory Scholars) application submission under Neel Nanda.  
Project focus: **mechanistic interpretability of refusal behavior** in a small language model, using **activation patching** and **activation steering** to isolate and test a “refusal direction”.

---

## Problem Statement

Modern instruction-tuned models often refuse harmful requests — but they also sometimes **over-refuse** benign or “edge” requests.  
I wanted to test a concrete mechanistic hypothesis:

> **Hypothesis:** There exists a direction in the model’s residual stream that acts like a *refusal propensity feature*: adding it increases refusal-like behavior, subtracting it decreases it.

If true, we should be able to:
1) **extract** a direction from activations using data-driven differences,  
2) **steer** the model by adding that direction at a specific layer/site, and  
3) **validate** that the effect is not an artifact (controls + generalization tests).

---

## Core Idea

We intervene by adding a vector to activations:  
> $\alpha$ is the scalar steering strength that multiplies the learned direction before adding it to the residual stream.

---

## What I Did

### 1) Build a scoring signal for “refusal-ness”
- Ran baseline model generations on a curated prompt set with two main subsets:
  - **HARMFUL**: prompts that should be refused
  - **EDGE**: borderline/benign prompts where over-refusal is undesirable
- Computed a per-prompt “refusal score” (see `experiments/refusal_score.py`) and saved to:
  - `runs/qwen25_scores.csv`

### 2) Locate where refusal behavior is represented (patching)
- Used activation patching across layers/sites to identify where interventions have the largest causal effect on refusal scoring.
- Key scripts:
  - `experiments/patch_sites_qwen25.py`
  - `experiments/patch_resid_pre_qwen25.py`
  - `experiments/patch_lastK_qwen25.py`
  - `experiments/patch_lastK_mlp_out_qwen25.py`

### 3) Extract a refusal direction and steer
- Built a “refusal direction” using **top vs bottom** EDGE prompts ranked by refusal score (balanced selection).
- Injected the direction at `hook_resid_pre` at the final token position (`pos=-1`), sweeping \(\alpha\) over a grid.
- Key scripts:
  - `experiments/steer_refusal_direction_qwen25.py`
  - `experiments/steering_balanced_qwen25.py`
  - `experiments/steering_balanced_dump_scores_qwen25.py` (exact per-example dumps; no approximations)

### 4) Controls (to rule out direction overfit / generic vector effects)
**A) Holdout-direction control**
- Learn the direction using only a random half of EDGE prompts (train split), then evaluate on EDGE-holdout and HARMFUL.
- Script:
  - `experiments/steering_holdout_control_qwen25.py`

**B) Random-direction baseline**
- Replace learned direction with random unit vectors at the same layer/site (many seeds), to measure “typical” steering effects.
- Script:
  - `experiments/steering_random_baseline_qwen25.py`

### 5) Generalization check on an external “edge-case” set (XSTest)
- Evaluated the holdout direction on XSTest categories (450 prompts) to quantify benign over-refusal vs steering strength.
- Script:
  - `experiments/eval_direction_on_xstest_qwen25.py`

---

## Experiments Summary

### Steering experiments
- Layers tested: **20, 21, 23**
- Site: **`hook_resid_pre`**
- \(\alpha\) sweep included negative and positive strengths (e.g. `-10 ... 6`)
- Logged both:
  - **refusal-score means** and
  - **refusal rates** (thresholded)

### Controls
- Holdout direction learned on EDGE-train only, evaluated on EDGE-holdout and HARMFUL.
- Random baselines across many seeds (e.g. 20) to get mean + variability bands.

### External evaluation
- XSTest refusal rates broken down by category and \(\alpha\).

---

## Main Results (high-level)

1) **Steering works**: increasing $\alpha$ systematically increases refusal behavior, decreasing $\alpha$ reduces it.  
2) The effect is **not just “any vector”**:
   - The **holdout-learned** direction produces stronger and more consistent separation than random unit vectors.
3) The direction behaves like a **global refusal propensity dial**:
   - It increases refusal on harmful prompts **and also** increases refusal on edge/benign prompts (over-refusal).
4) **Layer matters** (20/21/23 behave similarly but differ in strength/shape of tradeoff curves).

---

## Conclusions

- There is strong empirical evidence for a **linearly steerable refusal feature** in the residual stream of Qwen2.5-0.5B.
- However, the feature appears **entangled** with general refusal/hedging behavior: it’s not cleanly “harm-only”.
- The most important mechanistic takeaway is the existence of a **causal, transferable direction** (holdout generalizes) with measurable separation beyond random baselines.

---

## Future Directions

1) **Decompose the direction into circuits**
   - Which attention heads / MLP neurons contribute most to the refusal direction?
   - Use attribution / projection / head ablations / neuron knockout.

2) **Localize to token positions**
   - Does steering at earlier positions vs last-token change over-refusal patterns?

3) **Conditioned steering**
   - Learn multiple directions (harm-specific vs politeness/deflection) and test subspace separation.

4) **Better generalization benchmarks**
   - Extend beyond XSTest: other jailbreak/edge datasets; measure helpfulness degradation.

---

## Repository Structure

- `data/`  
  Prompt sets (`prompts.jsonl`) and XSTest (`xstest_prompts.csv`).

- `experiments/`  
  Core scripts for scoring, patching, steering, controls, and XSTest evaluation.

- `runs/`  
  JSON/CSV outputs of runs, plus generated plots.

- `plots/`  
  Main paper-style plots (patching, histograms, heatmaps, etc.).

- `scripts/`  
  Plotting utilities and result-comparison tools.

- `logs/`  
  Logs for all experiments.

---

## How to Reproduce (typical commands)

Most experiments are run from repo root. Example patterns:

```bash
python experiments/score_models_qwen25.py ...
python experiments/patch_sites_qwen25.py ...
python experiments/steering_balanced_qwen25.py --layer 20 --site hook_resid_pre ...
python experiments/steering_holdout_control_qwen25.py --layer 20 --site hook_resid_pre ...
python experiments/steering_random_baseline_qwen25.py --layer 20 --site hook_resid_pre ...
python experiments/eval_direction_on_xstest_qwen25.py --layer 20 --site hook_resid_pre ...
python scripts/plot_controls_pareto.py ...
python scripts/plot_holdout_random_controls.py ...
