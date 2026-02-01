# Prox-Informed SAM

This repo explores **continual learning** through a decoupled lens: (1) _process stability at task switches_ and (2) _end-state robustness / long-term retention_. It implements and evaluates **Replay (ER)**, **Proximal Regularization (Prox)**, **Sharpness-Aware Minimization (SAM)**, and a unified method **Prox-Informed SAM** that _binds_ SAM’s perturbation radius to the current stability constraint.

## Motivation

In continual learning, “forgetting” is not a single phenomenon:

- **Switch-time collapse** can happen immediately after a task boundary (the model gets “shocked” and needs to recover), even if the final accuracy later looks acceptable.
- **Long-term forgetting** measures accumulated end-state damage after learning all tasks.

This project proposes to _measure and optimize these two axes separately_:

- **Stability Gap (SG)**: area-under-drop in a fixed recovery window after each task switch (plus diagnostics: **first/peak/last drop**, recovery time, recovery rate).
- **Long-term Forgetting (F)**: how much each past task degrades from its best pre-switch accuracy to the final model.

**Hypothesis**:

- Prox improves **SG** by constraining parameter drift at switches,
- SAM mainly improves **F** by selecting flatter minima that resist cumulative interference.
- If their geometry is aligned (shared metric / trust-region scale), SAM can improve robustness without “fighting” Prox’s stability constraint, yielding a better Acc/F vs. SG tradeoff than naive stacking.

**Key Idea**:

> To **align their geometry** by making SAM’s radius depend on how strongly Prox is constraining updates at that moment.

## Prox-Informed SAM: Gradient-ratio coupling

Let task gradient be $g_t=\nabla L_{\text{task}}(\theta)$ on the SAM batch, and Prox gradient be $p_t=\nabla L_{\text{prox}}(\theta)$, then:

$$
r_t=\frac{\|p_t\|}{\|g_t\|+\epsilon},
\quad
\rho_t = \text{clip}\Big(\frac{\rho_{\max}}{1+\beta r_t},\ \rho_{\min},\ \rho_{\max}\Big).
$$

**Intuition**  
Proximal regularization acts like a _stability fence_: when the current parameters have drifted away from an anchor (e.g., the weights saved at the last task switch), its gradient $p_t$ becomes large and “pulls back” toward that anchor. SAM, on the other hand, injects an adversarial perturbation of size $\rho$ to seek flatter solutions. If we keep $\rho$ fixed while Prox is strongly constraining updates, SAM wastes effort exploring directions that Prox will immediately undo and can even increase switch-time shock.

The ratio

$$
r_t=\frac{\|p_t\|}{\|g_t\|+\epsilon}
$$

is a dimensionless measure of _constraint dominance_: when $r_t$ is large, the update is mostly governed by stability (Prox); when $r_t$ is small, the task gradient dominates and we can afford a larger SAM perturbation.

**Design choices**

- **Rational shrinkage $\rho_{\max}/(1+\beta r_t)$**: keeps $\rho_t$ smooth and monotone in $r_t$. $\beta$ controls sensitivity: larger $\beta$ shrinks faster as Prox dominates.
- **Clipping to $[\rho_{\min},\rho_{\max}]$**: avoids pathological extremes (e.g., $\rho_t\to 0$ when $g_t$ is tiny, or $\rho_t>\rho_{\max}$ early in training) and makes the method behave like vanilla SAM when Prox is weak.
- **$\epsilon$ in the denominator**: prevents division blow-ups and stabilizes behavior when $\|g_t\|$ is near zero.

## Experiment design

### Benchmark and baseline

- **Dataset**: Split CIFAR-100 with frequent task switches (**10 tasks × 10 classes**).
- **Baseline**: **Experience Replay (ER)** with task-balanced sampling from a fixed-size FIFO buffer.

### Metrics

We use two complementary families of metrics: **switch-time stability** (how badly a task “shocks” older tasks right after a boundary) and **end-state robustness** (how much is retained at the end).

Let tasks be indexed by $t=1,\dots,T$. Let $s_t$ be the global step at the moment we switch into task $t$ (so $s_t^-$ / $s_t^+$ denote “just before/after the switch”). Let $A_k(\cdot)$ denote validation accuracy on task $k$ (in %, evaluated periodically during training). Let $W$ be the _post-switch_ recovery window (in steps).

#### End-state robustness

- **Long-term Forgetting (F)** (per task $k$):

$$
F_k = \max_{t\ge k} A_k(s_t^-) - A_k(s_T^+),
\qquad
F = \frac{1}{T}\sum_{k=1}^T F_k.
$$

Intuition: how much each task drops from its best pre-switch accuracy to the final model.

- **Learning Success (LS)** (classic accuracy-matrix diagonal average):

$$
LS = \frac{1}{T}\sum_{t=1}^T A_t(s_t^+).
$$

- **Retention Ratio**:

$$
\text{RR} = \frac{\frac{1}{T}\sum_{k=1}^T A_k(s_T^+)}{LS}.
$$

#### Switch-time stability

- **Stability Gap (SG)** (_area-under-drop_ in a fixed window after each switch):

$$
\text{SG}^{\text{area}}_{t}
=\frac{1}{t-1}\sum_{k<t}\frac{1}{W}\sum_{j=s_t}^{s_t+W}\big[A_k(s_t^-)-A_k(j)\big]_+,
\qquad [x]_+=\max(0,x).
$$

In code we compute the same quantity from discrete evaluation points (piecewise-constant integration) and then average across switches/tasks.

- **Drop diagnostics** (for each past task $k$ after switch at $s_t$; reference is $A_k(s_t^-)$):
  - **First Drop**: $\max(0, A_k(s_t^-)-A_k(\text{first eval after }s_t))$
  - **Peak Drop**: $\max(0, A_k(s_t^-)-\min A_k(\cdot)\ \text{in the post-switch segment})$
  - **Last Drop**: $\max(0, A_k(s_t^-)-A_k(\text{last eval before next switch}))$

### Ablation groups

1. **Replay** (ER only).
2. **Replay + Prox**: scan `lambda_prox` (Prox-only sweep).
3. **Replay + SAM**: scan fixed `rho_base` (SAM-only sweep).
4. **Naive Replay + Prox + SAM**: grid search `lambda_prox × rho_base`.
5. **Unified (Ours): Prox-Informed SAM**: bind a dynamic radius `rho_t` to the current stability constraint.

## Results and Analysis

### Key results summary

**Mechanism isolation holds, synergy not yet achieved**:

- Prox behaves like a “stability fence” only when strong (but then plasticity collapses)
- SAM reduces forgetting and boosts end-state accuracy, but consistently increases switch-time shock (SG / First / Peak).
- Naive Prox+SAM and the current Prox-Informed coupling improve Acc/F but still worsen SG. No clear Pareto-improving point in the present setup.

**Limitations / next steps**:

- current `grad_ratio` signal is too narrow within-run → `rho_t` is nearly constant;
- the coupling uses Euclidean norms while the proposal calls for a unified metric (e.g., Adam-diagonal/Fisher-diagonal) and the corresponding SAM perturbation;

### Replay

| Buffer size | Avg Acc (%) | Forgetting (%) | Stability Gap (%) | First Drop (%) | Peak Drop (%) | Last Drop (%) | Learning Success (%) | Retention Ratio | Notes                                                                       |
| :---------- | ----------: | -------------: | ----------------: | -------------: | ------------: | ------------: | -------------------: | --------------: | :-------------------------------------------------------------------------- |
| **1000**    |       13.43 |          68.11 |         **11.32** |          14.89 |         16.83 |         13.89 |                74.73 |           0.180 | **Stable-but-forgetful**: smallest switch shock, worst final retention.     |
| **2000**    |       18.80 |          62.17 |             13.26 |          17.45 |         19.28 |         12.91 |            **74.75** |           0.252 | **Balanced baseline**: better retention with still relatively low SG.       |
| **5000**    |       25.35 |          54.28 |             17.32 |          22.79 |         24.51 |         11.75 |                74.20 |           0.342 | **Retention jump**: much lower forgetting, but worse switch shock/recovery. |
| **10000**   |   **27.08** |      **51.83** |             19.41 |          25.53 |         26.76 |         11.43 |                73.73 |       **0.367** | **Diminishing returns**: retention improves, stability worsens.             |

**Key insights**

- **Plasticity stays roughly constant**: Learning Success remains ~73–75%, so buffer size mostly affects retention rather than “can the model learn the current task”.
- **Retention increases monotonically with buffer**: Avg Acc and Retention Ratio rise, while Forgetting decreases (1000→10000: 68.11% → 51.83%).
- **Switch shock increases with buffer**: SG and First/Peak Drop increase (1000→10000: First 14.89→25.53; Peak 16.83→26.76; SG 11.32→19.41).
- **End-of-window drop decreases**: Last Drop decreases (13.89→11.43), suggesting larger buffers help return to higher accuracy at the end, but with larger transient shock.
- **After 5000, gains diminish**: 5000→10000 improves Avg Acc modestly (25.35→27.08) but stability degrades further.

**Default**: subsequent experiments use `buffer_size=5000` as the baseline.

### Prox-Only

Setup: ER baseline with `buffer_size=5000`, Prox enabled, SAM disabled.

| Config   | Avg Acc (%) | LS (%) | Retention Ratio | Forgetting (%) |   SG (%) | Peak Drop (%) | First Drop (%) | Last Drop (%) |
| :------- | ----------: | -----: | --------------: | -------------: | -------: | ------------: | -------------: | ------------: |
| Baseline |       25.35 |  74.20 |           0.342 |          54.28 |    17.32 |         24.51 |          22.79 |         11.75 |
| λ=0.01   |       25.75 |  73.58 |           0.350 |          53.14 |    17.67 |         25.16 |          23.25 |         11.54 |
| λ=0.03   |   **26.07** |  72.97 |       **0.357** |      **52.11** |    18.10 |         26.10 |          23.82 |         11.22 |
| λ=0.07   |       24.32 |  72.28 |           0.337 |          53.28 |    18.95 |         27.14 |          24.94 |         11.82 |
| λ=0.10   |       22.35 |  71.60 |           0.312 |          54.72 |    18.13 |         26.73 |          23.85 |         12.05 |
| λ=0.30   |       16.40 |  66.12 |           0.248 |          55.25 |    15.31 |         22.71 |          20.14 |         11.86 |
| λ=0.70   |       12.85 |  62.27 |           0.206 |          54.92 |    13.71 |         19.98 |          18.04 |         11.66 |
| λ=1.00   |       11.57 |  60.88 |           0.190 |          54.78 |    13.08 |         18.91 |          17.22 |         11.61 |
| λ=2.00   |       10.05 |  58.00 |           0.173 |          53.28 |    11.93 |         17.00 |          15.70 |         11.12 |
| λ=5.00   |        8.78 |  55.80 |           0.157 |          52.25 |    10.90 |         15.51 |          14.34 |         10.89 |
| λ=15.0   |        7.75 |  54.10 |           0.143 |          51.50 |     9.50 |         13.92 |          12.50 |         10.69 |
| λ=40.0   |        7.40 |  53.55 |           0.138 |          51.28 | **9.04** |     **13.19** |          11.89 |         10.62 |

**Key observations**

- **Weak Prox (λ≈0.01–0.03)** slightly improves end-state metrics but makes stability worse:
  - Example λ=0.03 vs baseline: Avg Acc **+0.73**, Retention Ratio **+0.016**, Forgetting **−2.17**, but SG **+0.78** and Peak Drop **+1.59** (and First Drop increases: 22.79→23.82).
- **Mid Prox (λ≈0.07–0.10) is dominated**: worse Avg Acc and worse stability.
- **As λ increases, stability improves monotonically, but plasticity collapses**:
  - From λ=0.3 onward, SG/Peak Drop decrease (15.31/22.71), but Avg Acc collapses (16.40).
  - At λ=40, stability is best (SG 9.04, Peak 13.19) but Avg Acc is only 7.40 and LS is 53.55.
- **Lower “forgetting” at large λ is not stronger memory**: it is consistent with underfitting (“retain more by learning less”).
- **Conclusion**: in this setup, Prox-only has no usable sweet spot that reduces SG without a major accuracy cost; Prox is better treated as a constraint component that needs a complementary plasticity-preserving mechanism (e.g., SAM).

**Expectation check (“Prox reduces SG, forget decreases slightly”)**

Partially true but not usable: SG decreases at large λ and forgetting decreases only slightly, but this is accompanied by severe plasticity collapse (LS/Avg Acc drop sharply).

### SAM-Only

| Config            | Avg Acc (%) |    LS (%) | Retention Ratio | Forgetting (%) |    SG (%) | First Drop (%) | Peak Drop (%) | Last Drop (%) | Notes                                              |
| :---------------- | ----------: | --------: | --------------: | -------------: | --------: | -------------: | ------------: | ------------: | :------------------------------------------------- |
| Baseline (no SAM) |       25.35 |     74.20 |           0.333 |          54.28 |     17.32 |          22.79 |         24.51 |         11.75 | Reference ER.                                      |
| ρ=0.01            |       31.13 |     68.23 |           0.340 |          41.22 |     20.20 |          26.58 |     **29.08** |          8.51 | Strong gain (Acc↑, F↓), but SG/Peak↑.              |
| ρ=0.015           |       30.10 |     67.78 |           0.339 |          41.86 |     20.30 |          26.71 |         28.80 |          8.68 | Strong gain.                                       |
| ρ=0.02            |       30.48 |     68.15 |           0.340 |          41.86 |     20.07 |          26.41 |         28.54 |          8.99 | Strong gain.                                       |
| ρ=0.03            |   **31.63** |     68.63 |       **0.341** |      **41.11** |     20.06 |          26.39 |         28.78 |          8.75 | Best “balanced” before LS collapses.               |
| ρ=0.05            |       30.48 |     68.63 |           0.343 |          42.39 |     20.19 |          26.57 |         28.64 |          8.81 | Strong gain, slight F rebound.                     |
| ρ=0.07            |       31.50 | **68.80** |           0.343 |          41.44 | **20.33** |          26.76 |         28.52 |          8.76 | Close to ρ=0.03.                                   |
| ρ=0.09            |       31.33 |     68.75 |           0.341 |          41.58 |     20.07 |          26.41 |         28.27 |          8.88 | Strong gain.                                       |
| ρ=0.10            |       30.53 |     68.35 |           0.342 |          42.03 |     20.01 |          26.33 |         28.61 |          8.74 | Strong gain.                                       |
| ρ=0.50            |       29.65 |     65.43 |           0.334 |          39.75 |     19.02 |          25.02 |         27.23 |          8.73 | Plasticity starts dropping.                        |
| ρ=1.00            |       28.45 |     54.33 |           0.319 |          30.47 |     19.19 |          25.24 |         27.52 |          6.62 | Plasticity collapse: “lower F” from learning less. |
| ρ=2.00            |       26.20 |     40.40 |           0.301 |          21.72 |     19.53 |          25.69 |         27.23 |          4.90 | Severe collapse.                                   |
| ρ=5.00            |       32.45 |     32.68 |           0.300 |           9.47 |     19.88 |          26.16 |         27.91 |          2.45 | Extreme degenerate regime.                         |

**Key observations**

- **SAM-only produces a large end-state improvement** in this setup:
  - Baseline Avg Acc 25.35 / F 54.28 → ρ=0.01–0.10 gives Avg Acc ~30–31.6 and F ~41–42.
  - The cost is **worse switch-time stability**: SG/Peak/First Drop increase (17.32/24.51/22.79 → ~20/28–29/26–27).
- **Balanced sweet spot** is roughly **ρ≈0.03–0.07** (good Avg Acc + low F without catastrophic LS drop).
- **Large ρ can make “forgetting” look lower due to plasticity collapse**:
  - ρ=1–2 reduces F further but LS collapses (≈54→40), consistent with “learn each task more shallowly”.
  - ρ=5.0 is an extreme degenerate regime (high Avg Acc but very low LS and near-zero forgetting), which must be filtered using LS or AUC-like plasticity constraints.
- **Why `sam_phase=last` is more stable**: applying SAM late (low-LR stage) behaves more like selecting a flatter final solution, rather than suppressing early task learning.

**Expectation check (“SAM reduces F, SG oscillates”)**

Partially met (signal stronger than expected): F drops clearly in ρ≈0.01–0.10, and final accuracy increases; SG does not “oscillate” but rises fairly consistently, behaving like a systematic tradeoff.

### Prox + SAM

| Config          | Avg Acc (%) | LS (%) | Retention Ratio | Forgetting (%) |    SG (%) | Peak Drop (%) | First Drop (%) | Last Drop (%) | Notes                                                         |
| :-------------- | ----------: | -----: | --------------: | -------------: | --------: | ------------: | -------------: | ------------: | :------------------------------------------------------------ |
| Baseline (ER)   |       25.35 |  74.20 |           0.342 |          54.28 |     17.32 |         24.51 |          22.79 |         11.75 | Reference.                                                    |
| λ=0.03, ρ=0.015 |       29.35 |  67.42 |           0.435 |          42.31 |     20.93 |         30.10 |          27.54 |          9.10 | Acc/F ↑, stability worse.                                     |
| λ=0.03, ρ=0.05  |       29.50 |  67.42 |           0.438 |          42.14 | **20.37** |     **29.49** |          26.80 |          8.91 | Most stable within λ=0.03 subset (still worse than baseline). |
| λ=0.03, ρ=0.10  |   **30.12** |  67.38 |       **0.447** |      **41.39** |     20.58 |         29.65 |          27.08 |          8.93 | Best Acc/F within λ=0.03 subset (stability worse).            |
| λ=0.30, ρ=0.015 |       21.98 |  58.30 |           0.377 |      **40.36** |     17.82 |         26.32 |          23.44 |          8.69 | Stability ~ baseline, but LS collapses → Acc drops.           |
| λ=0.30, ρ=0.05  |       21.30 |  57.65 |           0.369 |          40.39 |     17.12 |         25.44 |          22.52 |          8.66 | SG slightly lower, Acc still much worse.                      |
| λ=0.30, ρ=0.10  |       20.20 |  57.08 |           0.354 |          40.97 | **16.39** |     **24.35** |          21.57 |          8.81 | Closest SG/Peak to baseline, but largest Acc cost.            |

**Key observations (what we actually learned)**

- **Naive combination can strongly reduce F and raise final accuracy, but stability gets worse** (especially for λ=0.03):
  - Avg Acc 25.35 → 29.35–30.12 (+4.0–+4.8pt), F 54.28 → 41.39–42.31 (−12–−13pt), RR 0.342 → 0.435–0.447.
  - But SG 17.32 → 20.37–20.93 (+~3pt), Peak Drop 24.51 → 29.49–30.10, First Drop also rises.
- **No visible “true synergy point”** in this grid: no configuration achieves Acc↑/F↓ _and_ SG↓ simultaneously.
- **Within the λ=0.03 subset**, differences are small: ρ=0.10 best Avg Acc/F; ρ=0.05 slightly better SG/Peak but still worse than baseline.
- **The λ=0.30 subset behaves like “stability fence + plasticity loss”**: SG decreases (17.82→16.39) while LS drops (58.3→57.1), leading to much worse Avg Acc.

### Prox-Informed SAM

| Config       |    λ |   β | Avg Acc |    LS | Retention Ratio | Forgetting |    SG | Peak Drop | First Drop | Last Drop | ρ (median/p90) | r (median/p90) |
| :----------- | ---: | --: | ------: | ----: | --------------: | ---------: | ----: | --------: | ---------: | --------: | :------------- | :------------- |
| Baseline     |    - |   - |   25.35 | 74.20 |           0.342 |      54.28 | 17.32 |     24.51 |      22.79 |     11.75 | -              | -              |
| λ=0.03, β=1  | 0.03 |   1 |   26.45 | 61.85 |           0.428 |      39.44 | 18.50 |     26.68 |      24.34 |      8.57 | 0.480/0.486    | 0.043/0.053    |
| λ=0.03, β=4  | 0.03 |   4 |   27.25 | 63.40 |           0.430 |      40.17 | 18.93 |     27.14 |      24.91 |      8.73 | 0.429/0.449    | 0.041/0.052    |
| λ=0.03, β=8  | 0.03 |   8 |   27.62 | 64.12 |           0.431 |      40.56 | 18.55 |     27.38 |      24.41 |      8.79 | 0.376/0.407    | 0.041/0.052    |
| λ=0.03, β=16 | 0.03 |  16 |   27.70 | 64.85 |           0.427 |      41.28 | 19.05 |     27.84 |      25.06 |      8.86 | 0.310/0.344    | 0.038/0.049    |
| λ=0.30, β=4  | 0.30 |   4 |   15.45 | 51.53 |           0.300 |      40.08 | 14.15 |     20.79 |      18.62 |      8.61 | 0.323/0.350    | 0.137/0.164    |
| λ=0.30, β=16 | 0.30 |  16 |   17.73 | 55.03 |           0.322 |      41.44 | 15.17 |     22.46 |      19.96 |      8.89 | 0.176/0.204    | 0.115/0.136    |

**Key observations**

- **At λ=0.03, `grad_ratio` behaves almost like a fixed-ρ sweep**: $r_t$ is narrow (median ~0.038–0.043), so within a run $\rho_t$ forms a near-constant band; increasing β mainly shifts that band downward (consistent with $\rho\approx 0.5/(1+\beta\cdot 0.04)$).
- **The same three-way tradeoff appears (not the desired synergy)**:
  - End-state improves: Avg Acc +1.10–+2.35, Forgetting −13.00–−14.83, Last Drop −2.89–−3.18.
  - Peak learning is suppressed: LS drops 74.20 → 61.85–64.85.
  - Switch shock is not offset: SG +1.17–+1.72, Peak Drop +2.17–+3.33, First Drop +1.54–+2.27.
- **β has consistent directionality**: β↑ ⇒ ρ↓ ⇒ plasticity improves, forgetting slightly worsens, and switch shock slightly worsens (typical ρ-sweep shape).
- **Strong Prox (λ=0.3) increases r and shrinks ρ as designed, but overall enters plasticity collapse** (Avg Acc 15.45–17.73), consistent with the Prox-only conclusion.
- **Next step for true “switch-time automatic shrink”**: make $r_t$ vary _within a run_ (not just across λ), e.g., include replay gradients in $r_t$ consistently with `sam_batch_mode`, normalize norms under Adam-like preconditioning, or use a task-aware λ schedule to create genuine “Prox-dominant” phases.

## Backend Model

### Architecture Overview

We use a **ResNet-18** backbone as the feature extractor, paired with a linear classifier head that expands dynamically as new tasks arrive. This capacity choice balances several constraints specific to continual learning:

- **Expressivity vs. forgetting tradeoff**: Larger models can fit more tasks but also accumulate more catastrophic interference. ResNet-18 (~11M parameters) provides sufficient capacity for Split CIFAR-100 (10 tasks × 10 classes) while remaining sensitive to forgetting effects.
- **Computational efficiency**: The ablation study involves sweeping `lambda_prox`, `rho_base`, and their combinations—requiring hundreds of runs. ResNet-18 enables sufficient iteration speed.

Compared to deeper variants (ResNet-34/50), ResNet-18 serves as a "stress test" for continual learning methods: if Prox/SAM cannot help here, they are unlikely to scale to harder settings.

### CIFAR-Specific Adaptations

The ResNet implementation follows standard CIFAR-10/100 adaptations from the literature:

- **Initial conv layer**: Modified from `7×7, stride 2` to `3×3, stride 1`—CIFAR's 32×32 images cannot afford aggressive spatial reduction early in the network.
- **No initial pooling**: The first `maxpool` layer from ImageNet ResNets is removed to preserve spatial resolution for small inputs.
- **Batch Normalization**: Retained throughout, as BN provides stability during the adversarial perturbations used by SAM.

These modifications are standard for CIFAR experiments but worth noting explicitly: they affect the **effective receptive field** and **gradient flow**, which in turn influences how Prox constraints and SAM perturbations propagate through the network. The classifier head receives a 512-channel feature map after global average pooling (output size 4×4 → reduced to 1×1).

### Dynamic Classifier Expansion

Continual learning introduces new classes at each task boundary, requiring the classifier head to grow. The `IncrementalResNet` wrapper implements this via:

```python
def expand_classifier(self, n_new_classes):
    old_head = self.backbone.classifier
    old_weights = old_head.weight.data.clone()
    old_bias = old_head.bias.data.clone()

    new_total_classes = self.n_classes + n_new_classes
    new_head = nn.Linear(in_features, new_total_classes)

    # Preserve existing weights/bias, new rows are randomly initialized
    new_head.weight.data[:self.n_classes] = old_weights
    new_head.bias.data[:self.n_classes] = old_bias

    self.backbone.classifier = new_head
    self.n_classes = new_total_classes
```
