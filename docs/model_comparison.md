# Model Comparison

Four diffusion-based offline RL architectures. All trained with DDPM (predict noise ε) and sampled with DDIM (20 inference steps from 100 training steps).

---

## 1. Action CNN U-Net - `model.type: action`

**Paper:** Chi et al., 2023 - *Diffusion Policy* (CNN variant)  
**Code:** `src/diffplan/models/diffusion_unet1d.py`  
**Sampler:** `DiffusionSampler`  
**Parameters:** ~11M

| Property | Value |
|---|---|
| Denoises | Action sequences `(B, H, 6)` |
| Distribution | $p(A \mid S)$ - generates actions, conditioned on current state |
| Backbone | 1D temporal U-Net (Conv1d along horizon axis) |
| State conditioning | FiLM (Feature-wise Linear Modulation: γ·x + β) |
| Timestep conditioning | Sinusoidal embedding, merged with obs into a single conditioning vector |
| Architecture | Encoder: H → H/2 → H/4. Bottleneck. Decoder with skip connections. Channels: 128 → 256 → 512 |

### Data flow
1. Observation `(17)` → StateEncoder MLP → conditioning embedding
2. Diffusion timestep k → sinusoidal embedding
3. Both concatenated → single conditioning vector
4. Conditioning vector injected into every ResBlock via FiLM (γ·x + β)
5. Noisy actions `(B, 32, 6)` → Conv1d U-Net → predicted noise `(B, 32, 6)`

### Characteristics
- Convolutional inductive bias produces temporally smooth action sequences
- Robust default hyperparameters
- Low-pass bias can over-smooth high-frequency action changes

---

## 2. Diffusion Transformer - `model.type: transformer`

**Paper:** Chi et al., 2023 - *Diffusion Policy* (Transformer variant)  
**Code:** `src/diffplan/models/diffusion_transformer.py`  
**Sampler:** `DiffusionSampler`  
**Parameters:** ~8.6M

| Property | Value |
|---|---|
| Denoises | Action sequences `(B, H, 6)` |
| Distribution | $p(A \mid S)$ - generates actions, conditioned on current state |
| Backbone | Transformer decoder (8 layers, 256-dim, 4 heads) |
| State conditioning | Cross-attention (obs embeddings as keys/values) |
| Timestep conditioning | Sinusoidal embedding → projection → prepended as first token |

### Data flow
1. Observation `(17)` → MLP → obs embeddings `(B, 1, 256)`
2. Diffusion timestep k → sinusoidal embedding → linear projection → prepended token
3. Noisy actions `(B, H, 6)` → linear projection → action tokens
4. 8× decoder blocks: causal self-attention → cross-attention on obs embeddings → FFN
5. Output tokens → linear head → predicted noise `(B, H, 6)`

### Characteristics
- No convolutional low-pass bias - each action token attends to all others
- Handles high-frequency and sharp action changes well
- More sensitive to hyperparameters (dropout, weight decay, layer count)

---

## 3. Trajectory U-Net - `model.type: trajectory`

**Paper:** Janner et al., 2022 - *Planning with Diffusion for Flexible Behavior Synthesis*  
**Code:** `src/diffplan/models/trajectory_unet1d.py`, `guided_sampler.py`, `value_function.py`  
**Sampler:** `GuidedDiffusionSampler`  
**Parameters:** ~11M (U-Net) + ~411K (value function)

| Property | Value |
|---|---|
| Denoises | Full trajectories `(B, H, 23)` - concatenated `[state(17) \| action(6)]` per step |
| Distribution | $p(S, A)$ - generates states and actions together (no conditioning input) |
| Backbone | 1D temporal U-Net (Conv1d along horizon axis) |
| State conditioning | Inpainting: overwrite `x[:, 0, :17]` with noised current obs at each denoising step |
| Timestep conditioning | Sinusoidal embedding, additive into ResBlocks |
| Reward steering | Learned value function $V(x, t)$ provides gradient guidance: $\nabla_x V$ added to trajectory at each step |

### Data flow
1. Noisy trajectory `(B, 32, 23)` - each step is `[state(17) | action(6)]`
2. Inpainting: first state position overwritten with appropriately noised current observation
3. Diffusion timestep k → sinusoidal embedding → additive conditioning in ResBlocks
4. Conv1d U-Net → predicted noise `(B, 32, 23)`
5. (Optional) ValueFunction evaluates trajectory → gradient ∇V steers toward high-return regions

### Characteristics
- Learns environment dynamics implicitly (generates future states)
- Can visualize planned trajectories
- Value function guidance steers generation without retraining the diffusion model
- Requires separate value function training
- Higher-dimensional denoising (23 channels vs 6)

---

## 4. Decision Diffuser - `model.type: decision`

**Paper:** Ajay et al., 2023 - *Is Conditional Generative Modeling All You Need for Decision-Making?* (ICLR 2023)  
**Code:** `src/diffplan/models/state_unet1d.py`, `inverse_dynamics.py`, `decision_diffuser_sampler.py`  
**Sampler:** `DecisionDiffuserSampler`  
**Parameters:** ~11M (U-Net) + ~0.5M (inverse dynamics)

| Property | Value |
|---|---|
| Denoises | State sequences only `(B, H, 17)` - no actions in diffusion process |
| Distribution | $p(S \mid R)$ - generates states, conditioned on desired return |
| Backbone | 1D temporal U-Net (Conv1d along horizon axis) |
| State conditioning | Inpainting: overwrite first state with noised current obs at each denoising step |
| Timestep conditioning | Sinusoidal embedding → MLP → additive into ResBlocks |
| Return conditioning | Normalized return $R \in [0,1]$ → MLP → return embedding, concatenated with timestep embedding |
| Action extraction | Inverse dynamics model $f_\phi(s_t, s_{t+1}) \to a_t$ |
| Reward steering | Classifier-free guidance (CFG) with 25% conditioning dropout during training |

### Classifier-free guidance

Two forward passes per denoising step:

$$\hat{\varepsilon} = \varepsilon_\theta(x, \varnothing, k) + \omega \left(\varepsilon_\theta(x, R, k) - \varepsilon_\theta(x, \varnothing, k)\right)$$

Low-temperature sampling scales variance by $\alpha \in [0, 1)$ for higher-likelihood state sequences.

### Data flow
1. Return R → MLP → return embedding; timestep k → sinusoidal embedding → MLP → time embedding
2. Both concatenated → conditioning vector → additive injection into every ResBlock
3. Inpainting: first state position overwritten with appropriately noised current observation
4. Noisy states `(B, 32, 17)` → Conv1d U-Net → predicted noise `(B, 32, 17)`
5. CFG: conditional + unconditional forward passes combined with guidance scale ω
6. Denoised consecutive state pairs → inverse dynamics $f_\phi(s_t, s_{t+1})$ → actions `(B, 31, 6)`

### Characteristics
- No value function or Q-function required - CFG replaces gradient guidance
- Conditions can be composed at test time (return + constraints + skills) via score composition
- Avoids diffusing over high-frequency action signals
- Requires separately trained inverse dynamics model
- 2× inference cost per denoising step (dual forward passes for CFG)
- Assumes near-deterministic state transitions for inverse dynamics accuracy

---

## Comparison Table

| | Action CNN | Transformer | Trajectory | Decision Diffuser |
|---|---|---|---|---|
| Config | `action` | `transformer` | `trajectory` | `decision` |
| Denoises | Actions `(H, 6)` | Actions `(H, 6)` | States+Actions `(H, 23)` | States `(H, 17)` |
| Distribution | $p(A \mid S)$ | $p(A \mid S)$ | $p(S, A)$ | $p(S \mid R)$ |
| Backbone | Conv1d U-Net | Transformer decoder | Conv1d U-Net | Conv1d U-Net |
| State conditioning | FiLM | Cross-attention | Inpainting | Inpainting |
| Action source | Direct output | Direct output | Extracted from trajectory | Inverse dynamics |
| Reward steering | Simulate-and-score | Simulate-and-score | Value fn gradient | Classifier-free guidance |
| Learns dynamics | No | No | Yes (implicit) | No |
| Parameters | ~11M | ~8.6M | ~11M + ~411K | ~11M + ~0.5M |
| Eval modes | Policy, Planner | Policy, Planner | Policy, Planner | Policy, Planner |

---

## Evaluation Modes

All four architectures produce an action sequence of shape `(B, H, act_dim)` and plug into the same two evaluation modes. The model does not know or care which mode is used - it is the same trained weights either way. The policy/planner distinction is purely an inference-time choice.

**PolicyMode** (`src/diffplan/algorithms/policy_mode.py`): Sample 1 action sequence → execute first action → replan next step (receding horizon control). Fast.

**PlannerMode** (`src/diffplan/algorithms/planner_mode.py`): Sample K action sequences → simulate each forward in the real environment (save/restore state) → select the sequence with highest discounted return → execute first action. Slower but higher quality.

**Sampling:** DDPM training with 100 diffusion steps. DDIM inference with 20 steps (configurable).

**Normalization:** Observations normalized to zero-mean unit-variance. Actions normalized to [-1, 1].
