Optimizing tool selection...I'll create a short TODO plan for drafting the Methods section, then draft the full Methods text.

Created 5 todos

Progress: plan created and sections drafted — next I present the complete Methods text.

**Methods**

Overall, Compressed-Context Algorithm Distillation (CAD) is a two‑stage architecture that separates (1) a compression transformer which encodes high‑dimensional task context into a compact latent sequence, and (2) an Algorithm Distillation (AD) Transformer that consumes the compressed context to produce policy outputs or predictions. CAD is designed to preserve the information necessary for downstream decision-making while drastically reducing context length and compute for in‑context inference.

**1. Overall structure of CAD**

- **High-level architecture.** CAD consists of two neural modules trained to cooperate: (i) a compression transformer Cθ that maps a long context X (e.g., trajectories, demonstrations, or multimodal observations) to a compact latent sequence Z = Cθ(X) of fixed, small length m, and (ii) an AD Transformer Aφ that conditions on Z (and optionally short query tokens) to produce target outputs Y (actions, predictions, or next-step tokens). During evaluation, only Z and the query are provided to Aφ, enabling in‑context generalization with reduced memory and latency.

- **Input representation and preprocessing.** The raw context X is first tokenized by modality-specific encoders into a sequence of tokens X = [x1, …, xN]. For trajectory data this typically interleaves state, action, and reward tokens. Each token xi is embedded: e(xi) ∈ R^d, and positional (or relative) embeddings p(i) are added:
  $$
  \tilde{x}_i = e(x_i) + p(i).
  $$
  The compression transformer consumes [\tilde{x}_1, …, \tilde{x}_N] and outputs m compressed tokens Z = [z_1, …, z_m], where m ≪ N.

- **Compression transformer (Cθ).** Cθ is instantiated as a transformer encoder (stack of pre‑norm residual transformer blocks) whose output is pooled or projected into the compressed sequence Z. Two practical variants are supported:
  - Bottleneck projection: the final hidden states are passed through an attention‑based pooling module that attends from m learned latent queries Q = {q_j} to the sequence of hidden states H, producing Z_j = Attention(q_j, H, H).
  - Sequence downsampling: strided attention or learned downsampling maps N → m by learned linear projections followed by transformer layers.
  The compression transformer can be trained to reconstruct the original token sequence (see pretraining below), to minimize reconstruction error while encouraging compactness.

- **AD Transformer (Aφ).** Aφ is an autoregressive decoder or encoder‑decoder transformer that accepts Z as context via cross‑attention. Given a query token sequence Q (e.g., current state, few‑shot prompt), Aφ applies self‑attention over query tokens and cross‑attention with keys/values derived from Z. The cross‑attention uses the standard scaled dot‑product attention:
  $$
  \text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,
  $$
  where for cross‑attention $K,V$ are linear projections of Z. The AD Transformer thus implements a learned algorithm that maps compressed context + query → outputs.

- **Processing pipeline (inference).** At inference time:
  1. Raw context X is encoded and passed through Cθ to produce Z.
  2. Query tokens (current state or prompt) are embedded and fed to Aφ alongside Z through cross‑attention.
  3. Aφ autoregressively decodes the target tokens Y (e.g., action probabilities or next tokens).

- **Design considerations.** Keys on design include: (i) choosing m to trade off compression and task performance; (ii) selecting reconstruction objective and regularization for Cθ to preserve task‑relevant information; and (iii) using KD (knowledge distillation) to align Aφ outputs with a stronger teacher that sees the full context.

**2. Practical implementation**

This section details dataset generation, pretraining of the compression transformer, training the AD Transformer and fine‑tuning Cθ, and the in‑context evaluation protocol.

- **Dataset generation.** We generate a dataset D = {(X_i, Q_i, Y_i)} of context-query-target triples using environment rollouts or offline datasets. For reinforcement tasks, each X_i is a trajectory segment (states, actions, rewards) sampled from a policy or replay buffer; Q_i is a query state (or few‑shot prompt), and Y_i is the next action or sequence of future returns. To build robust compression, contexts are sampled from diverse seeds, task variations, and demonstration styles. For supervised domains, X_i are example input-output pairs and Q_i the test input.

- **Pretraining the compression transformer.** Before coupling with the AD Transformer, Cθ is pretrained to produce compressed latents Z that permit faithful reconstruction of the original token sequence. Let Rψ be a lightweight reconstruction head (decoder) that maps Z back to token logits (or decoded tokens). The pretraining objective minimizes a reconstruction loss L_rec:
  $$
  L_{\text{rec}}(\theta,\psi) = \mathbb{E}_{X\sim D}\left[ \ell_{\text{rec}}\big( R_\psi(C_\theta(X)), X\big)\right],
  $$
  where $\ell_{\text{rec}}$ is typically cross‑entropy for discrete token sequences or mean squared error for continuous features. Optionally, a capacity regularizer R(Z) (e.g., L2 penalty or information bottleneck term) is added:
  $$
  L_{\text{pre}}(\theta,\psi)=L_{\text{rec}}(\theta,\psi) + \lambda_{\text{cap}} \mathbb{E}_X[R(C_\theta(X))].
  $$
  This pretraining helps Cθ learn to compress general statistical structure of contexts and speeds downstream AD training.

- **Training the AD Transformer (Aφ).** The AD Transformer is trained to produce targets Y given compressed contexts Z. Two training paradigms are used:
  - Direct supervised learning: train Aφ on (Z, Q, Y) pairs where Z = Cθ(X) from the pretrained compression model (frozen or trainable).
  - Distillation from a full‑context teacher T: we train Aφ to match the teacher’s output distribution p_T(y|X,Q) while seeing only Z and Q. The distillation loss uses KL divergence between teacher logits and student logits, combined with task loss:
    $$
    L_{\text{AD}}(\phi) = \mathbb{E}_{(X,Q,Y)}\big[\alpha\,\ell_{\text{task}}(A_\phi(Z,Q),Y) + \beta\,\text{KL}\big(p_T(\cdot|X,Q)\,\|\,p_{A_\phi}(\cdot|Z,Q)\big)\big],
    $$
    where $Z=C_\theta(X)$ and $\ell_{\text{task}}$ is cross‑entropy (supervised) or negative expected return (RL).
  In practice we set $\alpha,\beta\ge 0$ to balance supervised signal and teacher guidance. The teacher T can be a model that has full access to X or a Monte‑Carlo estimate of optimal actions.

- **Fine‑tuning the compression transformer.** After initial AD training with a frozen Cθ, we optionally fine‑tune Cθ jointly with Aφ to better preserve task‑relevant information (rather than full reconstruction). The joint objective mixes the AD loss and an auxiliary reconstruction regularizer:
  $$
  L_{\text{joint}}(\theta,\phi,\psi) = L_{\text{AD}}(\phi;\theta) + \gamma\,L_{\text{rec}}(\theta,\psi),
  $$
  where $L_{\text{AD}}(\phi;\theta)$ emphasizes task performance given $Z=C_\theta(X)$ and $\gamma$ controls how strongly reconstruction is enforced. Fine‑tuning typically uses smaller learning rates for Cθ and weight decay to prevent catastrophic collapse of compression capacity.

- **Losses and optimization details.** Summarizing the losses used:
  - Compression pretraining: $L_{\text{pre}} = L_{\text{rec}} + \lambda_{\text{cap}} R(Z)$.
  - AD training (distillation): $L_{\text{AD}} = \alpha \ell_{\text{task}} + \beta \text{KL}(p_T\|p_S)$.
  - Joint fine‑tuning: $L_{\text{joint}} = L_{\text{AD}} + \gamma L_{\text{rec}}$.
  Optimization uses AdamW with learning‑rate schedules and gradient clipping; typical hyperparameters (batch size, learning rate, weight decay) are chosen via validation.

- **In‑context evaluation.** The evaluation protocol measures Aφ’s ability to generalize from compressed context tokens at inference time:
  1. For each held‑out episode, compress the full context X_test with Cθ to produce Z_test.
  2. Provide Z_test and query Q_test to Aφ and generate predicted outputs Ŷ.
  3. Evaluate task metrics (accuracy, cumulative reward, return prediction error) and compare to baselines: (a) teacher T using full X_test; (b) AD with raw full context truncated; (c) other compression baselines (e.g., PCA, simple pooling).
  Ablations vary m, reconstruction regularization, and whether Cθ is fine‑tuned jointly.

- **Practical choices and extensions.** In practice:
  - Z may be continuous vectors or discrete latent tokens; discrete tokens allow cheaper storage/lookup and hybrid retrieval-augmented methods.
  - Compression can be made causal (preserving temporal order) or noncausal, depending on whether online compression is required.
  - The AD Transformer can be adapted to compute cross‑attention using multiple heads and multiple layers; positional encodings for compressed tokens help preserve order information.

**Summary of theoretical intuition.** CAD separates representation compression from algorithmic mapping: Cθ focuses on compactly encoding context, while Aφ learns the algorithmic reasoning required for downstream tasks from compressed signals. Distillation from a full‑context teacher ensures that learning signals carry rich algorithmic structure that Aφ can emulate despite limited context capacity. Joint fine‑tuning balances compression fidelity and task relevance.

If you’d like, I can (a) convert this into LaTeX-ready paragraphs for direct insertion into your manuscript, (b) add a figure caption describing the CAD pipeline, or (c) tailor the description to match exact hyperparameters and architecture choices in your code (for example, by inspecting compression.py and ad.py). Which would you prefer next?