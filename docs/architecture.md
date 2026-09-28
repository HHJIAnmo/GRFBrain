# Architecture and parameter contract

## 1. Conditional source

For the mean history graph, construct the symmetric normalized Laplacian

`L_C = I - D^{-1/2} A_sym D^{-1/2}`

and trace-normalized covariance

`K_C = TrNorm((I + alpha L_C)^(-nu))`.

Released TUSZ checkpoint values are `alpha=1`, `nu=1`,
`sigma_X=0.4956156681505653`, and `sigma_A=0.25`. `sigma_X` is a train-derived
scale rather than a universal EEG constant and must be recalibrated for a new
preprocessing contract or dataset.

Node source residuals are `R0_X = sigma_X K_C^(1/2) epsilon_X`. Edge source
residuals apply `K_C^(1/2)` on both sides of symmetric Gaussian edge noise and
use an additional exact edge-space trace normalization. The source is Gaussian,
history-conditioned and graph-spectral; it is not asserted to lie on the true
brain manifold.

## 2. Conditions and states

| Quantity | Shape | Construction |
|---|---:|---|
| `X_s` | `B×11×19×100` | past log-FFT sequence |
| `A_s` | `B×11×19×19` | past directed graphs |
| `M_X` | `B×19×100` | arithmetic history mean |
| `M_A` | `B×171` | frozen/learned graph forecaster output |
| node context | `B×19×64` | pooled `M_X` + zero-init projected per-channel GRU |
| edge global | `B×64` | graph forecaster context |
| node residual | `B×19×100` | Flow state relative to `M_X` |
| edge residual | `B×171` | Flow state relative to `M_A` |

The temporal encoder is one shared unidirectional `GRU(100,64)` applied to each
of the 19 channels over 11 physical windows. Its output passes through a
zero-initialized `Linear(64,64)` residual adapter.

## 3. Joint Node–Edge Transformer

Frozen defaults:

| Hyperparameter | Value |
|---|---:|
| nodes / node features | `19 / 100` |
| undirected edge tokens | `171` |
| hidden width | `64` |
| attention heads | `4` |
| head width | `16` |
| Joint blocks | `2` |
| tau embedding | `32` |
| node/edge context | `64 / 64` |
| dropout | `0.0` |
| interaction | `bidirectional` |

Exact parameter counts in the released default implementation are: graph
forecaster `125,803`, temporal condition encoder `36,032`, Joint velocity field
`393,581`, and node-hidden classifier `65` (`555,481` total). The Joint field
and graph-forecaster state-dictionary names are compatible with the original
EvoBrain modules; the supplied import command assembles the split checkpoints.

Input projections:

- Node token: `Linear([node residual, M_X], 200 -> 64)` plus node context.
- Edge token: `Linear([edge residual, M_A], 2 -> 64)` plus a symmetric pair
  encoding of endpoint node contexts and `Linear([L_C(i,j),K_C(i,j)],2->64)`.
- Global FiLM condition: `[node global(64), edge global(64), tau(32)]`, total 160.

Each block performs:

1. LayerNorm + FiLM scale/shift for node and edge streams.
2. Four-head node self-attention; edge tokens add a per-head attention bias.
3. Edge-to-node incident messages using `Linear(64,64)` and degree averaging.
4. Node FFN `64 -> 256 -> 64`, GELU.
5. Independent edge MLP `64 -> 128 -> 64`, GELU.
6. Node-to-edge symmetric message from `[h_i+h_j, |h_i-h_j|, h_i*h_j]`,
   `192 -> 128 -> 64`.
7. Edge FFN `64 -> 256 -> 64`, GELU.

Velocity heads are `LayerNorm(64) -> Linear(64,100)` for nodes and
`LayerNorm(64) -> Linear(64,1)` for each edge token. The latter reconstructs a
symmetric zero-diagonal matrix.

## 4. Flow Matching training

With one shared `tau ~ Uniform(0,1)`:

`R_tau = (1-tau) R0 + tau R1`, `u = R1-R0`, where
`R1_X=X_target-M_X` and `R1_A=A_target-M_A`.

The normalized loss is

`L_FM = MSE(v_X,u_X)/q_X + MSE(v_A,u_A)/q_A`.

The released TUSZ values are `q_X=0.491373396248178` and
`q_A=0.12629429877215387`. They must be estimated on train only for a new data
contract. Consistency is diagnostic by default (`lambda_consistency=0`). A
linear conditional interpolation is not called optimal transport unless an OT
coupling is separately implemented.

## 5. Node hidden no transport

At downstream inference, `R_X=0`, `R_A=0`, `tau=1`, and the Joint network is
called exactly once (`NFE=0`, no ODE solver). Classification reads the node
tokens entering the final node velocity linear layer:

`ReLU(H_X) -> shared Linear(64,1) -> max over 19 nodes`.

Edge hidden tokens are not concatenated into the classifier, but edges still
affect node tokens through edge attention bias, edge-to-node messages, FiLM and
history operators. During downstream training, the objective is BCE plus
`0.1 * L_FM` after one frozen-field warmup epoch.
