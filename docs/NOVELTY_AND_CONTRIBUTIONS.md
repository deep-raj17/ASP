# Novelty and Contributions Assessment

## A. Proposed research problem

We propose to improve industrial acoustic anomaly detection by learning when each anomaly signal should be trusted for a given input, rather than applying a single fixed fusion rule across all samples. The research problem is not simply to build a stronger detector, but to make the anomaly score robust to varying machine type, noise condition, and sample difficulty in a way that is trainable from validation data only.

The motivating setting is the existing CHAAD-style pipeline in this repository: multiple complementary anomaly signals are computed from the same acoustic input, but the final decision is produced by a fixed weighted combination of those signals. This design is practical, but it does not model the fact that different signals may be more reliable under different conditions.

## B. Research gap

The current repository already contains several strong building blocks:

- a hybrid encoder–decoder architecture,
- multiple anomaly signals (reconstruction, embedding distance, Mahalanobis distance, contrastive distance),
- z-score-based calibration from normal-only training data,
- and a final threshold chosen from validation data.

However, the pipeline does not yet contain a mechanism for conditional reliability estimation. In other words, the system treats every score as if it contributes with the same credibility for every sample. That is a meaningful limitation because the reliability of reconstruction error, embedding distance, geometric distance, and contrastive distance can change substantially across machine type, noise level, and sample complexity.

The research gap is therefore:

- existing systems use fixed or globally tuned fusion weights,
- existing calibration is global and condition-agnostic,
- and the current implementation does not learn which source of evidence should dominate for a given input.

This gap is not solved by merely combining several standard methods into one system. The novelty claim must be about the specific mechanism that learns reliability-aware fusion, not about the mere existence of a multi-branch architecture.

## C. Existing approaches and their limitations

### 1. Standard methods reused from prior literature

These components are standard and should not be treated as novel by themselves:

- EfficientNet or ResNet-style backbone feature extraction.
- Transformer or BiLSTM temporal modeling.
- Autoencoder reconstruction for anomaly scoring.
- Supervised contrastive representation learning.
- Mahalanobis distance using a covariance estimate.
- z-score normalization for score calibration.
- Threshold selection via ROC-based criteria such as Youden’s J.

These methods are well established in anomaly detection and representation learning. They are valuable engineering components, but they are not the claimed novelty of this work.

### 2. Engineering integrations in the current repository

The current repository is best viewed as an engineering integration of known methods:

- a hybrid model that combines a CNN backbone, temporal module, attention pooling, classifier head, and autoencoder branch;
- a multi-score anomaly pipeline that computes reconstruction, embedding, Mahalanobis, and contrastive signals;
- a calibration stage that estimates normal-data statistics for each signal;
- a production-oriented inference wrapper and export pipeline.

These are important implementation choices, but they are not, by themselves, a novel algorithmic contribution.

### 3. Candidate novel algorithmic contribution

We propose a condition-aware reliability-weighted anomaly-score fusion mechanism.

This is the primary contribution because it is not simply “combining multiple scores,” but rather learning a sample-dependent gating function that estimates how reliable each anomaly score is under the current operating condition.

## D. Exact proposed contribution

We propose a reliability-aware fusion module for industrial acoustic anomaly detection.

For each sample $x$, we compute a calibrated anomaly score $z_k(x)$ for each source $k$, and we estimate a reliability score $r_k(x)$ that reflects how trustworthy that source is for the current sample. The final anomaly score is formed by a softmax-weighted fusion:

$$
S(x) = \sum_{k=1}^{K} w_k(x)\, z_k(x), \quad w_k(x)=\frac{\exp(r_k(x)/\tau)}{\sum_j \exp(r_j(x)/\tau)}.
$$

The novelty is not in the existence of multiple anomaly scores. The novelty is that the fusion weights are learned as a function of the sample and its operating condition, instead of being fixed by hand or tuned once globally.

### Why this is different from standard methods

- It differs from standard weighted score fusion because the weights are sample-dependent or condition-dependent, rather than fixed.
- It differs from ordinary z-score calibration because the calibration is only a base score transform; reliability estimation is an additional learned layer that modulates the contribution of each score.
- It differs from ordinary Mahalanobis anomaly detection because it is not a single geometric distance detector; it is a learned fusion of multiple complementary score families.
- It differs from existing reconstruction-plus-classification systems because the proposed module explicitly models score reliability rather than relying on a fixed classifier or decoder output alone.

## E. Mathematical formulation

Let the base anomaly scores be:

- reconstruction score $s_{rec}(x)$,
- embedding-distance score $s_{emb}(x)$,
- Mahalanobis score $s_{mahal}(x)$,
- contrastive-distance score $s_{contra}(x)$.

Each score is first calibrated using normal-data statistics:

$$
z_k(x)=\frac{s_k(x)-\mu_k}{\sigma_k}, \quad k\in\{rec,emb,mahal,contra\}.
$$

The reliability estimator is a small function $g_\phi$ that maps the current sample representation and available metadata to reliability values:

$$
r_k(x)=g_{\phi,k}(\psi(x), m(x), c(x)),
$$

where:

- $\psi(x)$ is a feature representation such as the embedding or a summary of the score vector,
- $m(x)$ is machine type or machine ID,
- $c(x)$ is the noise condition or operating condition,
- $\phi$ are learnable parameters.

The final fused score is:

$$
S(x)=\sum_k w_k(x)\, \sigma(z_k(x)), \quad
w_k(x)=\text{softmax}(r_k(x)/\tau).
$$

The training objective can be written as a ranking loss over validation data:

$$
\mathcal{L}_{rel} = \sum_{(i,j)\in\mathcal{P}} \max\left(0,\, 1 - (S(x_i)-S(x_j))(y_i-y_j)\right),
$$

where $y_i\in\{0,1\}$ is the label and $\mathcal{P}$ is a set of validation pairs.

## F. Algorithm or pseudocode

```text
Input: training normals, validation set with labels, test set

1. Fit per-score calibration statistics on training normals:
   - estimate mu_k and sigma_k for each score k

2. Compute calibrated scores on validation data:
   - z_k(x) = (s_k(x) - mu_k) / sigma_k

3. Train the reliability estimator g_phi on validation data:
   - input: (feature embedding, machine metadata, noise condition, calibrated score vector)
   - target: ranking signal that encourages higher fused scores for anomalies
   - optimize the pairwise ranking loss

4. Freeze g_phi and the calibration statistics

5. At inference time:
   - compute s_k(x) for each score family
   - compute z_k(x)
   - compute reliability weights w_k(x)
   - fuse to obtain S(x)
   - apply a frozen threshold selected on validation data
```

## G. Complexity analysis

Let $K$ be the number of base scores, $d$ the embedding dimension, and $H$ the hidden width of the reliability estimator.

- Base score computation cost: dominated by the backbone and reconstruction branch, as in the current implementation.
- Reliability estimation cost: approximately $O(KH + Hd)$ for a small MLP, which is negligible relative to the backbone forward pass.
- Memory overhead: one additional small module and a few scalar statistics.
- Training overhead: one extra optimization stage on validation data, but no additional inference-time cost beyond the lightweight gating network.

The main added cost is therefore a small additional inference head and a validation-stage training step, not a major architectural burden.

## H. Required ablation experiments

To prove the contribution, the following ablations are necessary.

1. Fixed-weight fusion baseline
   - Replace the learned gate with the current hand-selected weights.
   - This directly tests whether the learned gating module adds value.

2. Equal-weight fusion baseline
   - Compare against uniform averaging of calibrated scores.

3. Global learned weights baseline
   - Learn one global weight vector on validation data and apply it to all samples.
   - This isolates the benefit of sample-dependent weighting.

4. Condition-agnostic reliability estimator
   - Train the same gate without machine/noise metadata.
   - This tests whether the condition signal is necessary.

5. Full proposed method
   - Use the condition-aware reliability-weighted fusion with a frozen validation threshold.

The most direct ablation is:

- keep the model, calibration, and threshold selection unchanged,
- replace the learned reliability gate with the existing fixed weights,
- and compare performance on the same held-out test set.

If the proposed method improves ROC-AUC, PR-AUC, or calibration under the same backbone and score set, the value of the contribution is demonstrated.

## I. Claims that are supported by evidence

The following claims are supported by the current repository:

- The system uses multiple anomaly sources: reconstruction, embedding distance, Mahalanobis distance, and contrastive distance.
- The current scoring pipeline calibrates each signal with a simple z-score transform using normal-data statistics.
- The current fusion rule uses fixed weights rather than a learned reliability gate.
- The threshold is selected from validation data and is not yet explicitly condition-aware.
- The repository currently lacks a dedicated reliability-estimation module and lacks an ablation study that isolates the effect of fusion design.

Repository evidence:

- The fusion rule in [inference/detector.py](../inference/detector.py) combines four calibrated signals with fixed weights taken from [config.py](../config.py).
- The training objective in [training/loss.py](../training/loss.py) is a standard multi-objective combination of BCE, contrastive, and reconstruction losses.
- The model architecture in [models/hybrid_model.py](../models/hybrid_model.py) is a hybrid reconstruction-plus-classification network, but it does not include a reliability-weighting module.
- No implementation of a learned reliability-aware fusion gate or condition-aware weighting mechanism was found in the repository.

## J. Claims that are not yet supported

The following claims are not yet supported by the repository as it currently exists:

- That the proposed reliability-aware fusion improves anomaly detection over fixed-weight fusion.
- That the proposed method is more robust to unseen machine IDs or unseen noise conditions.
- That the method yields statistically significant gains over ordinary z-score calibration and ordinary weighted fusion.
- That the contribution is practically superior in a publication-ready setting without controlled ablation and test-set evaluation.

These claims should be treated as hypotheses to be tested, not as established results.

## Current status

3. A candidate contribution is defined but not experimentally validated.
