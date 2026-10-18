"""CrowdGuard validation and vote fusion.

Reference: P. Rieger et al., "CrowdGuard: Federated Backdoor Detection in
Federated Learning," NDSS 2024. Adapted from the authors' OpenFL tutorial:
openfl-tutorials/experimental/workflow/CrowdGuard/.

This implementation evaluates hidden-layer behavior and combines client votes
in a local process. The original protocol places these operations in TEEs.
Convolutional features are spatially averaged by default and validation uses
a configurable sample cap, applied uniformly across clients.
"""

from __future__ import annotations

import math
from copy import deepcopy
from enum import Enum
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import cosine_similarity


VOTE_FOR_BENIGN = 1
VOTE_FOR_POISONED = 0

DEFAULT_MAX_VALIDATION_SAMPLES = 64
_SIGNIFICANCE_P = 0.01
_THREE_SIGMA = 3
_BOXPLOT_WHIS = 5.5


class DistanceMetric(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclid"


def _euclid(t1: torch.Tensor, t2: torch.Tensor) -> float:
    return float(torch.norm(t1.view(-1) - t2.view(-1), 2).cpu().item())


def _cosine(t1: torch.Tensor, t2: torch.Tensor) -> float:
    a = t1.view(-1).reshape(1, -1)
    b = t2.view(-1).reshape(1, -1)
    return float(1 - cosine_similarity(a, b).cpu().item())


def _distance(metric: DistanceMetric, t1: torch.Tensor, t2: torch.Tensor) -> float:
    if metric == DistanceMetric.COSINE:
        return _cosine(t1, t2)
    if metric == DistanceMetric.EUCLIDEAN:
        return _euclid(t1, t2)
    raise ValueError(f"Unsupported distance metric: {metric}")


def _reduce_layer_output(value: torch.Tensor, spatial_average: bool) -> torch.Tensor:
    """Flatten a sample's layer output, optionally averaging spatial dimensions."""
    v = value.detach().to("cpu", dtype=torch.float32)
    if spatial_average and v.dim() >= 3:
        v = v.mean(dim=tuple(range(1, v.dim())))
    return v.reshape(-1)


def _predict_internal_states(model, data_batches, device, spatial_average):
    """Deep-layer outputs per sample, plus the label of each sample."""
    predictions: List[List[torch.Tensor]] = []
    labels: List[int] = []
    num_layers: Optional[int] = None
    was_training = model.training
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for data, target in data_batches:
            data = data.to(device)
            outputs = model.predict_internal_states(data)
            if num_layers is None:
                num_layers = len(outputs)
            elif num_layers != len(outputs):
                raise RuntimeError("model returned inconsistent layer counts")
            batch_n = int(target.shape[0])
            base = len(predictions)
            for _ in range(batch_n):
                predictions.append([])
            for layer_values in outputs:
                for idx in range(batch_n):
                    predictions[base + idx].append(
                        _reduce_layer_output(layer_values[idx], spatial_average)
                    )
            for t in target:
                labels.append(int(t.detach().cpu().item()))
    model.to("cpu")
    if was_training:
        model.train()
    return predictions, labels, (num_layers or 0)


def _batched_distances(metric, model_stack, global_stack):
    """Per-sample distance between a model's layer outputs and the global's.

    ``model_stack`` and ``global_stack`` are ``(n_samples, dim)``.  Returns a
    ``(n_samples,)`` tensor.  Mathematically identical to calling ``_distance``
    once per sample, but as one tensor op instead of ``n_samples`` of them --
    the per-sample Python loop is what forced the validation-sample cap.
    """
    if metric == DistanceMetric.COSINE:
        return 1.0 - torch.nn.functional.cosine_similarity(model_stack, global_stack, dim=1)
    if metric == DistanceMetric.EUCLIDEAN:
        return torch.norm(model_stack - global_stack, p=2, dim=1)
    raise ValueError(f"Unsupported distance metric: {metric}")


def _hlbim(
    metric, prediction_matrix, global_predictions, sample_indices_by_label, own_index, stacked=None
):
    """Hidden Layer Backdoor Inspection Metric: label x model x layer."""
    sample_count = len(prediction_matrix)
    model_count = len(prediction_matrix[0])
    layer_count = len(prediction_matrix[0][0])

    global_distance = [
        [[0.0] * layer_count for _ in range(model_count)] for _ in range(sample_count)
    ]
    if stacked is not None:
        model_stacks, global_stacks = stacked
        for m_i in range(model_count):
            for l_i in range(layer_count):
                d = _batched_distances(metric, model_stacks[m_i][l_i], global_stacks[l_i])
                for s_i, value in enumerate(d.tolist()):
                    global_distance[s_i][m_i][l_i] = value
    else:
        for s_i, per_model in enumerate(prediction_matrix):
            g = global_predictions[s_i]
            for m_i, per_layer in enumerate(per_model):
                for l_i, value in enumerate(per_layer):
                    global_distance[s_i][m_i][l_i] = _distance(metric, value, g[l_i])

    # Divide by this client's own distances, so each client reads the cohort
    # relative to itself.
    relation = [[[0.0] * layer_count for _ in range(model_count)] for _ in range(sample_count)]
    for s_i, per_model in enumerate(global_distance):
        own = per_model[own_index]
        for m_j, per_layer in enumerate(per_model):
            for l_i, value in enumerate(per_layer):
                relation[s_i][m_j][l_i] = (value / own[l_i]) if own[l_i] != 0 else 0.0

    # Average per label, then subtract 1 and square while keeping the sign.
    out: Dict[int, List[List[float]]] = {}
    for label, sample_list in sample_indices_by_label.items():
        acc = [[[] for _ in range(layer_count)] for _ in range(model_count)]
        for s_i in sample_list:
            for m_j in range(model_count):
                for l_i in range(layer_count):
                    acc[m_j][l_i].append(relation[s_i][m_j][l_i])
        out[label] = [[0.0] * layer_count for _ in range(model_count)]
        for m_j in range(model_count):
            for l_i in range(layer_count):
                mean = float(np.mean(acc[m_j][l_i]).item()) if acc[m_j][l_i] else 0.0
                x = mean - 1.0
                sq = x * x
                out[label][m_j][l_i] = sq if x >= 0 else -sq
    return out


def _prune_poisoned_models(num_layers, total_models, own_client_index, distances_by_metric):
    """Iterative significance-test-gated pruning (reference Alg. 2)."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from scipy.stats import kstest, levene, ttest_ind
    from sklearn import preprocessing
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import PCA

    detected: List[int] = []
    for _metric, per_label in distances_by_metric.items():
        layer_length = num_layers * len(per_label)
        flat = [[0.0] * layer_length for _ in range(total_models)]
        for label_count, (_, per_model) in enumerate(sorted(per_label.items())):
            for m_i, per_layer in enumerate(per_model):
                for l_i, value in enumerate(per_layer):
                    flat[m_i][l_i + label_count * num_layers] = value

        client_indices = [x for x in range(len(flat)) if x != own_client_index]
        pruned: List[int] = []
        has_malicious = True
        new_round_needed = True
        max_pruning_count = int(math.floor((len(flat) - 1) / 2))

        while has_malicious and new_round_needed:
            pruned_local = deepcopy(pruned)
            candidate_rows = [
                v for x, v in enumerate(flat) if x != own_client_index and x not in pruned
            ]
            candidate_idx = [
                x for x in range(len(flat)) if x != own_client_index and x not in pruned
            ]
            if len(candidate_rows) <= 1:
                break

            median_graph = [
                float(np.median([row[l] for row in candidate_rows]).item())
                for l in range(layer_length)
            ]
            pca_input = list(candidate_rows) + [median_graph]
            scaled = preprocessing.scale(pca_input)
            pca = PCA()
            pca.fit(scaled)
            pca_data = pca.transform(scaled)
            first = pca_data.T[0]
            cluster_input = [[float(v)] for v in first]
            plain = [float(v) for v in first]

            median_val = float(np.median(plain))
            x_values = [abs(v - median_val) for v in plain if v >= median_val]
            y_values = [abs(v - median_val) for v in plain if v < median_val]

            def _p(fn):
                try:
                    value = float(fn()[1])
                except Exception:
                    return 1.0
                return 1.0 if math.isnan(value) else value

            t_p = _p(lambda: ttest_ind(x_values, y_values))
            ks_p = _p(lambda: kstest(x_values, y_values))
            bartlett_p = _p(lambda: levene(x_values, y_values))

            bp = plt.boxplot(plain, whis=_BOXPLOT_WHIS)
            outlier_boxplot = len(bp["fliers"][0].get_ydata())
            plt.close()

            mean = float(np.mean(plain))
            std = abs(float(np.std(plain)))
            max_rule = 0.0
            if std > 0:
                for v in plain:
                    max_rule = max(max_rule, abs(v - mean) / std)

            has_malicious = (
                t_p < _SIGNIFICANCE_P
                or ks_p < _SIGNIFICANCE_P
                or bartlett_p < _SIGNIFICANCE_P
                or outlier_boxplot > 0
                or max_rule >= _THREE_SIGMA
            )

            ac = AgglomerativeClustering(
                n_clusters=2,
                distance_threshold=None,
                compute_full_tree=True,
                metric="euclidean",
                memory=None,
                connectivity=None,
                linkage="single",
                compute_distances=True,
            ).fit(cluster_input)
            labels = ac.labels_.tolist()
            median_cluster_label = labels[-1]
            malicious_slots = [i for i, v in enumerate(labels) if v != median_cluster_label]

            for slot, client in enumerate(candidate_idx):
                if slot in malicious_slots:
                    pruned_local.append(client)
            pruned_local = list(set(pruned_local))

            pruned_too_much = True
            if len(pruned_local) > max_pruning_count:
                dists = [abs(cluster_input[s][0] - cluster_input[-1][0]) for s in malicious_slots]
                order = sorted(range(len(dists)), key=lambda i: dists[i])
                sorted_clients = []
                for oi in order:
                    slot = malicious_slots[oi]
                    if slot < len(candidate_idx):
                        sorted_clients.append(candidate_idx[slot])
                overflow = len(pruned_local) - max_pruning_count
                for _ in range(overflow):
                    if not sorted_clients:
                        break
                    victim = sorted_clients.pop()
                    if victim in pruned_local:
                        pruned_local.remove(victim)
                pruned_too_much = False

            still_pruning = len(pruned) < len(pruned_local)
            new_round_needed = still_pruning and pruned_too_much
            if has_malicious and new_round_needed:
                pruned = pruned_local

        for client in client_indices:
            if client in pruned:
                detected.append(client)

    return sorted(set(detected))


def validate_models(
    global_model, models, own_client_index, local_data, device, *, spatial_average: bool = True
):
    """One client's verdict on every model in the cohort.

    Returns the indices (into ``models``) this client believes are poisoned.
    """
    if len(models) <= 1:
        return []

    per_model_predictions = []
    for model in models:
        preds, _, _ = _predict_internal_states(model, local_data, device, spatial_average)
        per_model_predictions.append(preds)

    global_predictions, labels, num_layers = _predict_internal_states(
        global_model, local_data, device, spatial_average
    )

    sample_count = len(global_predictions)
    prediction_matrix = [
        [per_model_predictions[m][s] for m in range(len(models))] for s in range(sample_count)
    ]

    sample_indices_by_label: Dict[int, List[int]] = {}
    for s_i, label in enumerate(labels):
        sample_indices_by_label.setdefault(label, []).append(s_i)

    # Stack once, reuse for both metrics: (model)(layer) -> (n_samples, dim).
    layer_count = len(global_predictions[0]) if global_predictions else 0
    global_stacks = [
        torch.stack([global_predictions[s][l] for s in range(sample_count)])
        for l in range(layer_count)
    ]
    model_stacks = [
        [
            torch.stack([per_model_predictions[m][s][l] for s in range(sample_count)])
            for l in range(layer_count)
        ]
        for m in range(len(models))
    ]

    distances_by_metric = {}
    for metric in (DistanceMetric.COSINE, DistanceMetric.EUCLIDEAN):
        distances_by_metric[metric] = _hlbim(
            metric,
            prediction_matrix,
            global_predictions,
            sample_indices_by_label,
            own_client_index,
            stacked=(model_stacks, global_stacks),
        )

    return _prune_poisoned_models(num_layers, len(models), own_client_index, distances_by_metric)


def _cluster_map(n, labels):
    out: Dict[int, List[int]] = {}
    for i in range(n):
        out.setdefault(labels[i], []).append(i)
    return out


def _biggest_cluster(cluster_map):
    return max(cluster_map, key=lambda k: len(cluster_map[k]))


def fuse_votes(binary_votes: Sequence[Sequence[int]]) -> List[int]:
    """Server-side stacked clustering over the client vote matrix.

    ``binary_votes[i][j]`` is client i's vote on model j, using
    ``VOTE_FOR_BENIGN`` / ``VOTE_FOR_POISONED``.  Returns the indices of the
    models accepted as benign.
    """
    from sklearn.cluster import DBSCAN, AgglomerativeClustering

    n = len(binary_votes)
    if n == 0:
        return []
    if n == 1:
        return [j for j, v in enumerate(binary_votes[0]) if v == VOTE_FOR_BENIGN]

    ac = AgglomerativeClustering(n_clusters=2, linkage="single", compute_distances=True).fit(
        binary_votes
    )
    agglomerative = _cluster_map(n, ac.labels_.tolist())
    negative_cluster = agglomerative[_biggest_cluster(agglomerative)]

    db_input = [list(binary_votes[i]) for i in negative_cluster]
    db = DBSCAN(eps=0.5, min_samples=1).fit(db_input)
    db_clusters = _cluster_map(len(negative_cluster), db.labels_.tolist())
    biggest = db_clusters[_biggest_cluster(db_clusters)]

    final_voting = db_input[biggest[0]]
    return [j for j, vote in enumerate(final_voting) if vote == VOTE_FOR_BENIGN]
