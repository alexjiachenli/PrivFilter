"""Patch-poisoning helpers for client training and evaluation.

The shared-patch attack uses pattern 0. DBA assigns one local shard to each
malicious client and evaluates the combined global trigger.
"""

import copy
import math


DBA_ATTACK_MODES = {"dba", "distributed_backdoor_attack"}


def canonical_attack_mode(params):
    return str(params.get("attack_mode", params.get("attack.mode", ""))).strip().lower()


def is_dba_attack(params):
    return canonical_attack_mode(params) in DBA_ATTACK_MODES


def pattern_indices(params, pattern_index=None):
    """Return the configured pattern indices for one local or global trigger."""
    trigger_num = int(params.get("trigger_num", 1))
    if trigger_num < 1:
        raise ValueError(f"trigger_num must be positive, got {trigger_num}")
    if pattern_index is None:
        return list(range(trigger_num))
    index = int(pattern_index)
    if index < 0 or index >= trigger_num:
        raise ValueError(f"pattern index {index} outside [0, {trigger_num})")
    return [index]


def pattern_coordinates(params, indices):
    coordinates = []
    for index in indices:
        key = f"{int(index)}_poison_pattern"
        if key not in params:
            raise KeyError(f"missing configured poison pattern: {key}")
        pattern = params[key]
        if not isinstance(pattern, list) or not pattern:
            raise ValueError(f"poison pattern {key} must be a non-empty list")
        coordinates.extend(pattern)
    return coordinates


def validate_dba_configuration(params):
    """Fail early on malformed DBA shards without consuming any RNG state."""
    if not is_dba_attack(params):
        return
    trigger_num = int(params.get("trigger_num", 1))
    if trigger_num < 2:
        raise ValueError("DBA requires trigger_num >= 2")
    poison_client_num = int(params.get("poison_client_num", trigger_num))
    if poison_client_num < trigger_num:
        raise ValueError(
            "DBA requires at least one malicious client per trigger shard: "
            f"poison_client_num={poison_client_num}, trigger_num={trigger_num}"
        )
    shards = []
    for index in range(trigger_num):
        coords = pattern_coordinates(params, [index])
        normalized = [tuple(int(v) for v in pos) for pos in coords]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"DBA shard {index} contains duplicate coordinates")
        shards.append(set(normalized))
    if bool(params.get("dba_require_disjoint", True)):
        seen = set()
        for index, shard in enumerate(shards):
            overlap = seen.intersection(shard)
            if overlap:
                raise ValueError(f"DBA shard {index} overlaps earlier shards at {sorted(overlap)}")
            seen.update(shard)


def build_client_pattern_map(adversarial_ids, trigger_num):
    """Assign sorted malicious clients round-robin without touching global RNG."""
    trigger_num = int(trigger_num)
    if trigger_num < 1:
        raise ValueError(f"trigger_num must be positive, got {trigger_num}")
    return {
        int(client_id): rank % trigger_num
        for rank, client_id in enumerate(sorted(int(cid) for cid in adversarial_ids))
    }


def resolve_client_pattern_index(params, client_id, handle=None):
    if not is_dba_attack(params):
        return 0
    mapping = getattr(handle, "adversarial_pattern_map", None) if handle is not None else None
    if mapping is None:
        mapping = params.get("adversarial_pattern_map")
    if not mapping:
        raise RuntimeError("DBA client-pattern map is not initialized")
    client_id = int(client_id)
    if client_id in mapping:
        return int(mapping[client_id])
    if str(client_id) in mapping:
        return int(mapping[str(client_id)])
    raise KeyError(f"malicious client {client_id} has no DBA pattern assignment")


def apply_pixel_patterns(ori_image, params, indices, target):
    image = copy.deepcopy(ori_image)
    coordinates = pattern_coordinates(params, indices)
    channels = 1 if str(params.get("type", "")).upper() == "MNIST" else 3
    for row, column in coordinates:
        for channel in range(channels):
            image[channel][int(row)][int(column)] = 1
    return image, target


def poison_batch(images, labels, params, client_id, handle, poison_rate=None):
    """Poison the leading batch fraction with the client's configured trigger."""
    rate = float(params.get("poison_rate", 0.0) if poison_rate is None else poison_rate)
    rate = min(1.0, max(0.0, rate))
    count = min(len(images), max(0, int(math.ceil(len(images) * rate))))
    poisoned_images = images.clone()
    poisoned_labels = labels.clone()
    pattern_index = resolve_client_pattern_index(params, client_id, handle)
    for index in range(count):
        poisoned_images[index], poisoned_labels[index] = apply_pixel_patterns(
            poisoned_images[index], params, [pattern_index], handle.target
        )
    return poisoned_images, poisoned_labels
