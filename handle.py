import copy
import os
import logging
import random
import math
import numpy as np
import torch
import torch.utils.data
from torchvision import datasets
from torchvision.transforms import transforms
from collections import defaultdict
from models.ResNet8 import ResNet8
from models.CNNMnist import CNNMnist
from models.ResNet18 import ResNet18
from models.ResNet50Cifar import ResNet50Cifar
from backdoor_attack import (
    build_client_pattern_map,
    is_dba_attack,
    poison_batch,
    validate_dba_configuration,
)
from device import device

logger = logging.getLogger("logger")


class Handle:
    def __init__(self, current_time, params, name):
        self.current_time = current_time
        self.model = None
        self.train_dataset = None
        self.test_dataset = None
        self.test_dataset_poisoned = None
        self.train_data = None
        self.clients_data_num = None
        self.test_data = None
        self.poisoned_test_data = None
        self.test_un_target_label_data = None
        self.test_target_label_data = None
        self.participants_list = None
        self.adversarial_namelist = None
        self.benign_namelist = None
        self.adversarial_pattern_map = {}
        self.participation_counts = None
        self._balanced_schedule = None
        self._frozen_schedule = None
        self.round_client_schedule = []
        self.schedule_type = None
        self.start_epoch = 1
        self.classes_dict = None
        self.params = params
        self.name = name
        self.target = 0
        output_root = self.params.get("output_root", "saved_models")
        self.folder_path = os.path.join(output_root, f"model_{self.name}_{current_time}")
        self.tinydata = []
        try:
            os.makedirs(self.folder_path, exist_ok=True)
        except Exception:
            logger.info("Folder already exists")
        logger.addHandler(logging.FileHandler(filename=f"{self.folder_path}/log.txt"))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)
        logger.info(f"current path: {self.folder_path}")
        if not self.params.get("environment_name", False):
            self.params["environment_name"] = self.name
        self.params["current_time"] = self.current_time
        self.params["folder_path"] = self.folder_path

    def _canonical_type(self, t: str) -> str:
        """Normalize dataset type aliases."""
        t_low = str(t).lower()
        if t_low in ["mnist"]:
            return "MNIST"
        if t_low in ["cifar10", "cifar-10"]:
            return "CIFAR10"
        if t_low in ["cifar100", "cifar-100"]:
            return "CIFAR100"
        if t_low in ["gtsrb"]:
            return "GTSRB"
        if t_low in ["tinyimagenet", "tiny-imagenet"]:
            return "TinyImagenet"
        return str(t)

    def load_data(self):
        logger.info("Loading data")
        dataPath = os.path.expanduser(str(self.params.get("data_root", "./data")))
        ds_type = self._canonical_type(self.params["type"])
        if ds_type == "CIFAR10":
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            self.train_dataset = datasets.CIFAR10(
                dataPath, train=True, download=True, transform=transform_train
            )
            self.test_dataset = datasets.CIFAR10(
                dataPath, train=False, download=True, transform=transform_test
            )
            self.test_dataset_poisoned = datasets.CIFAR10(
                dataPath, train=False, download=True, transform=transform_test
            )
        elif ds_type == "CIFAR100":
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
                ]
            )
            self.train_dataset = datasets.CIFAR100(
                dataPath, train=True, download=True, transform=transform_train
            )
            self.test_dataset = datasets.CIFAR100(
                dataPath, train=False, download=True, transform=transform_test
            )
            self.test_dataset_poisoned = datasets.CIFAR100(
                dataPath, train=False, download=True, transform=transform_test
            )
        elif ds_type == "GTSRB":
            transform_train = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629]
                    ),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629]
                    ),
                ]
            )
            try:
                from torchvision.datasets import GTSRB as TorchGTSRB

                self.train_dataset = TorchGTSRB(
                    root=dataPath, split="train", download=True, transform=transform_train
                )
                self.test_dataset = TorchGTSRB(
                    root=dataPath, split="test", download=True, transform=transform_test
                )
                self.test_dataset_poisoned = TorchGTSRB(
                    root=dataPath, split="test", download=True, transform=transform_test
                )
            except Exception:
                from datasets.GTSRB import GTSRB

                self.train_dataset = GTSRB(dataPath, train=True, transform=transform_train)
                self.test_dataset = GTSRB(dataPath, train=False, transform=transform_test)
                self.test_dataset_poisoned = GTSRB(dataPath, train=False, transform=transform_test)
        else:
            logger.info("no this type!!!!!")
        logger.info("reading data done")
        self.classes_dict = self.build_classes_dict()
        logger.info("build_classes_dict done")
        if self.params["sampling_dirichlet"]:
            indices_per_participant, dict_users = self.sample_dirichlet_train_data(
                self.params["number_of_total_participants"], alpha=self.params["dirichlet_alpha"]
            )
            train_loaders = [
                (pos, self.get_train(indices)) for pos, indices in indices_per_participant.items()
            ]
            self.clients_data_num = dict_users
        else:
            num_items = int(len(self.train_dataset) / self.params["number_of_total_participants"])
            dict_users, all_idxs = ({}, [i for i in range(len(self.train_dataset))])
            for i in range(self.params["number_of_total_participants"]):
                dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
                all_idxs = list(set(all_idxs) - dict_users[i])
            for i in range(self.params["number_of_total_participants"]):
                dict_users[i] = np.array(list(dict_users[i])).tolist()
            train_loaders = [(pos, self.get_train(indices)) for pos, indices in dict_users.items()]
            self.clients_data_num = [len(indices) for pos, indices in dict_users.items()]
        logger.info("train loaders done")
        self.train_data = train_loaders
        self.test_data = self.get_test()
        self.test_un_target_label_data, self.test_target_label_data = self.poison_test_dataset_ind()
        self.participants_list = list(range(self.params["number_of_total_participants"]))
        self.adversarial_namelist = random.sample(
            self.participants_list, self.params["poison_client_num"]
        )
        self.benign_namelist = list(set(self.participants_list) - set(self.adversarial_namelist))
        if is_dba_attack(self.params):
            validate_dba_configuration(self.params)
            self.adversarial_pattern_map = build_client_pattern_map(
                self.adversarial_namelist, self.params["trigger_num"]
            )
            self.params["adversarial_pattern_map"] = dict(self.adversarial_pattern_map)
            logger.info(
                f"[DBA] deterministic client-pattern assignments: {self.adversarial_pattern_map}"
            )
        if self.params.get("gpu_preload", False):
            if torch.cuda.is_available():
                logger.info("Preloading training data to GPU...")
                self.preload_train_data_to_gpu()
                logger.info("Preloading test data to GPU...")
                self.preload_test_data_to_gpu()
                logger.info("GPU preload completed")
            else:
                logger.warning("GPU not available, skip GPU preload to avoid CUDA errors")

    def build_classes_dict(self):
        classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if label in classes:
                classes[label].append(ind)
            else:
                classes[label] = [ind]
        return classes

    def sample_dirichlet_train_data(self, num_participants, alpha):
        """Partition class indices across clients using a Dirichlet distribution."""
        classes = self.classes_dict
        class_size = len(classes[0])
        per_participant_list = defaultdict(list)
        num_classes = len(classes.keys())
        image_nums = []
        for n in range(num_classes):
            image_num = []
            random.shuffle(classes[n])
            if self.params["type"] == "GTSRB" or self.params["type"] == "CIFAR-100":
                class_size = len(classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(num_participants * [alpha])
            )
            for user in range(num_participants):
                num_imgs = int(round(sampled_probabilities[user]))
                sampled_list = classes[n][: min(len(classes[n]), num_imgs)]
                image_num.append(len(sampled_list))
                per_participant_list[user].extend(sampled_list)
                classes[n] = classes[n][min(len(classes[n]), num_imgs) :]
            image_nums.append(image_num)
        dict_users = []
        for i in range(num_participants):
            dict_users.append(len(per_participant_list[i]))
        return (per_participant_list, dict_users)

    def get_train(self, indices):
        try:
            use_pinned = torch.cuda.is_available() and getattr(device, "type", "cpu") == "cuda"
        except Exception:
            use_pinned = False
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.params["batch_size"],
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
            pin_memory=use_pinned,
            num_workers=0,
        )
        return train_loader

    def get_test(self):
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.params["batch_size"], shuffle=False
        )
        return test_loader

    def poison_test_dataset_ind(self):
        logger.info("get poison test loader")
        test_classes = {}
        for ind, x in enumerate(self.test_dataset_poisoned):
            _, label = x
            if label in test_classes:
                test_classes[label].append(ind)
            else:
                test_classes[label] = [ind]
        range_num_id = list(range(0, len(self.test_dataset_poisoned)))
        for image_ind in test_classes[self.target]:
            if image_ind in range_num_id:
                range_num_id.remove(image_ind)
        poison_label_inds = test_classes[self.target]
        return (
            torch.utils.data.DataLoader(
                self.test_dataset_poisoned,
                batch_size=self.params["batch_size"],
                sampler=torch.utils.data.sampler.SubsetRandomSampler(range_num_id),
            ),
            torch.utils.data.DataLoader(
                self.test_dataset_poisoned,
                batch_size=self.params["batch_size"],
                sampler=torch.utils.data.sampler.SubsetRandomSampler(poison_label_inds),
            ),
        )

    def random_get_clients(self, epoch):
        adversarial_name_keys = []
        if self.params["is_poison"] is True:
            if self.params["poison_type"] == "single_attack":
                for idx in range(0, len(self.params["adversary_list"])):
                    if epoch in self.params[str(idx) + "_poison_epochs"]:
                        if self.params["adversary_list"][idx] not in adversarial_name_keys:
                            adversarial_name_keys.append(self.params["adversary_list"][idx])
                non_attacker = []
                for adv in self.params["adversary_list"]:
                    if adv not in adversarial_name_keys:
                        non_attacker.append(copy.deepcopy(adv))
                benign_num = self.params["num_clients"] - len(adversarial_name_keys)
                random_agent_name_keys = random.sample(
                    self.benign_namelist + non_attacker, benign_num
                )
                agent_name_keys = adversarial_name_keys + random_agent_name_keys
            else:
                if epoch <= self.params["start_poison"]:
                    agent_name_keys = random.sample(
                        self.benign_namelist, self.params["num_clients"]
                    )
                else:
                    agent_name_keys = random.sample(
                        self.participants_list, self.params["num_clients"]
                    )
                for client in agent_name_keys:
                    if client in self.adversarial_namelist:
                        adversarial_name_keys.append(client)
        else:
            agent_name_keys = random.sample(self.benign_namelist, self.params["num_clients"])
        return (agent_name_keys, adversarial_name_keys)

    def _init_balanced_schedule(self):
        total = int(self.params["number_of_total_participants"])
        k = int(self.params["num_clients"])
        rounds = int(self.params["epochs"])
        ids = list(self.participants_list)
        rng = random.Random(int(self.params.get("seed", 1)))
        total_slots = rounds * k
        base = total_slots // total
        extra = total_slots % total
        shuffled_once = ids.copy()
        rng.shuffle(shuffled_once)
        bonus_ids = set(shuffled_once[:extra])
        targets = {cid: base + (1 if cid in bonus_ids else 0) for cid in ids}
        remaining = targets.copy()
        schedule = []
        for r in range(rounds):
            available = [cid for cid in ids if remaining[cid] > 0]
            available.sort(key=lambda cid: (-remaining[cid], cid))
            top = available[:k]
            if len(top) < k:
                raise RuntimeError(
                    f"Balanced schedule could not fill round {r + 1}: only {len(top)} clients available for K={k}"
                )
            rng.shuffle(top)
            batch = top[:k]
            for cid in batch:
                remaining[cid] -= 1
            schedule.append(batch)
        counts = {cid: targets[cid] - remaining[cid] for cid in ids}
        self._balanced_schedule = schedule
        self.participation_counts = {cid: 0 for cid in ids}
        ti_vals = list(counts.values())
        self.schedule_type = "legacy_balanced"
        logger.info(
            f"[balanced_schedule] initialized N={total}, K={k}, T={rounds}, Ti_min={min(ti_vals)}, Ti_max={max(ti_vals)}, total_slots={total_slots}"
        )

    def _init_block_balanced_schedule(self):
        total = int(self.params["number_of_total_participants"])
        k = int(self.params["num_clients"])
        rounds = int(self.params["epochs"])
        if total % k != 0:
            raise ValueError(f"balanced_block requires N divisible by K, got N={total}, K={k}")
        rounds_per_block = total // k
        ids = list(self.participants_list)
        rng = random.Random(int(self.params.get("seed", 1)))
        schedule = []
        for block_start in range(0, rounds, rounds_per_block):
            permuted = ids.copy()
            rng.shuffle(permuted)
            groups = [permuted[i : i + k] for i in range(0, len(permuted), k)]
            take = min(rounds_per_block, rounds - block_start)
            for idx in range(take):
                if len(groups[idx]) != k:
                    raise RuntimeError(
                        f"balanced_block produced malformed group size={len(groups[idx])} for block={block_start // rounds_per_block}"
                    )
                schedule.append(list(groups[idx]))
        counts = {cid: 0 for cid in ids}
        for batch in schedule:
            for cid in batch:
                counts[cid] += 1
        self._balanced_schedule = schedule
        self.participation_counts = {cid: 0 for cid in ids}
        self.schedule_type = "balanced_block"
        ti_vals = list(counts.values())
        logger.info(
            f"[balanced_block] initialized N={total}, K={k}, T={rounds}, rounds_per_block={rounds_per_block}, Ti_min={min(ti_vals)}, Ti_max={max(ti_vals)}"
        )

    def get_clients_for_epoch(self, epoch):
        return self.random_get_clients(epoch)

    def build_dba_complete_shards_schedule(self):
        """Build a deterministic schedule containing every DBA shard in each round."""
        if not is_dba_attack(self.params):
            raise ValueError("DBA complete-shards schedule requires attack_mode=dba")
        trigger_num = int(self.params.get("trigger_num", 0))
        k = int(self.params.get("num_clients", 0))
        rounds = int(self.params.get("epochs", 0))
        malicious = sorted((int(c) for c in self.adversarial_namelist))
        benign = sorted((int(c) for c in self.benign_namelist))
        if rounds <= 0:
            raise ValueError("DBA complete-shards schedule requires epochs > 0")
        if trigger_num <= 0 or trigger_num > k:
            raise ValueError(
                f"DBA complete-shards schedule requires 0 < trigger_num <= num_clients; got trigger_num={trigger_num}, num_clients={k}"
            )
        pools = []
        for pattern_index in range(trigger_num):
            pool = [
                client_id
                for client_id in malicious
                if int(self.adversarial_pattern_map.get(client_id, -1)) == pattern_index
            ]
            if not pool:
                raise ValueError(
                    f"DBA complete-shards schedule has no malicious client for shard {pattern_index}"
                )
            pools.append(pool)
        invalid_mappings = {
            client_id: self.adversarial_pattern_map.get(client_id)
            for client_id in malicious
            if self.adversarial_pattern_map.get(client_id) not in range(trigger_num)
        }
        if invalid_mappings:
            raise ValueError(
                f"DBA complete-shards schedule has invalid client-to-shard mappings: {invalid_mappings}"
            )
        benign_per_round = k - trigger_num
        if benign_per_round > len(benign):
            raise ValueError(
                f"DBA complete-shards schedule needs {benign_per_round} benign clients per round but only {len(benign)} are available"
            )
        schedule = []
        for epoch in range(1, rounds + 1):
            representatives = [pool[(epoch - 1) % len(pool)] for pool in pools]
            round_rng = random.Random(int(self.params.get("seed", 1)) * 1000003 + epoch * 9176)
            round_clients = representatives + round_rng.sample(benign, benign_per_round)
            round_rng.shuffle(round_clients)
            schedule.append(round_clients)
        return schedule

    def create_model(self):
        model = None
        ds_type = self._canonical_type(self.params["type"])
        arch = str(self.params.get("arch", "")).lower().strip()
        if ds_type == "CIFAR10":
            if arch in ["", "resnet8"]:
                model = ResNet8()
            elif arch in ["resnet18", "resnet18_cifar"]:
                model = ResNet18(num_classes=10)
            elif arch == "resnet50_cifar":
                model = ResNet50Cifar(num_classes=10)
            else:
                raise ValueError(f"Unsupported CIFAR10 arch: {arch}")
        elif ds_type == "CIFAR100":
            model = ResNet18(num_classes=100)
        elif ds_type == "GTSRB":
            model = ResNet8(num_classes=43)
        if model is None:
            raise ValueError(f"Unsupported dataset/model combination: type={ds_type}, arch={arch}")
        model = model.to(device)
        self.model = model

    def preload_train_data_to_gpu(self):
        """Cache client training batches on the GPU."""
        self.gpu_train_data = {}
        for client_id, data_loader in self.train_data:
            client_data = []
            is_poisoned_client = client_id in self.adversarial_namelist
            for batch_idx, (images, labels) in enumerate(data_loader):
                if is_poisoned_client:
                    images, labels = poison_batch(
                        images,
                        labels,
                        self.params,
                        client_id,
                        self,
                        poison_rate=self.params["poison_rate"],
                    )
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                else:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                client_data.append((images, labels))
            self.gpu_train_data[client_id] = client_data
            poison_status = "(poisoned)" if is_poisoned_client else "(clean)"

    def preload_test_data_to_gpu(self):
        """Cache clean and poisoned test batches on the GPU."""
        from test import add_pixel_pattern

        self.gpu_test_data = []
        for batch_idx, (images, labels) in enumerate(self.test_data):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            self.gpu_test_data.append((images, labels))
        self.gpu_test_un_target_label_data = []
        for batch_idx, (images, labels) in enumerate(self.test_un_target_label_data):
            poisoned_images = []
            poisoned_labels = []
            for i in range(len(images)):
                poisoned_img, poisoned_label = add_pixel_pattern(images[i], self.params, self)
                poisoned_images.append(poisoned_img)
                poisoned_labels.append(poisoned_label)
            poisoned_images = torch.stack(poisoned_images).to(device, non_blocking=True)
            poisoned_labels = torch.tensor(poisoned_labels).to(device, non_blocking=True)
            self.gpu_test_un_target_label_data.append((poisoned_images, poisoned_labels))
        self.gpu_test_target_label_data = []
        for batch_idx, (images, labels) in enumerate(self.test_target_label_data):
            poisoned_images = []
            poisoned_labels = []
            for i in range(len(images)):
                poisoned_img, poisoned_label = add_pixel_pattern(images[i], self.params, self)
                poisoned_images.append(poisoned_img)
                poisoned_labels.append(poisoned_label)
            poisoned_images = torch.stack(poisoned_images).to(device, non_blocking=True)
            poisoned_labels = torch.tensor(poisoned_labels).to(device, non_blocking=True)
            self.gpu_test_target_label_data.append((poisoned_images, poisoned_labels))
        logger.info("Preloaded all test data (including pre-poisoned data) to GPU")

    def get_gpu_train_data(self, client_id):
        """Return cached training batches for a client."""
        if hasattr(self, "gpu_train_data") and client_id in self.gpu_train_data:
            return self.gpu_train_data[client_id]
        else:
            data_loader = next((data for cid, data in self.train_data if cid == client_id), None)
            return data_loader

    def get_gpu_test_data(self, poison=False, target_label=False):
        """Return cached clean or poisoned test batches."""
        if hasattr(self, "gpu_test_data"):
            if poison:
                if target_label:
                    return self.gpu_test_target_label_data
                else:
                    return self.gpu_test_un_target_label_data
            else:
                return self.gpu_test_data
        elif poison:
            if target_label:
                return self.test_target_label_data
            else:
                return self.test_un_target_label_data
        else:
            return self.test_data
