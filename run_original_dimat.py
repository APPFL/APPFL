"""
Run original DIMAT code for CIFAR-100, IID, 5 agents, Ring topology.
This is a self-contained script that uses the DIMAT/ codebase directly,
bypassing the unused 'clip' and 'fvcore' imports.

Usage:
    python run_original_dimat.py [--pretrain_epochs 100] [--merge_rounds 100]
"""

import sys
import os
import argparse
import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json

# Add DIMAT to path
DIMAT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DIMAT")
sys.path.insert(0, DIMAT_DIR)

# Patch out unused imports before importing DIMAT modules
import types

# Create stub modules for clip, fvcore, einops
for mod_name in ["clip", "fvcore", "fvcore.nn", "fvcore.nn.flop_count", "einops"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)
        if mod_name == "fvcore.nn.flop_count":
            sys.modules[mod_name].flop_count = lambda *a, **kw: None

# Now import DIMAT modules
from Models.resnetzip import resnet20
from data import Data_Loader
from dataloader import get_partition_dataloader
from utils.am_utils import reset_bn_stats, SpaceInterceptor
from utils.model_merger import ModelMerge
from graphs.resnet_graph import resnet20 as resnet20_graph
from utils.matching_functions import match_tensors_permute
from utils.metric_calculators import CovarianceMetric, MeanMetric


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(int(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(model, epochs, train_loader, device, optimizer_type="adam"):
    """Train model for given epochs (original DIMAT training)."""
    model.train()
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            output = F.log_softmax(output, dim=1)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={running_loss/(batch_idx+1):.4f}, "
                  f"train_acc={100.*correct/total:.2f}%")


def test_accuracy(model, dataloader, device):
    """Evaluate model accuracy."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    acc = 100.0 * correct / total
    avg_loss = test_loss / total
    return acc, avg_loss


def DIMAT_merge(models, cdict, graph_func, device, trainloader, save_dir):
    """Run DIMAT merge for ring topology (exp=2, not FC)."""
    match_func = match_tensors_permute
    metric_classes = (CovarianceMetric, MeanMetric)
    merged_models = []

    for m_idx in range(len(models)):
        print(f"  Merging agent {m_idx}...")
        neighbors = [i for i, val in enumerate(cdict["connectivity"][m_idx]) if val > 0]
        pi_m_idx = cdict["pi"][m_idx]
        merged_models.append(copy.deepcopy(models[m_idx]))

        temp_models = []
        interp_w = []

        # Reset BN for self
        models[m_idx] = reset_bn_stats(models[m_idx], trainloader)
        temp_models.append(copy.deepcopy(models[m_idx]))
        interp_w.append(pi_m_idx[m_idx])

        # Reset BN for neighbors
        for neighbor_id in neighbors:
            if neighbor_id == m_idx:
                continue
            models[neighbor_id] = reset_bn_stats(models[neighbor_id], trainloader)
            temp_models.append(copy.deepcopy(models[neighbor_id]))
            interp_w.append(pi_m_idx[neighbor_id])

        covsave_path = os.path.join(save_dir, f"cov_agent_{m_idx}")
        corrsave_path = os.path.join(save_dir, f"corr_agent_{m_idx}")

        graphs = [graph_func(agent).graphify() for agent in temp_models]
        del temp_models

        Merge = ModelMerge(*graphs, device=device)
        Merge.transform(
            merged_models[m_idx],
            trainloader,
            covsave_path=covsave_path,
            corrsave_path=corrsave_path,
            transform_fn=match_func,
            metric_classes=metric_classes,
            stop_at=None,
            interp_w=interp_w,
            **{"a": 0.0001, "b": 0.075},
        )
        del graphs

        # Reset BN after merge (line 171 of original main.py)
        reset_bn_stats(Merge.to(device), trainloader)
        new_state_dict = copy.deepcopy(Merge.merged_model.state_dict())
        merged_models[m_idx].load_state_dict(new_state_dict)
        del Merge
        del new_state_dict

    return merged_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--merge_rounds", type=int, default=100)
    parser.add_argument("--train_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_models", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data (original DIMAT data loading)
    os.chdir(DIMAT_DIR)
    data_loader = Data_Loader("cifar100", args.batch_size)
    trainset, testset, trainloader, testloader, num_classes = data_loader.load_data()

    # Load ring connectivity (exp=2)
    with open("Connectivity/5_2.json", "r") as f:
        cdict = json.load(f)
    print(f"Topology: {cdict['graph_type']}")
    print(f"Agents: {args.num_models}, Ring, IID")

    save_dir = os.path.join(DIMAT_DIR, "appfl_repro_results")
    os.makedirs(save_dir, exist_ok=True)

    # ===== Phase 1: Pre-train =====
    print(f"\n{'='*60}")
    print(f"Phase 1: Pre-training {args.num_models} models for {args.pretrain_epochs} epochs")
    print(f"{'='*60}")

    models = []
    for i in range(args.num_models):
        torch.manual_seed(i + int(args.seed))
        model = resnet20(w=8, num_classes=num_classes).to(device)
        print(f"\n[Agent {i}] Pre-training...")
        agent_trainloader = get_partition_dataloader(
            trainset, "iid", args.batch_size, args.num_models, "cifar100", i, "pretrain"
        )
        train_model(model, args.pretrain_epochs, agent_trainloader, device)

        # Test after pre-training
        acc, loss = test_accuracy(model, testloader, device)
        print(f"[Agent {i}] Pre-train test accuracy: {acc:.2f}%")
        models.append(model)

    # ===== Phase 2: Merge-Train =====
    print(f"\n{'='*60}")
    print(f"Phase 2: Merge-Train for {args.merge_rounds} rounds")
    print(f"{'='*60}")

    for round_idx in range(args.merge_rounds):
        print(f"\n--- Round {round_idx + 1}/{args.merge_rounds} ---")
        t0 = time.time()

        # Merge
        models = DIMAT_merge(models, cdict, resnet20_graph, device, trainloader, save_dir)

        # Evaluate after merge (pre-validation)
        accs = []
        for i, model in enumerate(models):
            acc, loss = test_accuracy(model, testloader, device)
            accs.append(acc)
        avg_acc = np.mean(accs)
        print(f"  [Post-merge] Avg accuracy: {avg_acc:.2f}% (individual: {[f'{a:.1f}' for a in accs]})")

        # Train
        if round_idx < args.merge_rounds - 1 or round_idx == 0:
            for i, model in enumerate(models):
                agent_trainloader = get_partition_dataloader(
                    trainset, "iid", args.batch_size, args.num_models, "cifar100", i, "finetune"
                )
                train_model(model, args.train_epochs, agent_trainloader, device)

            # Evaluate after training
            accs_post = []
            for i, model in enumerate(models):
                acc, loss = test_accuracy(model, testloader, device)
                accs_post.append(acc)
            avg_acc_post = np.mean(accs_post)
            print(f"  [Post-train] Avg accuracy: {avg_acc_post:.2f}%")

        elapsed = time.time() - t0
        print(f"  Round time: {elapsed:.1f}s")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
