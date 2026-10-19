#!/usr/bin/env python3
"""
train_act_pusht.py

Train Action Chunking Transformer (ACT) from LeRobot on the Push-T Zarr dataset.
Splits episodes 80:20 into train and test sets.
Normalizes observations and actions based on train-set statistics.
Saves model weights, configuration, and dataset statistics for inference.
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import zarr

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


class PushTZarrDataset(Dataset):
    """
    PyTorch Dataset for Push-T Action Chunking.
    Constructed over a given subset of episode indices.
    """

    def __init__(
        self,
        zarr_path: str,
        episode_indices: list[int],
        chunk_size: int = 16,
        stats: dict | None = None,
    ):
        self.chunk_size = chunk_size
        self.episode_indices = episode_indices

        store = zarr.open(zarr_path, mode="r")
        all_actions = store["data"]["action"][:]  # (N, 2)
        all_states = store["data"]["state"][:]  # (N, 5)
        all_images = store["data"]["img"][:]  # (N, 96, 96, 3)
        all_episode_ends = store["meta"]["episode_ends"][:]  # (num_episodes,)

        # Build list of episodes: (start_idx, end_idx)
        self.samples = []  # list of (global_t, ep_start, ep_end)
        self.actions = []
        self.states = []
        self.images = []

        for ep_idx in episode_indices:
            ep_start = 0 if ep_idx == 0 else all_episode_ends[ep_idx - 1]
            ep_end = all_episode_ends[ep_idx]
            for t in range(ep_start, ep_end):
                self.samples.append((t, ep_start, ep_end))

        self.raw_actions = all_actions
        self.raw_states = all_states
        self.raw_images = all_images

        # Compute or assign normalization statistics
        if stats is None:
            # Gather all train-set states and actions to compute stats
            train_states = np.concatenate(
                [
                    all_states[
                        (0 if ep == 0 else all_episode_ends[ep - 1]) : all_episode_ends[ep]
                    ]
                    for ep in episode_indices
                ],
                axis=0,
            )
            train_actions = np.concatenate(
                [
                    all_actions[
                        (0 if ep == 0 else all_episode_ends[ep - 1]) : all_episode_ends[ep]
                    ]
                    for ep in episode_indices
                ],
                axis=0,
            )

            self.stats = {
                "state_mean": train_states.mean(axis=0).tolist(),
                "state_std": np.clip(train_states.std(axis=0), a_min=1e-4, a_max=None).tolist(),
                "action_mean": train_actions.mean(axis=0).tolist(),
                "action_std": np.clip(train_actions.std(axis=0), a_min=1e-4, a_max=None).tolist(),
            }
        else:
            self.stats = stats

        self.state_mean = np.array(self.stats["state_mean"], dtype=np.float32)
        self.state_std = np.array(self.stats["state_std"], dtype=np.float32)
        self.action_mean = np.array(self.stats["action_mean"], dtype=np.float32)
        self.action_std = np.array(self.stats["action_std"], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        t, ep_start, ep_end = self.samples[idx]

        # 1. Observation image: (96, 96, 3) -> (3, 96, 96), float32 in [0, 1]
        img = self.raw_images[t].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        # 2. Observation state: normalized (5,)
        state = (self.raw_states[t].astype(np.float32) - self.state_mean) / self.state_std

        # 3. Action chunk: slice [t, t + chunk_size]
        actions_chunk = np.zeros((self.chunk_size, 2), dtype=np.float32)
        action_is_pad = np.zeros((self.chunk_size,), dtype=bool)

        slice_end = min(t + self.chunk_size, ep_end)
        valid_len = slice_end - t

        raw_chunk = self.raw_actions[t:slice_end].astype(np.float32)
        norm_chunk = (raw_chunk - self.action_mean) / self.action_std
        actions_chunk[:valid_len] = norm_chunk

        if valid_len < self.chunk_size:
            # Pad with the last action in the episode
            last_action_norm = norm_chunk[-1]
            actions_chunk[valid_len:] = last_action_norm
            action_is_pad[valid_len:] = True

        return {
            "observation.image": torch.from_numpy(img),
            "observation.state": torch.from_numpy(state),
            "action": torch.from_numpy(actions_chunk),
            "action_is_pad": torch.from_numpy(action_is_pad),
        }


def split_episodes(total_episodes: int, train_ratio: float = 0.8, seed: int = 42):
    """Split episode indices into train and test sets."""
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(total_episodes).tolist()
    n_train = int(total_episodes * train_ratio)
    train_indices = sorted(shuffled_indices[:n_train])
    test_indices = sorted(shuffled_indices[n_train:])
    return train_indices, test_indices


def main():
    parser = argparse.ArgumentParser(description="Train LeRobot ACT on Push-T dataset")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset/pusht/pusht_cchi_v7_replay.zarr",
        help="Path to Push-T zarr dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tutorials/lerobot_implement/outputs/train_act_pusht_local",
        help="Output directory to save model and stats",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Transformer learning rate")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Vision backbone learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--chunk_size", type=int, default=16, help="Action chunk size")
    parser.add_argument("--dim_model", type=int, default=256, help="Hidden dimension of transformer")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dim_feedforward", type=int, default=1024, help="FFN dimension")
    parser.add_argument("--n_encoder_layers", type=int, default=2, help="Encoder layers")
    parser.add_argument("--n_decoder_layers", type=int, default=1, help="Decoder layers")
    parser.add_argument("--latent_dim", type=int, default=16, help="VAE latent dimension")
    parser.add_argument("--kl_weight", type=float, default=10.0, help="KL loss weight")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Inspect episodes and split 80:20
    store = zarr.open(args.dataset_path, mode="r")
    episode_ends = store["meta"]["episode_ends"][:]
    total_episodes = len(episode_ends)
    train_indices, test_indices = split_episodes(total_episodes, train_ratio=0.8, seed=args.seed)

    print(f"Total episodes: {total_episodes}")
    print(f"Train episodes: {len(train_indices)} ({len(train_indices)/total_episodes*100:.1f}%)")
    print(f"Test episodes:  {len(test_indices)} ({len(test_indices)/total_episodes*100:.1f}%)")

    # 2. Build datasets and data loaders
    train_dataset = PushTZarrDataset(
        args.dataset_path, train_indices, chunk_size=args.chunk_size
    )
    test_dataset = PushTZarrDataset(
        args.dataset_path, test_indices, chunk_size=args.chunk_size, stats=train_dataset.stats
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Train transitions: {len(train_dataset)}, Batches: {len(train_loader)}")
    print(f"Test transitions:  {len(test_dataset)}, Batches: {len(test_loader)}")

    # 3. Configure LeRobot ACT Policy
    config = ACTConfig(
        input_features={
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(5,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        },
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        dim_model=args.dim_model,
        n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        use_vae=True,
        latent_dim=args.latent_dim,
        kl_weight=args.kl_weight,
        optimizer_lr=args.lr,
        optimizer_lr_backbone=args.lr_backbone,
        optimizer_weight_decay=args.weight_decay,
    )

    policy = ACTPolicy(config).to(device)

    # 4. Setup Optimizer
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 5. Save split metadata and normalization stats
    meta_info = {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "chunk_size": args.chunk_size,
        "stats": train_dataset.stats,
    }
    with open(os.path.join(args.output_dir, "dataset_meta.json"), "w") as f:
        json.dump(meta_info, f, indent=2)

    # 6. Training Loop
    best_val_loss = float("inf")
    print("\nStarting ACT Training...")

    for epoch in range(1, args.epochs + 1):
        policy.train()
        train_loss = 0.0
        train_l1 = 0.0
        train_kld = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            loss, loss_dict = policy(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
            optimizer.step()

            train_loss += loss.item()
            train_l1 += loss_dict.get("l1_loss", 0.0)
            train_kld += loss_dict.get("kld_loss", 0.0)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "l1": f"{loss_dict.get('l1_loss', 0.0):.4f}",
                "kld": f"{loss_dict.get('kld_loss', 0.0):.4f}",
            })

        avg_train_loss = train_loss / len(train_loader)
        avg_train_l1 = train_l1 / len(train_loader)
        avg_train_kld = train_kld / len(train_loader)

        # Validation on test set (eval mode)
        policy.eval()
        val_l1 = 0.0

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                pred_actions = policy.predict_action_chunk(batch)
                l1 = (
                    torch.nn.functional.l1_loss(batch["action"], pred_actions, reduction="none")
                    * ~batch["action_is_pad"].unsqueeze(-1)
                ).mean()
                val_l1 += l1.item()

        avg_val_loss = val_l1 / len(test_loader)

        print(
            f"Epoch {epoch:02d} Summary: "
            f"Train Loss: {avg_train_loss:.4f} (L1: {avg_train_l1:.4f}, KLD: {avg_train_kld:.4f}) | "
            f"Val L1 Loss: {avg_val_loss:.4f}"
        )

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "val_loss": avg_val_loss,
        }
        torch.save(checkpoint, os.path.join(args.output_dir, "checkpoint_last.pt"))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, os.path.join(args.output_dir, "checkpoint_best.pt"))
            print(f"[*] New best validation loss: {best_val_loss:.4f} - Saved checkpoint_best.pt")

    print("\nTraining complete! Artifacts saved to:", args.output_dir)


if __name__ == "__main__":
    main()
