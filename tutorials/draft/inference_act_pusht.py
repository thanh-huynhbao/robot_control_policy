#!/usr/bin/env python3
"""
inference_act_pusht.py

Run inference using a trained ACT policy on the Push-T test dataset.
Loads the checkpoint and normalization stats saved by train_act_pusht.py.
Computes evaluation error metrics (MSE, RMSE, MAE, Max Error) between ground truth
and predicted actions across test episodes, and generates visual comparison plots.
"""

import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import zarr

from lerobot.policies.act.modeling_act import ACTPolicy


def load_model_and_metadata(checkpoint_path: str, meta_path: str, device: torch.device):
    """Load trained ACTPolicy, configuration, and dataset metadata."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    with open(meta_path, "r") as f:
        meta_info = json.load(f)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    policy = ACTPolicy(config).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    return policy, config, meta_info


def run_episode_inference(
    policy: ACTPolicy,
    ep_images: np.ndarray,
    ep_states: np.ndarray,
    stats: dict,
    device: torch.device,
) -> np.ndarray:
    """
    Run step-by-step policy rollout on an episode.
    Uses policy.reset() and policy.select_action().
    """
    state_mean = np.array(stats["state_mean"], dtype=np.float32)
    state_std = np.array(stats["state_std"], dtype=np.float32)
    action_mean = np.array(stats["action_mean"], dtype=np.float32)
    action_std = np.array(stats["action_std"], dtype=np.float32)

    T = len(ep_images)
    predicted_actions = []

    # Reset internal action queue/temporal ensembler
    policy.reset()

    with torch.no_grad():
        for t in range(T):
            # Preprocess observation
            img = ep_images[t].astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # (3, 96, 96)
            img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)  # (1, 3, 96, 96)

            state = (ep_states[t].astype(np.float32) - state_mean) / state_std
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)  # (1, 5)

            batch = {
                "observation.image": img_tensor,
                "observation.state": state_tensor,
            }

            # Query policy
            pred_norm_action = policy.select_action(batch)  # (1, 2) or (2,)
            pred_norm_action = pred_norm_action.cpu().numpy().squeeze(0)

            # Unnormalize
            pred_action = pred_norm_action * action_std + action_mean
            predicted_actions.append(pred_action)

    return np.array(predicted_actions)


def compute_metrics(ground_truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute MSE, RMSE, MAE, and per-axis errors between GT and predictions."""
    diff = predicted - ground_truth
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))
    max_err = np.max(np.abs(diff))

    mae_x = np.mean(np.abs(diff[:, 0]))
    mae_y = np.mean(np.abs(diff[:, 1]))

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAE_X": float(mae_x),
        "MAE_Y": float(mae_y),
        "Max_Error": float(max_err),
    }


def plot_comparison(
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    episode_idx: int,
    metrics: dict[str, float],
    output_path: str,
):
    """Plot 2D trajectory and time-series comparison between GT and predicted actions."""
    fig = plt.figure(figsize=(15, 6))

    # Subplot 1: 2D Trajectory
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(gt_actions[:, 0], gt_actions[:, 1], "b-", linewidth=2.0, label="Ground Truth", alpha=0.8)
    ax1.plot(pred_actions[:, 0], pred_actions[:, 1], "r--", linewidth=2.0, label="ACT Prediction", alpha=0.8)

    # Mark Start and End
    ax1.scatter(gt_actions[0, 0], gt_actions[0, 1], c="green", s=90, marker="o", label="Start", zorder=5)
    ax1.scatter(gt_actions[-1, 0], gt_actions[-1, 1], c="blue", s=90, marker="x", label="GT End", zorder=5)
    ax1.scatter(pred_actions[-1, 0], pred_actions[-1, 1], c="red", s=90, marker="^", label="Pred End", zorder=5)

    ax1.set_title(f"Test Episode {episode_idx}: 2D Action Trajectory", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Action Dim 0 (X coordinate)", fontsize=10)
    ax1.set_ylabel("Action Dim 1 (Y coordinate)", fontsize=10)
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Subplot 2: Time Series for X and Y
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(gt_actions[:, 0], "b-", label="GT X", linewidth=1.8)
    ax2.plot(pred_actions[:, 0], "r--", label="Pred X", linewidth=1.8)
    ax2.set_title(f"Action Dim 0 (X) [MAE: {metrics['MAE_X']:.2f}]", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Position X", fontsize=9)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.6)

    ax3 = fig.add_subplot(2, 2, 4)
    ax3.plot(gt_actions[:, 1], "b-", label="GT Y", linewidth=1.8)
    ax3.plot(pred_actions[:, 1], "r--", label="Pred Y", linewidth=1.8)
    ax3.set_title(f"Action Dim 1 (Y) [MAE: {metrics['MAE_Y']:.2f}]", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Time Step", fontsize=10)
    ax3.set_ylabel("Position Y", fontsize=9)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle(
        f"ACT Push-T Evaluation (Episode {episode_idx}) | Overall MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved inference visualization plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ACT policy on Push-T test episodes")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset/pusht/pusht_cchi_v7_replay.zarr",
        help="Path to Push-T zarr dataset",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="tutorials/lerobot_implement/outputs/train_act_pusht_local",
        help="Directory containing checkpoint_best.pt and dataset_meta.json",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="checkpoint_best.pt",
        help="Name of checkpoint file",
    )
    parser.add_argument(
        "--eval_episode",
        type=int,
        default=None,
        help="Specific test episode index to plot (default: first episode in test set)",
    )
    parser.add_argument(
        "--eval_all",
        action="store_true",
        default=True,
        help="Compute aggregate metrics across all test episodes",
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default="tutorials/lerobot_implement/outputs/act_test_inference_eval.png",
        help="Output path for plot image",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    meta_path = os.path.join(args.checkpoint_dir, "dataset_meta.json")

    # 1. Load model and metadata
    policy, config, meta_info = load_model_and_metadata(checkpoint_path, meta_path, device)
    test_indices = meta_info["test_indices"]
    stats = meta_info["stats"]

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Test episodes count: {len(test_indices)} (indices: {test_indices[:5]} ...)")

    # 2. Open dataset
    store = zarr.open(args.dataset_path, mode="r")
    actions = store["data"]["action"][:]
    states = store["data"]["state"][:]
    images = store["data"]["img"][:]
    episode_ends = store["meta"]["episode_ends"][:]

    # 3. Evaluate across test episodes
    all_metrics = []
    target_plot_ep = args.eval_episode if args.eval_episode is not None else test_indices[0]
    plot_gt, plot_pred, plot_metrics = None, None, None

    print("\nRunning inference across test set...")
    for ep_idx in tqdm(test_indices, desc="Evaluating Test Episodes"):
        ep_start = 0 if ep_idx == 0 else episode_ends[ep_idx - 1]
        ep_end = episode_ends[ep_idx]

        ep_gt_actions = actions[ep_start:ep_end]
        ep_states = states[ep_start:ep_end]
        ep_images = images[ep_start:ep_end]

        ep_pred_actions = run_episode_inference(policy, ep_images, ep_states, stats, device)
        ep_metrics = compute_metrics(ep_gt_actions, ep_pred_actions)
        all_metrics.append(ep_metrics)

        if ep_idx == target_plot_ep:
            plot_gt = ep_gt_actions
            plot_pred = ep_pred_actions
            plot_metrics = ep_metrics

    # 4. Aggregate metrics
    mean_mse = np.mean([m["MSE"] for m in all_metrics])
    mean_rmse = np.mean([m["RMSE"] for m in all_metrics])
    mean_mae = np.mean([m["MAE"] for m in all_metrics])
    mean_mae_x = np.mean([m["MAE_X"] for m in all_metrics])
    mean_mae_y = np.mean([m["MAE_Y"] for m in all_metrics])
    max_err = np.max([m["Max_Error"] for m in all_metrics])

    print("\n" + "=" * 55)
    print("           TEST SET EVALUATION METRICS SUMMARY           ")
    print("=" * 55)
    print(f"Evaluated Test Episodes: {len(test_indices)}")
    print(f"Mean Squared Error (MSE):      {mean_mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {mean_rmse:.4f}")
    print(f"Mean Absolute Error (MAE):     {mean_mae:.4f}")
    print(f"  - Action X MAE:              {mean_mae_x:.4f}")
    print(f"  - Action Y MAE:              {mean_mae_y:.4f}")
    print(f"Max Absolute Error:            {max_err:.4f}")
    print("=" * 55)

    # 5. Plot for target episode
    if plot_gt is not None and plot_pred is not None:
        plot_comparison(plot_gt, plot_pred, target_plot_ep, plot_metrics, args.output_plot)


if __name__ == "__main__":
    main()
