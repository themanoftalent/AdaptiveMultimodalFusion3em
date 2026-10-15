from __future__ import annotations

import argparse
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.calibration import calibration_curve

from src.utils.io import ensure_dir, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reproducibility figures from saved outputs.")
    parser.add_argument("--config", default="configs/fusion_gate.yaml")
    args = parser.parse_args()
    _ = load_yaml(args.config)
    ensure_dir("figures")

    # Figure 1: proposed workflow diagram
    fig1_src = Path("figures/figure1.png")
    fig1_dst = Path("figures/Figure_1_Adaptive_Multimodal_Fusion_Architecture.png")
    if fig1_src.exists():
        shutil.copy(fig1_src, fig1_dst)
        print(f"Copied {fig1_src} to {fig1_dst} (Figure 1)")
    elif fig1_dst.exists():
        print(f"Figure 1 already exists at {fig1_dst}")
    else:
        print("Figure 1 source (figures/figure1.png) not found.")

    # Figure 2: Fusion Strategy Comparison
    baselines_path = Path("results/metrics/baseline_metrics_p004.json")
    gate_metrics_path = Path("results/metrics/adaptive_gate_p004_metrics.json")
    if baselines_path.exists() and gate_metrics_path.exists():
        with baselines_path.open("r", encoding="utf-8") as handle:
            baselines = json.load(handle)
        with gate_metrics_path.open("r", encoding="utf-8") as handle:
            gate_metrics = json.load(handle)
        
        # Combine metrics
        all_metrics = {}
        for model, m in baselines.items():
            all_metrics[model] = m
        all_metrics["adaptive_gate"] = gate_metrics
        
        # Sort methods as requested: baselines first, then adaptive_gate
        requested_order = [
            "attention_late_fusion",
            "fixed_alpha",
            "linear_svm",
            "logistic_regression",
            "max_confidence",
            "shallow_mlp",
            "xgboost",
            "adaptive_gate"
        ]
        
        display_map = {
            "attention_late_fusion": "Attention Late Fusion",
            "fixed_alpha": "Fixed Alpha",
            "linear_svm": "Linear SVM",
            "logistic_regression": "Logistic Regression",
            "max_confidence": "Max Confidence",
            "shallow_mlp": "Shallow MLP",
            "xgboost": "XGBoost",
            "adaptive_gate": "Adaptive Gate"
        }
        
        models = [m for m in requested_order if m in all_metrics]
        display_names = [display_map[m] for m in models]
        accuracies = [all_metrics[m].get("accuracy", 0) * 100 for m in models]
        macro_f1s = [all_metrics[m].get("macro_f1", 0) * 100 for m in models]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Accuracy bar chart
        color_acc = "#1f77b4"  # Slate Blue
        ax1.set_ylabel("Accuracy (%)", color=color_acc, fontweight="bold")
        bars = ax1.bar(display_names, accuracies, color=color_acc, alpha=0.6, label="Accuracy", width=0.5)
        ax1.tick_params(axis='y', labelcolor=color_acc)
        ax1.set_ylim(0, 105)
        
        # Twin axis for F1 line plot
        ax2 = ax1.twinx()
        color_f1 = "#d62728"  # Coral Red
        ax2.set_ylabel("Macro F1 (%)", color=color_f1, fontweight="bold")
        line = ax2.plot(display_names, macro_f1s, color=color_f1, marker='o', linewidth=2, markersize=8, label="Macro F1")
        ax2.tick_params(axis='y', labelcolor=color_f1)
        ax2.set_ylim(0, 105)
        
        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, color="blue")
                        
        # Add labels on the line points
        for i, val in enumerate(macro_f1s):
            ax2.annotate(f'{val:.1f}%',
                        xy=(i, val),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, color="red")
        
        ax1.set_xticks(np.arange(len(display_names)))
        ax1.set_xticklabels(display_names, rotation=25, ha="right", fontsize=9)
        
        plt.title("Figure 2: Fusion strategy comparison on the P004 held-out test set.", pad=20, fontweight="bold")
        fig.tight_layout()
        plt.savefig("figures/Figure_2_Fusion_Strategy_Comparison_P004.png", dpi=300)
        plt.close()
        print("Saved figures/Figure_2_Fusion_Strategy_Comparison_P004.png (Figure 2)")
    else:
        print("Baseline or gate metrics missing.")

    # Figure 3: Confusion Matrix
    pred_path = Path("results/probabilities/adaptive_gate_p004_predictions.csv")
    if pred_path.exists():
        df = pd.read_csv(pred_path)
        y_true = df["label"].to_numpy()
        y_pred = df["predicted_label"].to_numpy()
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        classes = ["Non tumor", "Viable tumor", "Necrotic tumor"]
        
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax, label="Number of Tiles")
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title="Figure 3: Confusion matrix for adaptive gating on the P004 held-out test set.",
               ylabel="True Label",
               xlabel="Predicted Label")
               
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=12, fontweight="bold")
        fig.tight_layout()
        plt.savefig("figures/Figure_3_Confusion_Matrix_Adaptive_Gating_P004.png", dpi=300)
        plt.close()
        print("Saved figures/Figure_3_Confusion_Matrix_Adaptive_Gating_P004.png (Figure 3)")
    else:
        print("P004 predictions not found.")

    # Figure 4: Adaptive Gating Weight Summary
    if pred_path.exists():
        frame = pd.read_csv(pred_path)
        if "alpha_hat" in frame.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
            
            # Panel (a): Histogram of alpha_hat
            ax1.hist(frame["alpha_hat"], bins=10, color="#aec7e8", edgecolor="black", alpha=0.8)
            ax1.set_xlabel("Radiograph-branch weight alpha_hat", fontsize=10)
            ax1.set_ylabel("Number of P004 test tiles", fontsize=10)
            ax1.set_title("(a) Distribution of predicted weights", fontsize=11, fontweight="bold")
            
            mean_alpha = frame["alpha_hat"].mean()
            sd_alpha = frame["alpha_hat"].std()
            ax1.text(0.05, 0.95, f"Mean: {mean_alpha:.2f}\nSD: {sd_alpha:.2f}", 
                     transform=ax1.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
            # Panel (b): Class-specific mean weights
            classes = ["Non tumor", "Viable tumor", "Necrotic tumor"]
            mean_weights = []
            for c in [0, 1, 2]:
                class_subset = frame[frame["label"] == c]
                if not class_subset.empty:
                    mean_weights.append(class_subset["alpha_hat"].mean())
                else:
                    mean_weights.append(0.0)
            
            colors = ["#2ca02c", "#ff7f0e", "#9467bd"]
            bars = ax2.bar(classes, mean_weights, color=colors, edgecolor="black", alpha=0.8, width=0.6)
            ax2.set_ylabel("Mean alpha_hat", fontsize=10)
            ax2.set_ylim(0, 1.0)
            ax2.set_title("(b) Class-specific mean weights", fontsize=11, fontweight="bold")
            
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight="bold")
            
            fig.suptitle("Figure 4: Adaptive gating weight summary on the P004 held-out test set.", fontsize=13, fontweight="bold", y=0.98)
            fig.tight_layout()
            plt.savefig("figures/Figure_4_Adaptive_Gating_Weight_Summary_P004.png", dpi=300)
            plt.close()
            print("Saved figures/Figure_4_Adaptive_Gating_Weight_Summary_P004.png (Figure 4)")
    else:
        print("P004 predictions not found.")

    # Figure 5: External Validation on Hitit Cohort
    # Hitit University Faculty of Medicine data are restricted and not redistributed.
    # This figure is generated only when an authorized user supplies local prediction files.
    # No synthetic or placeholder Hitit results are generated.
    hitit_pred_path = Path("results/probabilities/hitit_predictions_authorized.csv")
    hitit_meta_path = Path("data/hitit_authorized_metadata.csv")

    if hitit_pred_path.exists() and hitit_meta_path.exists():
        meta = pd.read_csv(hitit_meta_path)
        pred_df = pd.read_csv(hitit_pred_path)
        if "binary_label" not in meta.columns or "prob_tumor" not in pred_df.columns:
            print(
                "Figure 5 skipped: authorized metadata must contain 'binary_label' and "
                "authorized predictions must contain 'prob_tumor'. "
                "See scripts/10_external_hitit_interface.py."
            )
        else:
            y_true_h = meta["binary_label"].to_numpy(dtype=int)
            y_prob_h = pred_df["prob_tumor"].to_numpy(dtype=float)
            y_pred_h = (y_prob_h >= 0.5).astype(int)

            fig, axs = plt.subplots(2, 2, figsize=(12, 10))

            fpr_h, tpr_h, _ = roc_curve(y_true_h, y_prob_h)
            roc_auc_h = auc(fpr_h, tpr_h)
            axs[0, 0].plot(fpr_h, tpr_h, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_h:.2f})')
            axs[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axs[0, 0].set_xlim([0.0, 1.0])
            axs[0, 0].set_ylim([0.0, 1.05])
            axs[0, 0].set_xlabel('False Positive Rate')
            axs[0, 0].set_ylabel('True Positive Rate')
            axs[0, 0].set_title('(a) ROC Curve')
            axs[0, 0].legend(loc="lower right")
            axs[0, 0].grid(True, linestyle=':', alpha=0.6)

            precision_h, recall_h, _ = precision_recall_curve(y_true_h, y_prob_h)
            pr_auc_h = auc(recall_h, precision_h)
            axs[0, 1].plot(recall_h, precision_h, color='green', lw=2, label=f'PR curve (AUC = {pr_auc_h:.2f})')
            axs[0, 1].set_xlim([0.0, 1.0])
            axs[0, 1].set_ylim([0.0, 1.05])
            axs[0, 1].set_xlabel('Recall')
            axs[0, 1].set_ylabel('Precision')
            axs[0, 1].set_title('(b) Precision-Recall Curve')
            axs[0, 1].legend(loc="lower left")
            axs[0, 1].grid(True, linestyle=':', alpha=0.6)

            cm_h = confusion_matrix(y_true_h, y_pred_h)
            axs[1, 0].imshow(cm_h, interpolation='nearest', cmap=plt.cm.Greens)
            hit_classes = ["Non tumor", "Tumor"]
            axs[1, 0].set_xticks([0, 1])
            axs[1, 0].set_yticks([0, 1])
            axs[1, 0].set_xticklabels(hit_classes)
            axs[1, 0].set_yticklabels(hit_classes)
            axs[1, 0].set_xlabel('Predicted Label')
            axs[1, 0].set_ylabel('True Label')
            axs[1, 0].set_title('(c) Confusion Matrix')
            thresh_h = cm_h.max() / 2.
            for i in range(cm_h.shape[0]):
                for j in range(cm_h.shape[1]):
                    axs[1, 0].text(j, i, format(cm_h[i, j], 'd'),
                                   ha="center", va="center",
                                   color="white" if cm_h[i, j] > thresh_h else "black",
                                   fontsize=14, fontweight="bold")

            prob_true_c, prob_pred_c = calibration_curve(y_true_h, y_prob_h, n_bins=5)
            axs[1, 1].plot(prob_pred_c, prob_true_c, marker='o', linewidth=2, color='purple', label='Model')
            axs[1, 1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
            axs[1, 1].set_xlabel('Mean Predicted Probability')
            axs[1, 1].set_ylabel('Fraction of Positives')
            axs[1, 1].set_title('(d) Calibration Reliability Curve')
            axs[1, 1].legend(loc="upper left")
            axs[1, 1].grid(True, linestyle=':', alpha=0.6)

            fig.suptitle(
                "Figure 5: External validation on the Hitit University Faculty of Medicine cohort.",
                fontsize=14, fontweight="bold", y=0.98,
            )
            fig.tight_layout()
            plt.savefig("figures/Figure_5_External_Validation_Hitit_Cohort.png", dpi=300)
            plt.close()
            print("Saved figures/Figure_5_External_Validation_Hitit_Cohort.png (Figure 5)")
    else:
        print(
            "Figure 5 skipped: Hitit University data are restricted and not included in this repository. "
            "Authorized users must supply 'results/probabilities/hitit_predictions_authorized.csv' "
            "and 'data/hitit_authorized_metadata.csv', then run scripts/10_external_hitit_interface.py. "
            "No synthetic Hitit results are generated by this script."
        )


if __name__ == "__main__":
    main()
