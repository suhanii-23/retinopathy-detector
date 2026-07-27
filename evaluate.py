"""Evaluate a trained diabetic retinopathy model against the saved
validation split.

Usage:
    python evaluate.py --model output/diabetic_retinopathy_model.keras \
        --val-split output/val_split.npz --history output/history.json

Quadratic Weighted Kappa (QWK), not accuracy, is treated as the primary
metric here for two reasons:

1. The labels are ordinal (No DR < Mild < Moderate < Severe <
   Proliferative). Confusing No DR with Proliferative is a far worse
   mistake than confusing Mild with Moderate, but plain accuracy scores
   both errors identically. QWK weights each error by (i-j)^2, so it
   penalizes distant misclassifications much more than adjacent ones.
2. The class distribution is skewed — about 49% of images are No DR —
   so a model that always predicts "No DR" scores ~49% accuracy while
   being clinically useless. Kappa is chance-corrected, so that
   majority-class baseline scores close to 0. QWK is also the official
   metric of the APTOS 2019 Kaggle competition this dataset is from.
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from tensorflow.keras.models import load_model

from model import CLASS_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Path to the saved .keras model')
    parser.add_argument('--val-split', required=True, help='Path to val_split.npz produced by train.py')
    parser.add_argument('--history', required=True, help='Path to history.json produced by train.py')
    return parser.parse_args()


def plot_confusion_matrix(y_true, y_pred, out_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    # Row-normalized = per-class recall. Raw counts alone are unreadable
    # here because the No DR row dwarfs every other row.
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, data, title, fmt in [
        (axes[0], cm, 'Confusion Matrix (counts)', 'd'),
        (axes[1], cm_norm, 'Confusion Matrix (row-normalized / recall)', '.2f'),
    ]:
        im = ax.imshow(data, cmap='Blues')
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(title)
        thresh = data.max() / 2.0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, format(data[i, j], fmt), ha='center', va='center',
                         color='white' if data[i, j] > thresh else 'black')
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return cm


def plot_training_curves(history, out_path='training_curves.png'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history['accuracy'], label='train')
    axes[0].plot(history['val_accuracy'], label='val')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(history['loss'], label='train')
    axes[1].plot(history['val_loss'], label='val')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def referable_dr_metrics(y_true, y_pred):
    # Collapse to a binary decision: >= Moderate means "refer to an
    # ophthalmologist". This is the decision a screening tool actually
    # supports — a clinician doesn't need the model to distinguish Severe
    # from Proliferative, they need to know whether to refer at all. A
    # missed referral (false negative here) costs far more than an
    # unnecessary one (false positive), so sensitivity matters most.
    true_bin = (np.array(y_true) >= 2).astype(int)
    pred_bin = (np.array(y_pred) >= 2).astype(int)

    tp = int(np.sum((true_bin == 1) & (pred_bin == 1)))
    fn = int(np.sum((true_bin == 1) & (pred_bin == 0)))
    tn = int(np.sum((true_bin == 0) & (pred_bin == 0)))
    fp = int(np.sum((true_bin == 0) & (pred_bin == 1)))

    sensitivity = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    return {'sensitivity': sensitivity, 'specificity': specificity, 'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp}


def error_distance_histogram(y_true, y_pred):
    # Of the misclassifications, how many are 1/2/3/4 classes away. This
    # is exactly what QWK rewards and plain accuracy hides.
    distances = np.abs(np.array(y_true) - np.array(y_pred))
    misclassified = distances[distances > 0]
    return {str(d): int(np.sum(misclassified == d)) for d in range(1, 5)}


def main():
    args = parse_args()

    model = load_model(args.model)
    data = np.load(args.val_split)
    X_val, y_val = data['X_val'], data['y_val']

    with open(args.history) as f:
        history = json.load(f)

    probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    accuracy = float(np.mean(y_pred == y_val))
    qwk = float(cohen_kappa_score(y_val, y_pred, weights='quadratic'))

    print('=' * 40)
    print(f'  Validation Accuracy : {accuracy:.4f}')
    print(f'  Quadratic Weighted Kappa : {qwk:.4f}')
    print('=' * 40)

    report = classification_report(y_val, y_pred, target_names=CLASS_NAMES, digits=3, output_dict=True)
    print(classification_report(y_val, y_pred, target_names=CLASS_NAMES, digits=3))

    cm = plot_confusion_matrix(y_val, y_pred)
    plot_training_curves(history)

    referable = referable_dr_metrics(y_val, y_pred)
    error_hist = error_distance_histogram(y_val, y_pred)

    metrics = {
        'accuracy': accuracy,
        'qwk': qwk,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'referable_dr': referable,
        'error_distance_histogram': error_hist,
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
