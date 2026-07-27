"""Two-phase training script for the diabetic retinopathy classifier.

Usage:
    python train.py --data-dir /path/to/aptos2019 --output /path/to/output

Expects --data-dir to contain train.csv (columns: id_code, diagnosis) and
a train_images/ subdirectory of .png fundus images, matching the APTOS
2019 Kaggle layout.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import NUM_CLASSES, build_model, set_backbone_trainable
from preprocess import IMAGE_SIZE, preprocess_image

WARMUP_EPOCHS = 2
WARMUP_LR = 1e-3
FINE_TUNE_EPOCHS = 20
FINE_TUNE_LR = 1e-4
BATCH_SIZE = 16
VAL_SIZE = 0.15
RANDOM_STATE = 2006


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, help='Directory containing train.csv and train_images/')
    parser.add_argument('--output', required=True, help='Directory to write model, history, and val split to')
    return parser.parse_args()


def load_dataset(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    images_dir = os.path.join(data_dir, 'train_images')

    # Preprocessing is applied once, up front, to the full dataset before
    # the train/val split. This is leakage-free because preprocess_image
    # is per-image and stateless (crop/resize/high-pass/normalize) — it
    # never looks at any other row, so doing it before or after the split
    # produces identical results. It just avoids re-doing the same work
    # for both phases of training.
    X = np.stack([
        preprocess_image(os.path.join(images_dir, f'{image_id}.png'))
        for image_id in df['id_code']
    ])
    y = df['diagnosis'].to_numpy()
    return X, y


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    X, y = load_dataset(args.data_dir)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    np.savez(os.path.join(args.output, 'val_split.npz'), X_val=X_val, y_val=y_val)

    # Class weights are computed on the training split only. Computing
    # them on the full dataframe (train + val) would leak validation
    # class frequencies into the training objective.
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    y_train_cat = np.eye(NUM_CLASSES)[y_train]
    y_val_cat = np.eye(NUM_CLASSES)[y_val]

    # Fundus images have no canonical orientation — the eye is imaged
    # face-on and rotating or mirroring it does not change the label.
    # 360-degree rotation and both horizontal/vertical flips are
    # therefore label-preserving augmentations. Brightness/contrast
    # jitter is deliberately excluded: it would partially undo the Ben
    # Graham illumination normalization already applied in preprocessing.
    train_datagen = ImageDataGenerator(
        rotation_range=360,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=[0.98, 1.02],
        width_shift_range=0.01,
        height_shift_range=0.01,
    )
    train_gen = train_datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE)

    model = build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    # val_loss rather than val_accuracy: accuracy is a step function on
    # a 550-image validation set and jumps around from epoch to epoch,
    # while loss is continuous and gives a cleaner, less noisy signal
    # for early stopping and LR reduction.
    checkpoint_path = os.path.join(args.output, 'diabetic_retinopathy_model.keras')
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ]

    # --- Phase 1: warmup, backbone frozen ---
    set_backbone_trainable(model, trainable=False)
    model.compile(optimizer=Adam(learning_rate=WARMUP_LR), loss='categorical_crossentropy', metrics=['accuracy'])
    history_warmup = model.fit(
        train_gen,
        validation_data=(X_val, y_val_cat),
        epochs=WARMUP_EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks,
    )

    # --- Phase 2: fine-tune, backbone unfrozen (BatchNorm kept frozen) ---
    set_backbone_trainable(model, trainable=True, freeze_batchnorm=True)
    model.compile(optimizer=Adam(learning_rate=FINE_TUNE_LR), loss='categorical_crossentropy', metrics=['accuracy'])
    history_fine_tune = model.fit(
        train_gen,
        validation_data=(X_val, y_val_cat),
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks,
    )

    model.save(checkpoint_path)

    # TF 2.x history keys are 'accuracy'/'val_accuracy', not the TF 1.x
    # 'acc'/'val_acc'. The original notebook used the TF 1.x names, which
    # raise KeyError against a TF 2.x history object — that's why no
    # training curves were ever produced from it.
    combined_history = {
        'accuracy': history_warmup.history['accuracy'] + history_fine_tune.history['accuracy'],
        'val_accuracy': history_warmup.history['val_accuracy'] + history_fine_tune.history['val_accuracy'],
        'loss': history_warmup.history['loss'] + history_fine_tune.history['loss'],
        'val_loss': history_warmup.history['val_loss'] + history_fine_tune.history['val_loss'],
    }
    with open(os.path.join(args.output, 'history.json'), 'w') as f:
        json.dump(combined_history, f)


if __name__ == '__main__':
    main()
