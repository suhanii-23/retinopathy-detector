"""Model architecture and backbone freeze/unfreeze logic."""

from tensorflow.keras import layers, models
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import BatchNormalization

NUM_CLASSES = 5
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']


def build_model(input_shape=(299, 299, 3), n_out=NUM_CLASSES, dropout=0.5):
    """Xception backbone + custom classification head.

    Xception over other ImageNet backbones:
    - Native 299x299 input. Retinopathy is graded from microaneurysms
      only a few pixels wide, so downscaling to 224 (as most backbones
      expect) loses the signal the model needs.
    - Depthwise separable convolutions give ImageNet-level capacity at
      far fewer parameters than a comparable non-separable network,
      which matters when fine-tuning on only ~3.1k training images.
    - It was the strongest single backbone in published APTOS 2019
      solutions.

    GlobalAveragePooling2D over Flatten: pooling adds zero parameters,
    where flattening Xception's final feature map into a dense layer
    would add tens of millions of parameters that would overfit almost
    immediately on a dataset this small.
    """
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout),
        layers.Dense(2048, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(n_out, activation='softmax'),
    ])

    return model


def set_backbone_trainable(model, trainable, freeze_batchnorm=True):
    """Toggle the Xception backbone between frozen (warmup) and
    unfrozen (fine-tuning), keeping BatchNorm layers in inference mode.

    BatchNorm layers track running mean/variance statistics. If they
    are left trainable while the backbone is unfrozen, those running
    statistics get updated from batches of just 16 fundus images whose
    distribution (colour, illumination, scale) is nothing like
    ImageNet's, which destabilises fine-tuning. Keeping BN layers in
    inference mode while the surrounding conv weights stay trainable is
    the standard mitigation for this. The original notebook did not do
    this — it's a real fix, not boilerplate.
    """
    base_model = model.layers[0]
    base_model.trainable = trainable

    if trainable and freeze_batchnorm:
        for layer in base_model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = False

    # The classification head is always trainable, in both phases.
    for layer in model.layers[1:]:
        layer.trainable = True

    return model
