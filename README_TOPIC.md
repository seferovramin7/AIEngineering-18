# Improving Your Image Models: Transfer Learning and Data Augmentation with Keras

A step-by-step mini-guide for AI Engineering students.

## 1. Motivation

- **Transfer learning**: Start from a network trained on a huge dataset (e.g., ImageNet). Reuse its learned visual features and adapt to your task. This cuts training time and reduces the amount of labeled data you need.
- **Data augmentation**: Randomly transform your images to expose the model to varied conditions and reduce overfitting.

## 2. Core ideas in Keras

- Load a pretrained backbone (e.g., `MobileNetV2`, `ResNet50`, `EfficientNetB0`).
- Freeze the backbone at first to keep generic features.
- Add a small classification head and train it.
- Unfreeze top layers and fine-tune with a low learning rate.
- Insert augmentation layers at the input to diversify data during training.

## 3. Minimal Keras recipe

```python
import tensorflow as tf

img_size = (160, 160)

# Augmentation
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# Pretrained backbone
base = tf.keras.applications.MobileNetV2(
    input_shape=img_size + (3,), include_top=False, weights="imagenet"
)
base.trainable = False  # freeze

inputs = tf.keras.Input(shape=img_size + (3,))
x = augment(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"]) 
```

## 4. Fine-tuning step

- Unfreeze the top N layers, keep the rest frozen.
- Use a smaller learning rate (e.g., 1e-5 to 5e-5).
- Train for a few more epochs.

```python
base.trainable = True
for layer in base.layers[:-20]:  # leave last ~20 layers trainable
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"]) 
```

## 5. Practical augmentation tips

- Start simple: flips, small rotations, zooms.
- Color jitter: brightness/contrast adjustments can help generalization.
- Keep augmentations realistic for your domain.
- Do augmentations in the model (layers) or in the input pipeline.

## 6. When to stop fine-tuning

- If validation accuracy plateaus or drops, you may be overfitting.
- Use early stopping or reduce learning rate.

## 7. Evaluate and iterate

- Look at confusion matrices and misclassified images.
- Try different backbones (EfficientNet often strong for small models).
- Adjust how many layers you unfreeze.

## 8. Resources

- Keras Applications: `https://keras.io/api/applications/`
- Data augmentation: `https://keras.io/api/layers/preprocessing_layers/image_augmentation/`
- Transfer learning tutorial: `https://www.tensorflow.org/tutorials/images/transfer_learning`
