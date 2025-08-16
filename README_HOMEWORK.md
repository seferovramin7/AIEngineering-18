# Homework: Transfer Learning & Data Augmentation

This assignment will guide you through the practical application of transfer learning and data augmentation, two essential techniques for building high-performance image classification models. You will use the concepts and code outlined in the lesson guide to complete the tasks.

---

## Part 1: Transfer Learning and Fine-Tuning

The goal of this part is to leverage a powerful, pre-trained model and adapt it to a new, specific task.

### Your Tasks:

1.  **Choose a Dataset:**
    *   You can use a classic dataset like CIFAR-10 or CIFAR-100 (`tensorflow.keras.datasets`).
    *   Alternatively, you can create your own dataset of images for a custom classification problem (e.g., cats vs. dogs, types of flowers).

2.  **Build the Initial Model (Steps 1 & 2 from the Guide):**
    *   Load a pre-trained model like `MobileNetV2` as your base, without its final classification layer (`include_top=False`).
    *   **Freeze** the weights of the base model (`base.trainable = False`).
    *   Add your own custom classification "head" on top of the base model. This should include a `GlobalAveragePooling2D` layer and a `Dense` output layer with the correct number of neurons for your chosen dataset.
    *   Compile the model with an `Adam` optimizer and an appropriate loss function (e.g., `SparseCategoricalCrossentropy`).

3.  **Initial Training:**
    *   Train the model for a few epochs. At this stage, you are only training the weights of your custom classification head.
    *   Record the validation accuracy.

4.  **Fine-Tuning (Step 3 from the Guide):**
    *   **Unfreeze** the base model (`base.trainable = True`).
    *   Freeze the majority of the layers in the base model, leaving only the top layers (e.g., the last 20) trainable. This allows the model to adapt its more complex feature extractors to your specific dataset.
    *   **Crucially, re-compile the model with a very low learning rate** (e.g., `1e-5`). This prevents the pre-trained weights from being destroyed.

5.  **Continue Training (Step 4 from the Guide):**
    *   Continue training the model for several more epochs.
    *   Monitor the validation accuracy. You should see an improvement over the initial training phase.

---

## Part 2: Data Augmentation

Here, you will explore how augmenting your training data can improve your model's robustness and reduce overfitting.

### Your Tasks:

1.  **Integrate Augmentation Layers:**
    *   Using the model from Part 1, add data augmentation layers at the beginning of your model definition using `tf.keras.Sequential`.
    *   Start with the recommended augmentations: `RandomFlip`, `RandomRotation`, and `RandomZoom`.

2.  **Train and Compare:**
    *   Train the fine-tuned model again from scratch (with the same frozen/unfrozen layers as in Part 1), but this time with the data augmentation layers included.
    *   Compare the final validation accuracy with the result from Part 1. Did data augmentation improve performance?

3.  **Experiment (Optional but Recommended):**
    *   Try adding or changing the augmentation layers. What happens if you increase the rotation or zoom factors?
    *   Visualize a few examples of your augmented training images to ensure the transformations are reasonable.

---

## Submission Requirements

Please submit the following:

1.  A **Python script** or **Jupyter Notebook** containing all the code for both parts of the assignment.
2.  A short **report** in a Markdown file (`REPORT.md`) that includes:
    *   A brief description of the dataset you used.
    *   The validation accuracy you achieved after initial training, after fine-tuning, and after adding data augmentation.
    *   A concluding paragraph explaining what you observed and why fine-tuning and data augmentation are effective.