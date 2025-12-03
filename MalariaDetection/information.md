# Malaria Detection using Machine Learning with Dimension Reduction

## Project Overview
This notebook implements a comprehensive approach to malaria detection from blood cell images using various machine learning techniques with a focus on dimension reduction methods.

### Dataset Challenges and Preprocessing Strategy

**Mixed Image Formats Issue:**
- The dataset contains images in different formats (.tiff and .png)
- For consistent processing, we will standardize all images to PNG format
- This ensures uniform handling across the entire pipeline

**Class Imbalance Problem:**
- Training set: 4,000 negative vs 800 positive samples (~5:1 ratio)
- Validation set: 1,531 negative vs 1,035 positive samples (~1.5:1 ratio)
- This significant imbalance requires special handling techniques


### Techniques to Handle Class Imbalance:

1. **Data-Level Approaches:**
   - **Data Augmentation:** Apply transformations specifically to minority class

2. **Algorithm-Level Approaches:**
   - **Class Weights:** Assign higher weights to minority class during training

3. **Evaluation Strategies:**
   - **Stratified Cross-Validation:** Maintain class distribution in folds
   - **Balanced Metrics:** Use F1-score, precision, recall, and AUC-ROC instead of just accuracy
   - **Confusion Matrix Analysis:** Detailed analysis of true/false positives and negatives

4. **Ensemble Methods:**
   - **Balanced Random Forest:** Built-in handling of class imbalance
   - **EasyEnsemble:** Combine multiple balanced classifiers
   - **BalanceCascade:** Sequential ensemble with balanced sampling

### Implementation Plan:
We will implement and compare multiple approaches to find the most effective combination for this specific dataset and problem.


---

## Data Exploration and Preprocessing

In this section, we will:
1. Load and examine the dataset structure
2. Analyze image properties and distributions
3. Visualize sample images from both classes
4. Implement preprocessing pipeline for format standardization
5. Apply class imbalance handling techniques
6. Prepare data for dimension reduction experiments


---

# Examining the Dataset

When examining the malaria dataset, we noticed that the images come in two different formats: TIFF and PNG. If we train a model directly on this mixed-format dataset, the model may unintentionally learn to distinguish file format artifacts instead of actual malaria parasites. To prevent this and ensure the model learns meaningful medical features, we need to convert all images into a consistent format.

To understand how many images exist in each format, we can run the following cell to count them:

---

# Dataset Format: TIFF and PNG

Based on the above analysis, we see that the dataset contains a mix of TIFF and PNG images, with all Negative training samples in TIFF format (4000 images) and most Positive samples in PNG format (800 training + all validation). Because of this imbalance, the model would learn to distinguish file formats instead of learning malaria-specific visual features. To avoid this issue and ensure consistent preprocessing, we will convert the entire dataset into PNG format.

We choose PNG because more than 45% of the dataset is already in PNG, and PNG is a widely used, lossless image format commonly used in biomedical imaging. In fact, standard malaria image benchmarks; such as the NIH Malaria Dataset (27,558 cell images) also distribute images in PNG/JPEG format, not TIFF. This means PNG is already accepted as a standard for malaria cell classification, and converting everything to PNG ensures compatibility with commonly used deep learning pipelines and pretrained models (e.g., ResNet, VGG), which expect standard RGB inputs rather than high–bit depth TIFF files.

One other files called Thumbs.db, which is a thumbnail cache file for Windows was found in the validation positive directory. We should remove it.

---

# Visualizing some positive and negative samples

---

# Verifying Image Dimensions

Consistent image dimensions are essential for proper processing because each pixel position must correspond across all samples. When images differ in width or height, the model cannot align pixel locations or form uniform tensors, causing shape mismatches and disrupting feature extraction. Standardizing the dimensions ensures that every image maps to the same spatial structure, allowing pixel-wise operations, batching, and neural network computations to function correctly.

---

# Why 64×64 Is the Ideal Resolution for This Dataset

A target input resolution of 64×64 is selected to balance feature preservation and computational efficiency. The dataset’s dimensions cluster around ~55 px, ranging from 30×32 to 67×86. Using a standard size like 224×224 would require extreme upscaling—up to 700% for the smallest images—introducing interpolation artifacts that can obscure parasite features. In contrast, 64×64 is the closest power-of-two resolution that fits the dataset’s natural scale: larger images downsample cleanly with minimal loss, while smaller images undergo only mild, non-destructive upscaling. This maintains critical morphological details across the full distribution.

---

# Standardizing Malaria Images to 64 × 64 Pixels

### Why Cropping or Resizing Alone Doesn’t Work

For the malaria dataset, converting all images to 64×64 cannot be achieved through cropping or direct resizing. Cropping may remove the parasite at the cell’s edge, and resizing distorts the aspect ratio, altering critical cell morphology.

### Downsampling and Padding: Preserving Biological Structure

To preserve the biological structure, we first downsample the images. Downsampling reduces the image resolution while maintaining the overall spatial relationships and morphology, ensuring that the parasite remains intact. After downsampling, we pad the images to reach the target 64×64 size without stretching or compressing the cells. This sequence of downsampling first, then padding, preserves both the parasite features and the surrounding cell context.

### What Pixel Value Is Best for Padding?
Selecting the appropriate padding value is crucial to prevent introducing artifacts. We will experiment with both zero-padding and padding using the image’s mean value.

---

# Finding an Artifact-Free Padding Value

Initially, we tried standardizing image sizes using zero-padding and mean-padding. However, both approaches introduced unwanted artifacts:

- Zero-padding (black) created sharp, high-contrast borders that were unnatural for microscope images.

- Mean-padding produced visible gray bands, as the dark parasite pixels pulled the average downward, creating artificial edges.

These artifacts are problematic because the model could mistakenly learn to rely on them instead of focusing on true biological features.

To address this, we aim to identify a better universal padding value that blends seamlessly with the microscope background, preserving cell morphology while avoiding any misleading cues.

### Proposed solution
A more reliable approach analyzes 5×5 pixel blocks from the four corners of all images in the training dataset. Since the corners consistently contain empty microscope background, they provide an accurate estimate of the true slide color. Using this value for padding preserves the natural background while avoiding any misleading cues, ensuring the model focuses on genuine biological features.

---

# Universal Background Color

From this analysis, a “universal background color” (approximately 253, 252, 253) is obtained. Using this value for padding makes the padded regions visually seamless, ensuring the model concentrates only on the parasitic infection rather than on unnatural padding borders.

---

# Artifact-Free Padding with Universal Background Color

The results show that using the universal background color (253, 252, 253) for padding introduces no visible artifacts, unlike zero-padding or mean-padding, which create high-contrast edges or gray bands. Given its seamless integration with the microscope background, we will use this universal background color as the standard padding value for all images.

---

## Fix for the Class imbalance (Data Augmentation)

Given the significant class imbalance where negative samples (4,000) vastly outnumber positive samples (800), applying data augmentation exclusively to the positive class is the optimal strategy to prevent the model from blindly predicting "Negative" to achieve high accuracy. To establish perfect class parity, we calculated a required expansion factor of 5 (4000 / 800 = 5), which dictates that for every single original positive image, we must generate exactly four unique synthetic copies to reach a balanced total (800 original + (800 x 4 copies) = 4,000). We specifically selected Rotate 90°, Rotate 180°, Vertical Flip, and Horizontal Flip because these rigid geometric transformations are mathematically lossless and biologically "safe"; unlike random rotations or blurring which introduce interpolation artifacts, these operations preserve the exact pixel sharpness and diagnostic color integrity of the parasites while leveraging the fact that blood cells are rotation-invariant, ensuring the model learns to recognize the infection regardless of its orientation without the risk of degrading the image quality.

---

# The Limitation of Traditional ML for Image Data
While algorithms like Logistic Regression, Support Vector Machines (SVM), and Random Forests can serve as a useful baseline (a minimum score to beat), they are fundamentally flawed for complex image classification tasks like malaria detection. The primary reason is the destructive process of Flattening.

To feed a 64×64 pixel color image into these models, we must transform the 3D matrix (64 height × 64 width × 3 channels) into a massive 1D list of 12,288 individual numbers (64 x 64 x 3 = 12,288).

---

## Data Loading and Normalization

The dataset is already organized into `train/` and `val/` folders. In this section, we will:
1. **Load images** from the existing train and validation directories
2. **Normalize pixel values** to [0, 1] range (divide by 255) - this is crucial for neural network training because neural networks initialize weights with small random numbers, feeding in large integers (0-255) creates massive mathematical variance that destabilizes the gradients, whereas a small 0-1 range creates a smooth error landscape that allows the optimizer to converge significantly faster.
3. **Create labels** (0 for negative/uninfected, 1 for positive/infected)
4. **Combine data** into NumPy arrays ready for modeling

**Note:** We use the existing train/val split provided in the dataset - no need to reorganize folders!

---

### Step 1: Create Image Loading and Normalization Function

We'll create a reusable function that:
- Loads images from a directory
- Converts them to NumPy arrays
- Normalizes pixel values from [0, 255] to [0, 1] by dividing by 255
- Returns both the image arrays and corresponding labels


---

### Step 2: Load Training and Validation Data

Now we'll load all images from both training and validation sets, creating separate arrays for each split.


---

### Step 3: Verify Normalization

We need to confirm that all pixel values are in the [0, 1] range. This is critical for neural network training.


---

### Step 4: Display Data Shapes and Class Distributions

Understanding the data structure is essential before modeling. We'll examine:
- **Data shapes**: Dimensions of our arrays
- **Class distributions**: Balance/imbalance in each split
- **Memory usage**: How much RAM our data requires


---

## Convolutional Neural Network (CNN) for Malaria Detection

We'll build a CNN from scratch to classify malaria-infected vs uninfected blood cells. This approach is superior to traditional ML because:
- **Preserves spatial information**: Convolutional layers learn spatial patterns (edges, shapes, textures)
- **Hierarchical feature learning**: Automatically learns low-level (edges) to high-level (parasite shapes) features
- **Translation invariance**: Can detect parasites regardless of their position in the image

### Strategy:
1. **Data Augmentation**: Apply rotations and flips to positive samples during training to balance classes (4000:4000)
2. **CNN Architecture**: Build a simple but effective CNN with convolutional, pooling, and dense layers
3. **Training with Validation**: Monitor validation metrics during training


---

### Step 1: Data Augmentation for Positive Class

We'll augment the positive training samples using the same transformations we demonstrated earlier (rotations and flips) to balance the classes from 5:1 to approximately 1:1.


---

### Step 2: Define CNN Architecture

We'll build a CNN with:
- **Convolutional layers**: Extract spatial features (edges, textures, patterns)
- **Max pooling layers**: Reduce spatial dimensions and computational cost
- **Dropout layers**: Prevent overfitting
- **Dense layers**: Final classification


---

### Step 3: Compile Model and Set Up Training

We'll use:
- **Binary crossentropy loss**: Standard for binary classification
- **Adam optimizer**: Adaptive learning rate
- **Callbacks**: Early stopping and learning rate reduction


---

### Step 4: Train the Model

We'll train the model and monitor validation metrics in real-time. The training will show:
- Training and validation loss
- Training and validation accuracy
- Training and validation precision
- Training and validation recall


---

### Step 5: Visualize Training History

Plot training and validation metrics to analyze model performance.


---

Model is overfitting. 
To reduce the severe overfitting (where Training Accuracy is 100% but Validation Loss is exploding), we need to restrict the model's "freedom" to memorize. Currently, the model is too complex for simple 64x64 images.

1. Drastically Reduce Dense Layer Size (The Main Culprit)
You are using 512 and 256 neurons in your dense layers.

Problem: This gives the model millions of parameters to memorize specific pixels of the training images.

Fix: Drop this to 64 or 128. Malaria parasites are simple shapes (blobs/rings); you don't need a massive "brain" to recognize them.

2. Add Batch Normalization
You are missing Batch Normalization.

Problem: Without it, color values shift around during training, making the model chase moving targets.

Fix: Add layers.BatchNormalization() after every Convolution and before the Activation. This stabilizes the model and has a regularizing effect.

3. Add L2 Regularization (Weight Decay)
Problem: Your model is "Arrogant" (predicting 0.9999 probability).

Fix: Add L2 Regularization. This penalizes the model for having huge weights, forcing it to be "humble" and smoother in its predictions.

---

### Final Step: Evaluate on Validation Set with Detailed Metrics

Calculate comprehensive evaluation metrics including confusion matrix and F1-score.


---

## Back to Setp 2. We will again follow the step 2 to step 5.
Define a less complicated mode to reduce the overfitting.  

---

### Final Step: Evaluate on Validation Set with Detailed Metrics

Calculate comprehensive evaluation metrics including confusion matrix and F1-score.


---

# Batch Normalization: Non-Trainable Parameters Explained

The 576 non-trainable parameters in the model come from the Batch Normalization layers, which track statistics rather than being updated through backpropagation. Each Batch Normalization layer has **four parameters per channel**:

- **Gamma (γ)**: Trainable. Scales the normalized data.  
- **Beta (β)**: Trainable. Shifts the normalized data.  
- **Moving Mean (μ)**: Non-trainable. Tracks the average of the activations across batches.  
- **Moving Variance (σ²)**: Non-trainable. Tracks the spread (variance) of the activations across batches.  

These non-trainable statistics allow the model to normalize inputs consistently during inference.  

### Calculation of 576 Non-Trainable Parameters

- Conv Block 1: 32 filters → 32 channels  
- Conv Block 2: 64 filters → 64 channels  
- Conv Block 3: 128 filters → 128 channels  
- Dense Block: 64 neurons → 64 channels  

**Total channels:** 32 + 64 + 128 + 64 = 288  

**Non-trainable parameters:** 288 channels × 2 (moving mean + moving variance) = **576**  

This confirms that the Batch Normalization layers are correctly tracking data statistics to stabilize and accelerate training.


---

Back to Step 2: ONly change in i am making is incresing the batch size to 64 from 32. I believe by doing so the jagged up and down lines in those validation curve will reduce.

---

# Choosing the Right Batch Size: 32 vs. 64

Initially, it was thought that increasing the batch size from **32 to 64** would smooth out the training curves, reducing jagged spikes in loss and accuracy. However, the experiment revealed a different story:

- **Batch Size 64 Observations:**
  - Training became more **violent and unstable** in the early epochs.
  - **Epoch 9:** Loss exploded to nearly 10.0 and accuracy crashed to ~40%, a massive gradient explosion.
  - **Epoch 10+:** The model quickly recovered due to Batch Normalization, snapping back to ~90% accuracy.
  - Larger batches take **bigger steps** in the gradient space, which can cause extreme updates when "difficult" or outlier images are grouped together.
  
- **Batch Size 32 Observations:**
  - Training was **more stable**, with smaller, controlled fluctuations in loss and accuracy.
  - Smaller batches take **smaller, cautious steps**, allowing the optimizer to navigate outliers more gracefully.
  - Overall accuracy remained high (~96%) without catastrophic spikes.

**Conclusion:**  
Although Batch Size 64 eventually reached a good final accuracy, the jagged, unstable training makes the learning curve look risky. **Batch Size 32 is preferred** for both stability and clarity in reporting, providing a consistent, smooth learning trajectory.  

**Decision:** Stick with **Batch Size 32** for training and reporting purposes.


---

### Final Step: Evaluate on Validation Set with Detailed Metrics

Calculate comprehensive evaluation metrics including confusion matrix and F1-score.


