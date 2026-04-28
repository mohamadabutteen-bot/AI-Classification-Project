# =============================================================================
# Project: Comparative Study of Image Classification Models
# Author: [Your Name/Group Name]
# Date: [Current Date]
#
# Description:
# This script implements and compares three machine learning models for image
# classification: Naive Bayes, Decision Tree, and a Feedforward Neural Network
# (MLP). It handles dataset loading, preprocessing, model training,
# evaluation, and result visualization as per the project requirements.
#
# To Run:
# 1. Ensure you have the required libraries: scikit-learn, numpy, scikit-image,
#    matplotlib, seaborn.
#    (pip install scikit-learn numpy scikit-image matplotlib seaborn)
# 2. Place your dataset in a folder named 'dataset' in the same directory as
#    this script. The 'dataset' folder should contain subfolders, where each
#    subfolder name is a class label (e.g., 'cats', 'dogs', 'birds').
# =============================================================================

import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# --- 1. Configuration Settings ---
DATASET_PATH = "dataset"  # Path to the dataset folder
IMG_SIZE = (64, 64)       # Target image dimensions (height, width)
TEST_SPLIT_SIZE = 0.2     # Proportion of the dataset to include in the test split
RANDOM_STATE = 42         # Seed for reproducibility

# --- 2. Data Loading and Preprocessing ---
print("--- Starting Data Loading and Preprocessing ---")
X = [] # Feature vectors (flattened images)
y = [] # Labels

# Check if dataset path exists
if not os.path.isdir(DATASET_PATH):
    print(f"Error: Dataset path '{DATASET_PATH}' not found.")
    print("Please create the 'dataset' folder with class subdirectories.")
    exit()

# Iterate through each class folder in the dataset directory
for label_name in os.listdir(DATASET_PATH):
    label_folder = os.path.join(DATASET_PATH, label_name)
    if not os.path.isdir(label_folder):
        continue

    # Iterate through each image file in the class folder
    for file_name in os.listdir(label_folder):
        file_path = os.path.join(label_folder, file_name)
        try:
            # Read the image
            image = imread(file_path)

            # If the image has 3 or 4 channels (RGB or RGBA), convert to grayscale
            if image.ndim == 3:
                image = rgb2gray(image) # More accurate than mean

            # Resize the image to the specified dimensions
            image_resized = resize(image, IMG_SIZE, anti_aliasing=True)

            # Flatten the 2D image into a 1D feature vector and append to list
            X.append(image_resized.flatten())
            y.append(label_name)
        except Exception as e:
            print(f"Skipping file '{file_path}' due to error: {e}")

# Convert lists to NumPy arrays for efficient computation
X = np.array(X)
y = np.array(y)

if len(X) == 0:
    print("Error: No images were loaded. Check the dataset path and image files.")
    exit()

print(f"\nSuccessfully loaded {len(X)} images from {len(set(y))} classes: {list(set(y))}")
print(f"Shape of feature matrix X: {X.shape}") # (num_images, height*width)
print(f"Shape of label vector y: {y.shape}")   # (num_images,)

# Encode string labels (e.g., 'cats', 'dogs') into integers (0, 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_

print("\nLabel Encoding Mapping:")
for i, class_name in enumerate(class_names):
    print(f"  '{class_name}' => {i}")

# Split the dataset into training and testing sets
# 'stratify=y_encoded' ensures that the class distribution is the same in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
)

print(f"\nDataset split:")
print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples: {len(X_test)}")

# --- 3. Helper Function for Evaluation ---
def plot_confusion_matrix(y_true, y_pred, model_name, file_name):
    """Generates and saves a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()
    print(f"Saved confusion matrix to {file_name}")

# Dictionary to store all model results for final comparison
results = {}

# --- 4. Model Implementation and Evaluation ---

# === 4.1 Naive Bayes Classifier (on raw pixels) ===
print("\n" + "="*30)
print("  Model 1: Naive Bayes (Pixel-based)")
print("="*30)

nb_pixel = GaussianNB()
nb_pixel.fit(X_train, y_train)
y_pred_nb_pixel = nb_pixel.predict(X_test)

# Evaluate the model
acc_nb_pixel = accuracy_score(y_test, y_pred_nb_pixel)
report_nb_pixel = classification_report(y_test, y_pred_nb_pixel, target_names=class_names)
cv_nb_pixel = cross_val_score(nb_pixel, X, y_encoded, cv=5, scoring='accuracy')

print("Classification Report:")
print(report_nb_pixel)
print(f"Accuracy: {acc_nb_pixel:.4f}")
print(f"5-Fold CV Accuracy: {cv_nb_pixel.mean():.4f} (+/- {cv_nb_pixel.std() * 2:.4f})")

plot_confusion_matrix(y_test, y_pred_nb_pixel, "Naive Bayes (Pixels)", "cm_nb_pixel.png")
results['NB (Pixels)'] = {'Accuracy': acc_nb_pixel, 'CV Accuracy': cv_nb_pixel.mean()}


# === 4.2 Naive Bayes Classifier (on statistical features) ===
print("\n" + "="*30)
print("  Model 1b: Naive Bayes (Statistical Features)")
print("="*30)

def extract_stats(images):
    """Extracts mean and standard deviation for each image."""
    return np.array([[np.mean(img), np.std(img)] for img in images])

# Transform data to statistical features
X_train_stats = extract_stats(X_train.reshape(-1, *IMG_SIZE))
X_test_stats = extract_stats(X_test.reshape(-1, *IMG_SIZE))
X_stats_full = extract_stats(X.reshape(-1, *IMG_SIZE))

nb_stats = GaussianNB()
nb_stats.fit(X_train_stats, y_train)
y_pred_nb_stats = nb_stats.predict(X_test_stats)

# Evaluate the model
acc_nb_stats = accuracy_score(y_test, y_pred_nb_stats)
report_nb_stats = classification_report(y_test, y_pred_nb_stats, target_names=class_names)
cv_nb_stats = cross_val_score(nb_stats, X_stats_full, y_encoded, cv=5, scoring='accuracy')

print("Classification Report:")
print(report_nb_stats)
print(f"Accuracy: {acc_nb_stats:.4f}")
print(f"5-Fold CV Accuracy: {cv_nb_stats.mean():.4f} (+/- {cv_nb_stats.std() * 2:.4f})")

plot_confusion_matrix(y_test, y_pred_nb_stats, "Naive Bayes (Stats)", "cm_nb_stats.png")
results['NB (Stats)'] = {'Accuracy': acc_nb_stats, 'CV Accuracy': cv_nb_stats.mean()}


# === 4.3 Decision Tree Classifier ===
print("\n" + "="*30)
print("  Model 2: Decision Tree")
print("="*30)

# Initialize and train the Decision Tree
# `max_depth` is a hyperparameter to prevent overfitting.
dt = DecisionTreeClassifier(max_depth=20, random_state=RANDOM_STATE)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Evaluate the model
acc_dt = accuracy_score(y_test, y_pred_dt)
report_dt = classification_report(y_test, y_pred_dt, target_names=class_names)
cv_dt = cross_val_score(dt, X, y_encoded, cv=5, scoring='accuracy')

print("Classification Report:")
print(report_dt)
print(f"Accuracy: {acc_dt:.4f}")
print(f"5-Fold CV Accuracy: {cv_dt.mean():.4f} (+/- {cv_dt.std() * 2:.4f})")

plot_confusion_matrix(y_test, y_pred_dt, "Decision Tree", "cm_decision_tree.png")
results['Decision Tree'] = {'Accuracy': acc_dt, 'CV Accuracy': cv_dt.mean()}

# Visualize the top levels of the decision tree
print("\nVisualizing Decision Tree structure (first 3 levels)...")
plt.figure(figsize=(25, 15))
plot_tree(
    dt,
    max_depth=3,
    feature_names=[f"pixel_{i}" for i in range(X.shape[1])],
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=10,
    proportion=True,
    impurity=False
)
plt.title("Decision Tree Structure (Top 3 Levels)")
plt.savefig('decision_tree_structure.png', dpi=300)
plt.close()
print("Saved Decision Tree visualization to decision_tree_structure.png")


# === 4.4 Feedforward Neural Network (MLPClassifier) ===
print("\n" + "="*30)
print("  Model 3: MLP Classifier (Optional)")
print("="*30)

# Neural networks are sensitive to feature scaling, so we scale the data
# We fit the scaler on the training data and use it to transform both train and test data
print("Scaling data for MLP...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled_full = scaler.fit_transform(X) # For cross-validation

# Initialize and train the MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50), # Two hidden layers
    max_iter=300,
    random_state=RANDOM_STATE,
    verbose=False # Set to True to see training progress
)
print("Training MLP... (this may take a moment)")
mlp.fit(X_train_scaled, y_train)
y_pred_mlp = mlp.predict(X_test_scaled)

# Evaluate the model
acc_mlp = accuracy_score(y_test, y_pred_mlp)
report_mlp = classification_report(y_test, y_pred_mlp, target_names=class_names)
# NOTE: Cross-validation for MLP can be time-consuming
cv_mlp = cross_val_score(mlp, X_scaled_full, y_encoded, cv=5, scoring='accuracy')

print("\nClassification Report:")
print(report_mlp)
print(f"Accuracy: {acc_mlp:.4f}")
print(f"5-Fold CV Accuracy: {cv_mlp.mean():.4f} (+/- {cv_mlp.std() * 2:.4f})")

plot_confusion_matrix(y_test, y_pred_mlp, "MLP Classifier", "cm_mlp.png")
results['MLP'] = {'Accuracy': acc_mlp, 'CV Accuracy': cv_mlp.mean()}

# Plot MLP learning curve
plt.figure(figsize=(10, 6))
plt.plot(mlp.loss_curve_)
plt.title('MLP Learning Curve (Loss over Iterations)')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('mlp_learning_curve.png')
plt.close()
print("Saved MLP learning curve to mlp_learning_curve.png")

# --- 5. Comparative Analysis ---
print("\n" + "="*40)
print("      Final Model Comparison")
print("="*40)

model_names = list(results.keys())
accuracies = [res['Accuracy'] for res in results.values()]
cv_accuracies = [res['CV Accuracy'] for res in results.values()]

# Print summary table
print(f"{'Model':<20} | {'Test Accuracy':<15} | {'5-Fold CV Accuracy':<20}")
print("-"*60)
for i, model in enumerate(model_names):
    print(f"{model:<20} | {accuracies[i]:<15.4f} | {cv_accuracies[i]:<20.4f}")

# Plotting the comparison
x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width/2, accuracies, width, label='Test Set Accuracy', color='skyblue')
rects2 = ax.bar(x + width/2, cv_accuracies, width, label='5-Fold CV Accuracy', color='salmon')

ax.set_ylabel('Accuracy')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15)
ax.set_ylim(0, 1.1)
ax.legend()

# Add labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig('model_comparison.png')
plt.close()
print("\nSaved final model comparison chart to model_comparison.png")

print("\n--- Script finished successfully! All visualizations have been saved. ---")