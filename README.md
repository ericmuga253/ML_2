
# Brain MRI Images for Brain Tumor Detection
![image](https://github.com/user-attachments/assets/d5390add-ca03-48d6-b4fe-5cd45d3f114b)


This project utilizes a convolutional neural network (CNN) model, specifically a modified VGG16 architecture, to classify brain MRI images as either containing a tumor or not. The dataset used in this project, *Brain MRI Images for Brain Tumor Detection*, is available on Kaggle. This README will guide you through the steps needed to set up, train, and evaluate the model.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Model Architecture](#model-architecture)
6. [Training and Evaluation](#training-and-evaluation)
7. [Results](#results)
8. [License](#license)

## Project Overview

The primary objective of this project is to classify brain MRI images as either tumor-positive or tumor-negative using deep learning techniques. The VGG16 model, pretrained on the ImageNet dataset, is fine-tuned for this classification task. This model architecture is efficient for image classification, especially when transfer learning is applied to domain-specific data like medical imaging.

## Dataset

The dataset can be downloaded from Kaggle:
- [Brain MRI Images for Brain Tumor Detection on Kaggle](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)

The dataset contains images organized into folders (`yes` for images with tumors, `no` for images without tumors). The images are grayscale, which we resize to `(224, 224)` for compatibility with the VGG16 input layer.

## Requirements

- Python 3.x
- Jupyter Notebook or Google Colab for running the code
- Libraries:
  - `tensorflow`
  - `keras`
  - `opencv-python`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `sklearn`
  - `tqdm`

## Installation

1. Clone or download this repository.
2. Install the required libraries:
   ```bash
   pip install tensorflow keras opencv-python numpy pandas matplotlib sklearn tqdm
   ```

3. Upload `kaggle.json` to authenticate the Kaggle API, download the dataset, and place it in your working directory.

4. Download the dataset using the following code:
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```

## Model Architecture

This project uses transfer learning with a VGG16 model. Key steps include:

1. **VGG16 Base Model**: Load the VGG16 model pre-trained on ImageNet, excluding the top layer (`include_top=False`).
2. **Custom Fully Connected Head**: Add custom dense layers on top of VGG16 using `GlobalAveragePooling2D` and fully connected layers for binary classification.
3. **Compile**: The model is compiled using the Adam optimizer and `categorical_crossentropy` loss for multi-class classification (two classes: tumor and no-tumor).

### Code Snippet for Model Architecture

```python
from keras.applications import vgg16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# Load VGG16 base model
vgg = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in vgg.layers:
    layer.trainable = False  # Freeze layers

# Custom head for classification
def lw(bottom_model, num_classes):
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(512, activation='relu')(top_model)
    top_model = Dense(num_classes, activation='softmax')(top_model)
    return top_model

num_classes = 2
FC_Head = lw(vgg, num_classes)
model = Model(inputs=vgg.input, outputs=FC_Head)
```

## Training and Evaluation

1. **Training**: The model is trained for 5 epochs with a validation split to monitor overfitting.
2. **Evaluation**: Accuracy and loss curves are plotted for training and validation datasets to analyze performance.

### Training Code Example

```python
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=1)
```

## Results

Results will vary depending on dataset split and parameters, but generally, the model should achieve significant accuracy in distinguishing between tumor and non-tumor images.

### Example Plots

After training, accuracy and loss can be visualized:

```python
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.show()
```

## License

This project follows the licensing terms specified by Kaggle for the dataset and the terms associated with any pre-trained models used (like VGG16 from Keras).

---

This README provides a comprehensive overview to help you set up, train, and evaluate the model. Make sure to adjust paths and parameters as necessary. Good luck with your classification project!
