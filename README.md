
# **Face Mask Detection with Convolutional Neural Networks (CNN)**

This project aims to create a deep learning model to detect whether a person is wearing a face mask or not using images. The model is based on a Convolutional Neural Network (CNN) architecture, which is well-suited for image classification tasks. The dataset used for training and testing is manually prepared and includes images of individuals with and without face masks.

## **Dataset**
https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
The dataset consists of images organized into two main categories:

* **with\_mask**: Images of individuals wearing face masks.
* **without\_mask**: Images of individuals not wearing face masks.

The dataset was compressed into a zip file named `data.zip` and includes both training and testing images. The images were extracted from the zip file into appropriate subdirectories for further processing.

### **Folder Structure**:

```
/data
    /with_mask      <-- Contains images of people wearing masks
    /without_mask   <-- Contains images of people not wearing masks
```

## **Preprocessing**

The data is preprocessed to prepare it for training the CNN model. The preprocessing steps are as follows:

1. **Image Resizing**: All images are resized to **224x224 pixels**, which is the input size expected by the CNN model.
2. **Rescaling**: Pixel values of the images are rescaled to the range `[0, 1]` by dividing by `255` to normalize the data.
3. **Data Augmentation**: For the training data, several augmentations are applied, such as:

   * Random rotations.
   * Horizontal and vertical shifts.
   * Zooming.
   * Horizontal flipping.

   These augmentations help improve the model’s ability to generalize by introducing more variations in the training data.

## **Data Splitting**

The dataset is **not pre-split** into training and testing sets, so it was automatically split using `validation_split=0.2` from the `ImageDataGenerator` class. This ensures that 80% of the data is used for training and 20% for validation, without the need for manual splitting.

### **Data Splitting Details**:

* **Training Data**: 80% of the images from `with_mask` and `without_mask` directories are used for training.
* **Validation Data**: 20% of the images from both classes are used for validation.

## **Model Architecture**

The model is built using a **Convolutional Neural Network (CNN)**, which is ideal for image classification tasks. The architecture consists of the following layers:

1. **Convolutional Layers**: Three convolutional layers are used, with increasing filter sizes (32, 64, 128), to capture features from the images.
2. **MaxPooling Layers**: MaxPooling layers are applied after each convolutional layer to reduce spatial dimensions and retain the most important features.
3. **Fully Connected Layer**: A dense layer with 512 units is used to interpret the extracted features from the convolutional layers.
4. **Dropout**: A dropout rate of 0.5 is applied to prevent overfitting during training.
5. **Output Layer**: The output layer has a single neuron with a sigmoid activation function, which is used for binary classification (mask vs. no mask).

### **Model Compilation**:

* **Optimizer**: Adam optimizer was used for efficient training.
* **Loss Function**: Binary cross-entropy was chosen as the loss function, as this is a binary classification problem.
* **Metrics**: Accuracy is used as the evaluation metric.

## **Training**

The model was trained for **10 epochs** using the processed and augmented training data. The batch size was set to 32, meaning the model processes 32 images at a time during training.

### **Training Results**:

The model showed steady improvement in both training and validation accuracy across the epochs. By the end of 10 epochs, the following results were observed:

* **Final Training Accuracy**: 87.5%
* **Final Validation Accuracy**: 89.58%
* **Training Loss**: Reduced from 0.4988 to 0.2963
* **Validation Loss**: Reduced from 0.8924 to 0.2416

### **Validation Results**:

After training, the model was evaluated on the validation set (20% of the data). The results were as follows:

* **Test Accuracy**: 89.58%
* **Test Loss**: 0.2526

This indicates that the model is performing well and is able to generalize effectively on unseen data.

## **Model Performance Evaluation**

* **Validation Loss**: Initially high and gradually decreased, showing that the model was improving over time.
* **Test Accuracy**: The final test accuracy of **89.58%** indicates that the model is good at detecting whether a person is wearing a mask or not.

### **Confusion Matrix** 
A confusion matrix would give more insight into the performance by showing true positives, true negatives, false positives, and false negatives.

## **Conclusion**

This model was able to successfully detect whether a person is wearing a face mask with a high degree of accuracy (89.58% on the test set). It performs well on both training and unseen data, which indicates that the model is not overfitting and can generalize to new images.
Feel free to use this.
