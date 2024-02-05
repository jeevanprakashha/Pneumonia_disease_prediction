# Pneumonia Detection using Deep Learning

## Overview
This project aims to develop a deep learning-based system for detecting pneumonia from chest X-ray images. Leveraging advanced neural network models, the project seeks to improve the accuracy and efficiency of pneumonia diagnoses.

## Authors
- Jeevan Prakash HA (20BRS1259)
- Vignesh V (20BRS1148)

## Guided By
Dr. Shridevi S, Professor

## Objective
**Goal:** To create a highly accurate diagnostic tool using Convolutional Neural Networks (CNNs) for the detection of pneumonia in chest X-rays, enhancing the quality and speed of medical diagnoses.

## Methodology
### Data Collection
- Utilization of the ChestXray2017 dataset, containing X-ray images labeled as 'normal' and 'pneumonia'.

### Data Preprocessing
- Images are preprocessed and augmented using techniques like resizing, normalization, and random flips/rotations.

### Model Architecture
- Use of the VGG16 model, a pre-trained CNN, as the base model.
- The network is further customized for the specific task of pneumonia detection.

### Training Approach
- The model is trained using the prepared dataset, with class weights calculated to address imbalances between 'normal' and 'pneumonia' classes.

### Evaluation Metrics
- Accuracy, precision, recall, and loss metrics are used to evaluate the model's performance.

## Result
### Performance
- The model demonstrates high accuracy and precision in identifying pneumonia cases from X-rays.

### Visualization
- Training metrics are visualized using plots for precision, recall, accuracy, and loss, illustrating the model's effectiveness over epochs.

### Sample Prediction
- Showcasing a test image with its actual label and the predicted label, emphasizing the model's predictive capability.


## Analysis
- Insight: The system effectively uses deep learning to analyze medical images, showcasing the potential of AI in enhancing diagnostic processes.
- Grad-CAM Analysis: Implementation of Grad-CAM for visual explanations of model decisions, highlighting regions in X-ray images significant for pneumonia detection.

## Conclusion
- Implications: The project successfully demonstrates the use of deep learning in medical image analysis, providing a robust tool for pneumonia detection in chest X-rays. It highlights the potential for further applications of AI in healthcare diagnostics.
- Future Work: Opportunities for enhancing the model, including the integration of larger and more diverse datasets, and the exploration of different neural network architectures for improved performance.
