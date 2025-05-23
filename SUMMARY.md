This repository contains the code and data for a project that uses a Faster R-CNN (Region-based Convolutional Neural Network) model to detect red blood cells in microscopic images. The model is built using Keras with a TensorFlow backend and leverages transfer learning, likely with pre-trained ResNet50 weights.

The project includes the following key components:

*   **Dataset:** A collection of approximately 400 microscopic images of red blood cells, along with their corresponding annotations in PASCAL VOC XML format.
*   **Data Preprocessing (`pascal_voc_parser.py`):** A script that reads the images and their annotations, preparing them into a format suitable for training the Faster R-CNN model.
*   **Model Training (`train_frcnn.py`):** A script to train the Faster R-CNN model using the preprocessed dataset.
*   **Model Testing (`test_frcnn.py`):** A script to evaluate the performance of the trained model on new images.
*   **Core Model Library (`keras_frcnn/`):** This directory contains the fundamental building blocks of the Faster R-CNN model, including implementations for network architectures (like ResNet and VGG), ROI pooling, loss functions, and data augmentation.
*   **Pre-trained Weights (`model/`):** This directory likely stores pre-trained weights for the backbone network (e.g., ResNet50) to facilitate transfer learning.

The primary goal of this project is to provide an end-to-end solution for object detection in the context of medical imagery, specifically focusing on red blood cell identification.
