# Image-Classification
Neural network model tailored for image classification on the CIFAR-100 dataset

This repository features an advanced neural network model tailored for image classification on the CIFAR-100 dataset, utilizing a customized encoder architecture integrated into a comprehensive model structure. The CIFAR-100 dataset, known for its complexity with 100 different classes, poses a significant challenge for deep learning models, requiring sophisticated architectures for effective learning and classification.

The core of this project is the Net class, which incorporates a pre-trained encoder for feature extraction, followed by a series of fully connected layers for classification. The encoder, built from convolutional, ReLU activation, and max-pooling layers, captures intricate patterns and characteristics from input images. This model exemplifies a blend of transfer learning and neural network fine-tuning, designed to harness the power of pre-trained models while adapting to the specific nuances of the CIFAR-100 dataset.

Training this model involves several key steps, including data augmentation (e.g., random horizontal flips and crops) and normalization, to enhance the model's generalization capability. The training process is optimized with a Stochastic Gradient Descent (SGD) optimizer, accompanied by a learning rate scheduler to adjust the learning rate based on training progress, aiming for efficient convergence to the optimal solution.
