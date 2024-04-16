# **Detection of AI-Generated Sneakers using PyTorch**

## Context
This project specifically focuses on images of shoes from popular brands such as Nike, Adidas, and Converse. The dataset used in this study includes a mix of real images obtained from Google Images and AI-generated images produced by MidJourney, offering a unique challenge in the field of computer vision and machine learning.

This dataset has been carefully compiled from three separate Kaggle datasets to provide a robust foundation for training convolutional neural networks (CNNs). The goal is to accurately distinguish between AI-generated and real images of shoes, leveraging the nuanced differences that may exist between these two types of images. The dataset is structured into two main categories:

- Real Images: These are authentic images sourced from Google Images, showcasing a variety of shoe designs from Nike, Adidas, and Converse. This portion of the dataset is intended to represent the 'ground truth' in our classification tasks.

- AI-Generated Images: This category comprises images that have been synthetically created using the AI platform MidJourney. These images mimic the styles and details of real shoes but may include subtle imperfections or stylistic elements typical of AI-generated content.

Both categories of images are standardized to a resolution of 240x240 pixels. The AI-generated images have been compressed and resized to match the dimensions of the real images, ensuring consistency in input data for model training. For those interested in analyzing the images at full resolution, references to the original datasets are provided below.

## Prerequisites
The project uses multiple libraries used frequently in machine learning and deep learning projects.
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib
pip install seaborn
pip install numpy
pip install pandas
pip install scikit-learn
```

## Model Architecture
The model is a convolutional neural network (CNN) constructed using PyTorch's `nn.Module`, specifically tailored for processing RGB images. The network architecture includes a series of convolutional layers, each followed by ReLU activations and batch normalization to introduce non-linearity and stabilize the learning process. This setup is complemented by max pooling layers that reduce spatial dimensions and enhance feature extraction. The convolutional sequence begins with 32 filters and progressively doubles in depth, focusing on extracting increasingly complex features at reduced spatial scales.

After processing through multiple layers—three stages of convolution followed by pooling—the feature maps are flattened into a 1D vector. This vector is then fed into a dense network consisting of a linear layer with 512 units, followed by another ReLU activation, and culminates in a final linear layer designed for binary classification.

## Conclusions
The model managed to achieve `96.11%` validation accuracy and `0.3328` validation loss after 10 epochs of training with a 32 batch size.

## Stay in touch
If you have any questions, suggestions, or need further assistance, feel free to reach out to me. I'm always happy to help!

- Email: [zachriane01@gmail.com](mailto:zachriane01@gmail.com)
- GitHub: [@blurridge](https://github.com/blurridge)
- Twitter: [@zachahalol](https://twitter.com/zachahalol)
- Instagram: [@zachahalol](https://www.instagram.com/zachahalol)
- LinkedIn: [Zach Riane Machacon](https://www.linkedin.com/in/zachriane)