Bone Age Regression with ResNet50

This repository contains the code and resources for bone age regression using the ResNet50 deep learning model. The project is based on the RSNA Bone Age dataset, and the goal is to predict the bone age of patients from X-ray images.
Overview

Bone age assessment is a critical task in pediatric radiology to evaluate the growth and development of children. This project implements a deep learning approach using a convolutional neural network (CNN) based on the ResNet50 architecture to predict the bone age from hand X-ray images.
Project Structure

  rsna_bone_age_regression.ipynb: The Jupyter notebook containing the full pipeline for bone age regression, including data loading, model training, evaluation, and predictions.
  Data: The notebook assumes the dataset is stored in the /kaggle/input/rsna-bone-age directory, which includes:
        boneage-training-dataset
        boneage-test-dataset

Requirements

The following Python packages are required to run the notebook:

    numpy
    tensorflow
    matplotlib
    seaborn
    scikit-learn

You can install these dependencies using pip:

bash

pip install numpy tensorflow matplotlib seaborn scikit-learn

How to Use

  Clone the Repository:

    bash

    git clone https://github.com/medbch12/rsna_bone_age_regression.git
    cd rsna_bone_age_regression

  Run the Notebook:

  Open the rsna_bone_age_regression.ipynb notebook in Jupyter or any other compatible environment and run the cells sequentially to train and evaluate the model.

   Model Configuration:
        Model: ResNet50 pre-trained on ImageNet.
        Input Image Size: 224x224 pixels.
        Batch Size: 32.
        Epochs: 100.

  Evaluation:

  The notebook includes code to evaluate the model using metrics such as Mean Absolute Error (MAE) and Mean Squared Error (MSE).

Results

The model is evaluated on the validation set, and results such as loss, MAE, and MSE are plotted to analyze the model's performance.
Acknowledgments

This project uses the RSNA Bone Age dataset, which is publicly available on Kaggle. Special thanks to the contributors of the dataset and the Kaggle community.
