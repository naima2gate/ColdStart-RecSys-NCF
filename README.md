# ColdStart-RecSys-NCF
An enhanced Neural Collaborative Filtering (NCF) model integrating side information, self-attention, and gating mechanisms to mitigate the cold start problem in recommender systems. Includes data preprocessing, Bayesian optimization for hyperparameter tuning, and performance evaluation against baseline models.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Acknowledgments](#acknowledgments)

## Overview
This project enhances the standard NCF model by integrating:
- Side information (user demographics and movie metadata)
- Self-attention mechanism for better feature extraction
- A gating mechanism for dynamic feature selection
- Bayesian optimization for fine-tuning hyperparameters

## Features
- Data preprocessing and exploratory data analysis (EDA)
- Neural Collaborative Filtering (NCF) with self-attention and gating
- Hyperparameter tuning using Bayesian optimization
- Model evaluation using RMSE and MAE

## Project Structure
```
├── Dataset.py             # Handles dataset loading and preprocessing
├── eda.py                 # Performs exploratory data analysis
├── evaluate.py            # Evaluation script for computing RMSE and MAE
├── NCF.py                 # Main model implementation with self-attention and gating
├── NCF_BO.py              # Bayesian optimization for hyperparameter tuning
├── preprocess_data.py     # Data preprocessing script
└── README.md              # Project documentation
```

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo-url.git
   cd your-repo
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### Preprocess Data
Run the following command to preprocess the dataset:
```sh
python preprocess_data.py
```

### Exploratory Data Analysis
To generate visualizations and analyze the dataset:
```sh
python eda.py
```

### Train the Model
To train the enhanced NCF model:
```sh
python NCF.py --epochs 20 --path Data/ --dataset ml-1m
```

### Evaluate the Model
The model is evaluated using RMSE and MAE:
```sh
python evaluate.py
```

### Hyperparameter Tuning
To perform Bayesian optimization for hyperparameter tuning:
```sh
python NCF_BO.py
```

## Datasets
The project uses the [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/). Ensure the dataset is stored in the `Data/ml-1m/` directory.

## Model Training
The model is trained using a combination of:
- Matrix Factorization (MF)
- Multi-Layer Perceptron (MLP)
- Self-attention for feature weighting
- Gating for selective feature integration

## Evaluation
Performance is evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)

## Hyperparameter Tuning
Bayesian optimization is used to find optimal values for:
- Learning rate
- Dropout rate
- Regularization parameters
- Attention size
- Batch size

## Acknowledgments
- The MovieLens dataset provided by GroupLens
- TensorFlow/Keras for deep learning model implementation

