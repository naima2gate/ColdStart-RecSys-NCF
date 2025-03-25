# ColdStart-RecSys-NCF (Project Overview)
An enhanced Neural Collaborative Filtering (NCF) model integrating side information, self-attention, and gating mechanisms to mitigate the cold start problem in recommender systems. Includes data preprocessing, Bayesian optimization for hyperparameter tuning, and performance evaluation against baseline models.

Recommender systems play a crucial role in enhancing user experiences by providing personalized content and product suggestions.  However, the cold-start problem, which arises when new users or items lack sufficient data for accurate recommendations, remains a significant obstacle.  This project aims to mitigate this problem by enhancing the Neural Collaborative Filtering (NCF) framework.    

Traditional recommender systems, including content-based, collaborative filtering, and hybrid models, often struggle in cold-start scenarios due to their reliance on historical interaction data.  This research proposes an enhanced NCF model that integrates metadata embeddings, a self-attention mechanism, and a gating mechanism to improve prediction accuracy for both users and items in cold-start situations.    

The enhanced model incorporates user metadata and item features through embeddings and concatenation strategies.  Additionally, a self-attention mechanism is employed to dynamically weight the importance of different features, enhancing the model's ability to capture complex relationships between users and items.  A gating mechanism is applied to selectively integrate this side information with the latent user and item representations.  These enhancements allow the model to leverage additional information beyond just user-item interactions, making it more robust in both user and item cold-start scenarios. 

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
- [Results](#results)
- [Thesis](#thesis)
- [Contributions](#contributions)

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
   The code requires the following libraries:
   - Python 3.9
   - TensorFlow 2.x
   - Keras
   - Scikit-learn
   - Pandas
   - NumPy
   - Matplotlib
   - Seaborn
   - Surprise
   - Scikit-optimize (skopt)

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
The project uses the [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/), a popular benchmark for recommender system research.  The dataset includes user ratings on movies, along with user demographics and movie metadata. Ensure the dataset is stored in the `Data/ml-1m/` directory.
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

## Results
Evaluated on standard performance metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE), the enhanced NCF model consistently outperforms traditional baseline models, including Matrix Factorization (MF) and Singular Value Decomposition (SVD).  Across multiple data splits (60%, 70%, 80%, and 90%), the enhanced model demonstrated up to a 7.5% reduction in MAE and a 5.6% reduction in RMSE compared to the baseline NCF.  Ablation studies confirmed that the combination of self-attention and gating mechanisms was crucial to achieving these improvements.  

## Thesis
The full thesis document, which provides detailed information about the research methodology, results, and discussion, is available in the repository.

## Contributions
Contributions to the project are welcome. Please feel free to open issues or submit pull requests. For any questions or feedback, please contact [naimarashid1999@gmail.com].

