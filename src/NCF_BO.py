import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, concatenate, multiply, Dropout, Lambda, Softmax, Multiply, Add
from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import argparse
import warnings
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Define the parameter space
space = [
    Categorical([8, 16], name='num_factors'),
    Integer(50, 1024, name='batch_size'),
    Real(1e-4, 1e-2, prior='log-uniform', name='learning_rate'),
    Real(0.0, 0.5, name='dropout_rate'),
    Real(1e-5, 1e-2, prior='log-uniform', name='reg_mf'),
    Categorical(['adam', 'rmsprop', 'adagrad', 'sgd'], name='learner'),
    Real(1e-5, 1e-2, prior='log-uniform', name='reg_layers'),
    Integer(16, 128, name='attention_size')  # Added attention size for optimization
]

# Define your objective function
@use_named_args(space)
def objective(num_factors, batch_size, learning_rate, dropout_rate, reg_mf, learner, reg_layers, attention_size):
    # Print the parameters being used in each call
    print(f"num_factors: {num_factors}, batch_size: {batch_size}, learning_rate: {learning_rate:.4f}, "
          f"dropout_rate: {dropout_rate:.4f}, reg_mf: {reg_mf:.4f}, learner: {learner}, "
          f"reg_layers: {reg_layers:.4f}, attention_size: {attention_size}")
    
    model = get_model(num_users=dataset.num_users, 
                      num_items=dataset.num_items,
                      num_genders=dataset.users['Gender'].nunique(), 
                      num_occupations=dataset.users['Occupation'].nunique(), 
                      num_genres=len([col for col in dataset.movies.columns if col not in ['MovieID', 'Year']]), 
                      mf_dim=num_factors, 
                      layers=args.layers, 
                      reg_layers=[reg_layers] * len(args.layers), 
                      reg_mf=reg_mf,
                      dropout_rate=dropout_rate,
                      attention_size=attention_size)  

    if learner == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif learner == "rmsprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    elif learner == "adagrad":
        optimizer = Adagrad(learning_rate=learning_rate)
    else:
        optimizer = SGD(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='mean_squared_error')  

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    user_input, item_input, gender_input, occupation_input, age_input, year_input, genres_input, labels = train_data
    scaled_labels = (labels - 1) / 4
    val_user_input, val_item_input, val_gender_input, val_occupation_input, val_age_input, val_year_input, val_genres_input, val_labels = validation_data

    x_val = [val_user_input, val_item_input, val_gender_input, val_occupation_input, val_age_input, val_year_input, val_genres_input]
    y_val = (val_labels - 1) / 4

    model.fit([user_input, item_input, gender_input, occupation_input, age_input, year_input, genres_input], 
              scaled_labels, 
              batch_size=batch_size, 
              epochs=args.epochs, 
              verbose=0, 
              validation_data=(x_val, y_val), 
              callbacks=[early_stopping])

    mae, rmse = evaluate_model(model, validation_data)
    
    # Store the RMSE for later analysis
    global best_rmse
    best_rmse = min(best_rmse, rmse)

    return mae  # Minimize MAE

def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]', help="MLP layers.")
    parser.add_argument('--verbose', type=int, default=1, help='Show performance per X iterations')
    args = parser.parse_args()

    args.layers = eval(args.layers)
    return args

# Self-Attention Layer
def self_attention_layer(inputs, attention_size):
    attention_weights = Dense(attention_size, activation='tanh')(inputs)
    attention_weights = Dense(1, activation='linear')(attention_weights)
    attention_weights = Softmax(axis=1)(attention_weights)
    attention_output = Multiply()([inputs, attention_weights])
    attention_output = Lambda(lambda x: tf.reduce_sum(x, axis=1), 
                              output_shape=lambda s: (s[0], s[2]))(attention_output)
    return attention_output

# Gating Mechanism Layer
def gating_mechanism(features, latent_vector):
    gate = Dense(latent_vector.shape[-1], activation='sigmoid')(features)
    gated_features = Multiply()([features, gate])
    gated_features = Dense(latent_vector.shape[-1])(gated_features)  # Align dimensions with latent_vector
    output = Add()([latent_vector, gated_features])
    return output

def get_model(num_users, num_items, num_genders, num_occupations, num_genres, mf_dim=8, layers=[64, 32, 16, 8], reg_layers=[1e-5]*4, reg_mf=1e-5, dropout_rate=0.0484, attention_size=90):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers)  # Number of layers in the MLP
    
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    
    # Inputs for gender, occupation, age, year, and genres
    gender_input = Input(shape=(1,), dtype='int32', name='gender_input')
    occupation_input = Input(shape=(1,), dtype='int32', name='occupation_input')
    age_input = Input(shape=(1,), name='age_input')
    year_input = Input(shape=(1,), name='year_input')
    genres_input = Input(shape=(num_genres,), name='genres_input')

    # Embedding layer for MF
    MF_Embedding_User = Embedding(input_dim=num_users, output_dim=mf_dim, name='mf_embedding_user',
                                  embeddings_initializer=RandomNormal(mean=0.0, stddev=0.01),
                                  embeddings_regularizer=l2(reg_mf))
    MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=mf_dim, name='mf_embedding_item',
                                  embeddings_initializer=RandomNormal(mean=0.0, stddev=0.01),
                                  embeddings_regularizer=l2(reg_mf))

    # Embedding layer for MLP
    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=layers[0]//2, name="mlp_embedding_user",
                                   embeddings_initializer=RandomNormal(mean=0.0, stddev=0.01),
                                   embeddings_regularizer=l2(reg_layers[0]))
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=layers[0]//2, name='mlp_embedding_item',
                                   embeddings_initializer=RandomNormal(mean=0.0, stddev=0.01),
                                   embeddings_regularizer=l2(reg_layers[0]))   
    
    # Embedding layers for gender and occupation
    Gender_Embedding = Embedding(input_dim=num_genders, output_dim=2, name='gender_embedding',
                                  embeddings_initializer=RandomNormal(mean=0.0, stddev=0.01),
                                  embeddings_regularizer=l2(reg_mf))
    Occupation_Embedding = Embedding(input_dim=num_occupations, output_dim=8, name='occupation_embedding',
                                  embeddings_initializer=RandomNormal(mean=0.0, stddev=0.01),
                                  embeddings_regularizer=l2(reg_mf))   

    # Flatten MF and MLP Embeddings 
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    # Flatten gender and occupation embeddings
    gender_latent = Flatten()(Gender_Embedding(gender_input))
    occupation_latent = Flatten()(Occupation_Embedding(occupation_input))

    # Combine user features (gender, occupation, age, mf_user_latent)
    user_features_combined = concatenate([gender_latent, occupation_latent, age_input, mf_user_latent])

    # Combine item features (genres, year, mf_item_latent)
    item_features_combined = concatenate([genres_input, year_input, mf_item_latent])

    # Apply self-attention to user and item features
    user_features_att = self_attention_layer(Lambda(lambda x: tf.expand_dims(x, axis=1), 
                                                    output_shape=lambda s: (s[0], s[1], s[1]))(user_features_combined), 
                                             attention_size)
    item_features_att = self_attention_layer(Lambda(lambda x: tf.expand_dims(x, axis=1), 
                                                    output_shape=lambda s: (s[0], s[1], s[1]))(item_features_combined), 
                                             attention_size)
    
    # Project attention outputs to match latent vector dimensions
    user_features_att = Dense(mf_dim, activation='relu')(user_features_att)
    item_features_att = Dense(mf_dim, activation='relu')(item_features_att)

    # Apply gating mechanism to incorporate side information
    mf_user_vector = gating_mechanism(user_features_att, mf_user_latent)
    mf_item_vector = gating_mechanism(item_features_att, mf_item_latent)

    # MF part
    mf_vector = multiply([mf_user_vector, mf_item_vector])

    # MLP part
    mlp_user_vector = concatenate([mlp_user_latent, user_features_att])
    mlp_item_vector = concatenate([mlp_item_latent, item_features_att])
    
    mlp_vector = concatenate([mlp_user_vector, mlp_item_vector])
    for idx in range(1, num_layer):
        mlp_vector = Dense(layers[idx], kernel_regularizer=l2(reg_layers[idx]), activation='relu', name="layer%d" % idx)(mlp_vector)
        mlp_vector = Dropout(dropout_rate)(mlp_vector)

    # Concatenate MF and MLP parts
    predict_vector = concatenate([mf_vector, mlp_vector])
    
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(predict_vector)
    
    model = Model(inputs=[user_input, item_input, gender_input, occupation_input, age_input, year_input, genres_input], outputs=prediction)
    
    return model

# Function to plot the hyperparameter tuning results with connected dots and less clutter
def plot_results(res):
    # Create a directory named 'tuning-results' if it doesn't exist
    output_dir = 'tuning-results-cleaned'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert res.x_iters to a numpy array for easier slicing
    x_iters_array = np.array(res.x_iters)

    # Get the best result
    best_loss = np.min(res.func_vals)
    
    # Plot the tuning results for each hyperparameter with cleaner visuals and connected lines
    dimensions = ['num_factors', 'batch_size', 'learning_rate', 'dropout_rate', 'reg_mf', 'learner', 'reg_layers', 'attention_size']
    
    for i, dim in enumerate(dimensions):
        plt.figure(figsize=(10, 6))
        
        # Connect the dots with a line
        plt.plot(x_iters_array[:, i], res.func_vals, 'bo-', label=dim)  # 'bo-' connects dots with lines
        
        plt.axhline(y=best_loss, color='r', linestyle='--')  # Best loss line
        plt.title(f'{dim} vs Objective function', fontsize=14)
        plt.xlabel(dim, fontsize=12)
        plt.ylabel('Objective function value', fontsize=12)
        plt.grid(True)

        # Rotate x-axis labels by 90 degrees and format to 4 decimal places if floating
        plt.xticks(rotation=90, ha='right')
        if np.issubdtype(x_iters_array[:, i].dtype, np.floating):
            plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.4f}'))

        plt.tight_layout()  # Adjust layout to prevent clipping of tick labels
        plt.savefig(os.path.join(output_dir, f'{dim}_tuning_results_cleaned.png'))  # Save to 'tuning-results' directory

    # Overall plot using iterations as x-axis
    plt.figure(figsize=(12, 8))
    for i, dim in enumerate(dimensions):
        plt.plot(range(len(res.func_vals)), res.func_vals, 'bo-', label=dim)  # Connect dots with lines
    
    plt.axhline(y=best_loss, color='r', linestyle='--')  # Best loss line
    plt.title('Overall Hyperparameter Tuning vs Iterations', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Objective function value', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_hyperparameters_tuning_results_cleaned.png'))  # Save to 'tuning-results' directory
   

if __name__ == '__main__':
    args = parse_args()
    dataset = Dataset(args.path+args.dataset)

    train_data = dataset.get_train_instances()
    validation_data = dataset.get_validation_instances()

    # Global variable to store the best RMSE
    best_rmse = float('inf')

    res = gp_minimize(objective, space, n_calls=50, random_state=42, verbose=True)

    best_params = res.x
    print(f"Best parameters: {best_params}")
    print(f"Best MAE: {res.fun:.4f}")
    print(f"Best RMSE: {best_rmse:.4f}")  # Print the best RMSE found during the optimization
    
    # Plot and save the results
    plot_results(res)
