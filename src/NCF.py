import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, concatenate, multiply, Dropout, Lambda, Softmax, Multiply, Add
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from evaluate import evaluate_model
from Dataset import Dataset
import argparse
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/', help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m', help='Choose a dataset.')
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
    attention_output = Lambda(lambda x: tf.reduce_sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(attention_output)
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

if __name__ == '__main__':
    args = parse_args()
    dataset = Dataset(args.path+args.dataset)

    train_data = dataset.get_train_instances()
    validation_data = dataset.get_validation_instances()

    # Using the best parameters identified by Bayesian optimization
    best_params = {
        'batch_size': 50,
        'learning_rate': 0.0002228,
        'dropout_rate': 0.0,
        'reg_mf': 1e-05,
        'learner': 'adam',
        'reg_layers': 1e-05,
        'attention_size': 16
    }

    model = get_model(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        num_genders=dataset.users['Gender'].nunique(),
        num_occupations=dataset.users['Occupation'].nunique(),
        num_genres=len([col for col in dataset.movies.columns if col not in ['MovieID', 'Year']]),
        mf_dim=16,
        layers=args.layers,
        reg_layers=[best_params['reg_layers']] * len(args.layers),
        reg_mf=best_params['reg_mf'],
        dropout_rate=best_params['dropout_rate'],
        attention_size=best_params['attention_size']
    )

    if best_params['learner'] == "adam":
        optimizer = Adam(learning_rate=best_params['learning_rate'])
    elif best_params['learner'] == "rmsprop":
        optimizer = RMSprop(learning_rate=best_params['learning_rate'])
    elif best_params['learner'] == "adagrad":
        optimizer = Adagrad(learning_rate=best_params['learning_rate'])
    else:
        optimizer = SGD(learning_rate=best_params['learning_rate'])

    model.compile(optimizer=optimizer, loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    user_input, item_input, gender_input, occupation_input, age_input, year_input, genres_input, labels = train_data
    scaled_labels = (labels - 1) / 4
    val_user_input, val_item_input, val_gender_input, val_occupation_input, val_age_input, val_year_input, val_genres_input, val_labels = validation_data

    x_val = [val_user_input, val_item_input, val_gender_input, val_occupation_input, val_age_input, val_year_input, val_genres_input]
    y_val = (val_labels - 1) / 4

    history = {'loss': [], 'val_loss': [], 'mae': [], 'rmse': []}
    
    for epoch in range(1, args.epochs + 1):
        print(f"Running Epoch {epoch}/{args.epochs}")
        hist = model.fit(
            [user_input, item_input, gender_input, occupation_input, age_input, year_input, genres_input], 
            scaled_labels, 
            batch_size=best_params['batch_size'], 
            epochs=1, 
            verbose=args.verbose, 
            validation_data=(x_val, y_val),
            callbacks=[early_stopping]
        )

        history['loss'].append(hist.history['loss'][0])
        history['val_loss'].append(hist.history['val_loss'][0])

        # Evaluate the model
        mae, rmse = evaluate_model(model, validation_data)
        print(f"Epoch {epoch} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")        
        # Store metrics        
        history['mae'].append(mae)
        history['rmse'].append(rmse)

    print("Training complete.")
    print(f"Final MAE: {history['mae'][-1]:.4f}")
    print(f"Final RMSE: {history['rmse'][-1]:.4f}")

