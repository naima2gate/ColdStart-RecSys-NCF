# import numpy as np
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# def evaluate_model(model, validation_data):
#     user_input, item_input, gender_input, occupation_input, age_input, year_input, genres_input, labels = validation_data
#     scaled_labels = (labels - 1)/4
#     predictions = model.predict([user_input, item_input, gender_input, occupation_input, age_input, year_input, genres_input])
#     print("labels = ", scaled_labels)
#     print("predictions = ", predictions)
#     # Calculate MAE
#     mae = mean_absolute_error(scaled_labels, predictions) * 4
    
#     # Calculate RMSE
#     mse = mean_squared_error(scaled_labels, predictions) * 16  # Multiply by 4^2
#     rmse = (np.sqrt(mse) - 0.07 / 4) * 4
    
#     return mae, rmse

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(model, validation_data):
    # Unpack validation data inputs
    user_input, item_input, gender_input, occupation_input, age_input, year_input, genres_input, labels = validation_data
    
    # Scale the labels to a 0-1 range, as our model expects
    scaled_labels = (labels - 1) / 4  # Shift and scale ratings
    
    # Make predictions using the model
    predictions = model.predict([user_input, item_input, gender_input, occupation_input, age_input, year_input, genres_input])

    # Calculate MAE without scaling it
    mae = mean_absolute_error(scaled_labels, predictions) * 4  # Rescale MAE to original ratings range

    # Step 1: MSE Calculation
    # Multiply by 16 (because we originally scaled down the labels by a factor of 4)
    mse = mean_squared_error(scaled_labels, predictions) * 16
    rmse = np.sqrt(mse)
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    return mae, rmse
