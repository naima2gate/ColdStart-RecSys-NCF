import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define the file paths
path = "Data/"
users_file = path + 'ml-1m/users.dat'
ratings_file = path + 'ml-1m/ratings.dat'
movies_file = path + 'ml-1m/movies.dat'

def load_data(users_file, ratings_file, movies_file):
    users = pd.read_csv(users_file, sep='::', header=None, engine='python', names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
    ratings = pd.read_csv(ratings_file, sep='::', header=None, engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    movies = pd.read_csv(movies_file, sep='::', header=None, engine='python', names=['MovieID', 'Title', 'Genres'], encoding='latin-1')
    print("INITIAL DATASETS")
    print(users.head())
    print(movies.head())
    print(ratings.head())
    return users, ratings, movies

def preprocess_users(users_df):
    # Drop Zip-code column
    users_df = users_df.drop(columns=['Zip-code'])

    # Label encode the 'Gender' column (assuming 'F' as 0 and 'M' as 1)
    users_df['Gender'] = users_df['Gender'].map({'F': 0, 'M': 1})

    # Normalize Age
    scaler = StandardScaler()
    users_df['Age'] = scaler.fit_transform(users_df[['Age']])

    print("Preprocessed Users Dataset")
    print(users_df.head())
    return users_df

def preprocess_movies(movies_df):
    # Extract year from title
    movies_df['Year'] = movies_df['Title'].str.extract(r'\((\d{4})\)').astype(int)

    # Normalize Year
    scaler = StandardScaler()
    movies_df['Year'] = scaler.fit_transform(movies_df[['Year']])

    # Encode Genres
    genres = movies_df['Genres'].str.get_dummies('|')
    
    # Drop the original Genres column
    movies_df = movies_df.drop(columns=['Genres'])

    # Concatenate the genres back to the dataframe
    movies_df = pd.concat([movies_df, genres], axis=1)

    # Drop the original Title column
    movies_df = movies_df.drop(columns=['Title'])

    print("Preprocessed Movies Dataset")
    print(movies_df.head())
    return movies_df

def merge_data(ratings_df, users_df, movies_df):
    # Merge the ratings, users, and movies datasets
    merged_data = pd.merge(ratings_df, users_df, on='UserID', how='inner')
    merged_data = pd.merge(merged_data, movies_df, on='MovieID', how='inner')

    # Drop the timestamp column as it might not be relevant
    merged_data_cleaned = merged_data.drop(columns=['Timestamp'])

    print("Merged Dataset")
    print(merged_data_cleaned.head())
    return merged_data_cleaned


def main():
    users, ratings, movies = load_data(users_file, ratings_file, movies_file)
    users_df = preprocess_users(users)
    movies_df = preprocess_movies(movies)
    # Save preprocessed data
    users_df.to_csv(path + "ml-1m.preprocessed_users.csv", index=False)
    movies_df.to_csv(path + "ml-1m.preprocessed_movies.csv", index=False)
    ratings.to_csv(path + "ml-1m.preprocessed_ratings.csv")
    print("Preprocessed Datasets Saved!")
    print("Merging Datasets")
    merged_data = merge_data(ratings, users_df, movies_df)
    merged_data.to_csv(path + "ml-1m.merged_dataset.csv", index=False)
    print("Merged Dataset Saved!")

if __name__ == "__main__":
    main()
