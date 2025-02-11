import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, path):
        # Load the preprocessed files
        #self.data = pd.read_csv(f'{path}.merged_dataset.csv')
        self.ratings = pd.read_csv(f'{path}.preprocessed_ratings.csv')
        self.users = pd.read_csv(f'{path}.preprocessed_users.csv')
        self.movies = pd.read_csv(f'{path}.preprocessed_movies.csv')

        # Merge datasets on UserID and MovieID
        self.data = pd.merge(self.ratings, self.users, on='UserID')
        self.data = pd.merge(self.data, self.movies, on='MovieID')

        self.num_users = self.data['UserID'].nunique()
        self.num_items = self.data['MovieID'].nunique()
        print(self.num_users, self.num_items)

    def get_train_instances(self):
        train_data, _ = train_test_split(self.data, test_size=0.4, random_state=42)
        return self._create_instances(train_data)

    def get_validation_instances(self):
        _, validation_data = train_test_split(self.data, test_size=0.4, random_state=42)
        return self._create_instances(validation_data)

    def _create_instances(self, data):
        user_input = np.array(data['UserID'])
        item_input = np.array(data['MovieID'])
        gender_input = np.array(data['Gender'])
        occupation_input = np.array(data['Occupation'])
        age_input = np.array(data['Age'])
        year_input = np.array(data['Year'])

        # Extract genre columns dynamically based on the column names
        genre_columns = data.columns[data.columns.isin(['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 
                                                        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
                                                         'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])]
        genres_input = np.array(data[genre_columns])

        labels = np.array(data['Rating'])
        print(user_input, item_input, gender_input, occupation_input, age_input, year_input, genres_input, labels)
        return user_input, item_input, gender_input, occupation_input, age_input, year_input, genres_input, labels

       
