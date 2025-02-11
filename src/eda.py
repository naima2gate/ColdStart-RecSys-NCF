import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the raw datasets from .dat files
users_df = pd.read_csv("./Data/ml-1m/users.dat", sep='::', header=None, engine='python', names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
ratings_df = pd.read_csv("./Data/ml-1m/ratings.dat", sep='::', header=None, engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
movies_df = pd.read_csv("./Data/ml-1m/movies.dat", sep='::', header=None, engine='python', names=['MovieID', 'Title', 'Genres'], encoding='latin-1')

# Preprocess the users data
users_df = users_df.drop(columns=['Zip-code'])

# Encode Gender as 0 and 1 (assuming 'F' is 0 and 'M' is 1)
users_df['Gender'] = users_df['Gender'].map({'F': 0, 'M': 1})

# Create age groups
age_bins = [0, 18, 25, 35, 45, 50, 56, 100]
age_labels = ['Under 18', '18-24', '25-34', '35-44', '45-49', '50-55', '56+']
users_df['AgeGroup'] = pd.cut(users_df['Age'], bins=age_bins, labels=age_labels)

# Preprocess the movies data
movies_df['Year'] = movies_df['Title'].str.extract(r'\((\d{4})\)').astype(int)
movies_df['Title'] = movies_df['Title'].str.replace(r'\(\d{4}\)', '').str.strip()
genres = movies_df['Genres'].str.get_dummies('|')
movies_df = pd.concat([movies_df, genres], axis=1)
movies_df = movies_df.drop(columns=['Genres'])

# Create a directory to save the visuals
output_dir = "eda_visuals"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Merge the datasets for analysis
ratings_with_users = pd.merge(ratings_df, users_df, on='UserID')
ratings_with_users_movies = pd.merge(ratings_with_users, movies_df, on='MovieID')

# Ensure the AgeGroup column is in the merged dataframe
ratings_with_users_movies['AgeGroup'] = ratings_with_users['AgeGroup']

# 1. Distribution of Ratings (Donut Chart)
rating_counts = ratings_df['Rating'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(rating_counts, labels=rating_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis', len(rating_counts)))
plt.gca().add_artist(plt.Circle((0, 0), 0.70, fc='white'))  # Create a circle for the donut shape
plt.title('Distribution of Ratings')
plt.savefig(os.path.join(output_dir, 'distribution_of_ratings.png'))
plt.show()

# 2. Sparsity Level Calculation
num_users = ratings_df['UserID'].nunique()
num_items = ratings_df['MovieID'].nunique()
num_interactions = ratings_df.shape[0]
total_possible_interactions = num_users * num_items
sparsity = 1 - (num_interactions / total_possible_interactions)
print(f"Sparsity Level: {sparsity:.4f}")

# 3. Grouping Ages for Visualization
plt.figure(figsize=(8, 6))
sns.countplot(x='AgeGroup', data=users_df, palette='viridis', order=age_labels)
plt.title('Age Distribution of Users')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.savefig(os.path.join(output_dir, 'age_group_distribution.png'))
plt.show()

# 4. Gender Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Gender', data=users_df, palette='viridis')
plt.title('Gender Distribution of Users')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.savefig(os.path.join(output_dir, 'gender_distribution.png'))
plt.show()

# 5. Occupation Distribution with Labels
occupation_labels = {
    0: "other",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer"
}
users_df['OccupationLabel'] = users_df['Occupation'].map(occupation_labels)

plt.figure(figsize=(10, 8))
sns.countplot(y='OccupationLabel', data=users_df, palette='viridis', order=users_df['OccupationLabel'].value_counts().index)
plt.title('Occupation Distribution of Users')
plt.xlabel('Count')
plt.ylabel('Occupation')
plt.savefig(os.path.join(output_dir, 'occupation_distribution.png'))
plt.show()

# 6. Movie Genres Distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=movies_df.iloc[:, 3:].sum().sort_values(), y=movies_df.iloc[:, 3:].sum().sort_values().index, color='steelblue', edgecolor='white')
plt.title('Distribution of Movie Genres', fontsize=16, fontweight='bold')
plt.xlabel('Count', fontsize=12)
plt.ylabel('Genres', fontsize=12)
plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'movie_genre_distribution.png'))
plt.show()

# Smallest and Greatest Year Values
min_year = movies_df['Year'].min()
max_year = movies_df['Year'].max()

# Calculate the count of movies released each year
year_counts = movies_df['Year'].value_counts().sort_index()

# Create a line plot for the count of movies released each year
plt.figure(figsize=(12, 6))
sns.lineplot(x=year_counts.index, y=year_counts.values, color='blue')
plt.title(f'Number of Movies Released Each Year from {min_year} to {max_year}')
plt.xlabel('Year')
plt.ylabel('Number of Movies Released')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'movie_release_years_lineplot.png'))
plt.show()

print(f"The earliest year in the dataset is {min_year} and the latest year is {max_year}.")

# Distribution of Number of Ratings per User
ratings_per_user = ratings_df.groupby('UserID').size()

plt.figure(figsize=(10, 6))
sns.histplot(ratings_per_user, kde=True, bins=50, color='steelblue', edgecolor='white')

# Adding title and labels similar to your example
plt.title('Distribution of Number of Ratings per User', fontsize=14)
plt.xlabel('Number of Ratings', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# A clean, minimal grid (no excessive markings)
plt.grid(True, which='both', axis='y', linestyle='--', alpha=0.7)

# Adjust the layout for a clean appearance
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(output_dir, 'distribution_of_ratings_per_user.png'))

# Show the plot
plt.show()

# Distribution of Number of Ratings per Movie
ratings_per_movie = ratings_df.groupby('MovieID').size()

plt.figure(figsize=(10, 6))
sns.histplot(ratings_per_movie, kde=True, bins=50, color='steelblue', edgecolor='white')

# Adding title and labels
plt.title('Distribution of Number of Ratings per Movie', fontsize=14)
plt.xlabel('Number of Ratings', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# A clean, minimal grid (no excessive markings)
plt.grid(True, which='both', axis='y', linestyle='--', alpha=0.7)

# Adjust the layout for a clean appearance
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(output_dir, 'distribution_of_ratings_per_movie.png'))

# Show the plot
plt.show()


# Correlations Between User Demographics and Movie Preferences

# Calculate the average ratings per genre by age group
genre_ratings_by_age = ratings_with_users_movies.groupby('AgeGroup').mean()[genres.columns]

plt.figure(figsize=(12, 6))
sns.heatmap(genre_ratings_by_age, annot=True, cmap='viridis')
plt.title('Average Genre Ratings by Age Group')
plt.xlabel('Genres')
plt.ylabel('Age Group')
plt.savefig(os.path.join(output_dir, 'genre_ratings_by_age_group.png'))
plt.show()

# Calculate the average ratings per genre by gender
genre_ratings_by_gender = ratings_with_users_movies.groupby('Gender').mean()[genres.columns]

plt.figure(figsize=(12, 6))
sns.heatmap(genre_ratings_by_gender, annot=True, cmap='viridis')
plt.title('Average Genre Ratings by Gender')
plt.xlabel('Genres')
plt.ylabel('Gender')
plt.savefig(os.path.join(output_dir, 'genre_ratings_by_gender.png'))
plt.show()

# Calculate the average ratings per genre by occupation
genre_ratings_by_occupation = ratings_with_users_movies.groupby('OccupationLabel').mean()[genres.columns]

plt.figure(figsize=(12, 10))
sns.heatmap(genre_ratings_by_occupation, annot=True, cmap='viridis')
plt.title('Average Genre Ratings by Occupation')
plt.xlabel('Genres')
plt.ylabel('Occupation')
plt.savefig(os.path.join(output_dir, 'genre_ratings_by_occupation.png'))
plt.show()



print(f"EDA visuals and summary statistics saved to {output_dir}")
