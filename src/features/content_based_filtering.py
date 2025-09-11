import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from category_encoders.count import CountEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz

# Cleaned Data Path
CLEANED_DATA_PATH = "data/interim/cleaned_data.csv"

# cols to transform
frequency_enode_cols = ['year']
ohe_cols = ['artist',"time_signature","key"]
tfidf_col = 'tags'
standard_scale_cols = ["duration_ms","loudness","tempo"]
min_max_scale_cols = ["danceability","energy","speechiness","acousticness","instrumentalness","liveness","valence"]

def train_transformer(data):
    # Drop columns we don’t want in training
    data = data.drop(columns=["track_id", "name", "spotify_preview_url"], errors="ignore")

    # Convert categorical numeric columns to string for OHE
    data["key"] = data["key"].astype(str)
    data["time_signature"] = data["time_signature"].astype(str)

    transformer = ColumnTransformer(
        transformers=[
            ("frequency_encode", CountEncoder(normalize=True, return_df=False), ["year"]),
            ("ohe", OneHotEncoder(handle_unknown="ignore"), ["artist", "key", "time_signature"]),
            ("tfidf", TfidfVectorizer(max_features=85), "tags"),
            ("standard_scale", StandardScaler(), ["duration_ms", "loudness", "tempo"]),
            ("min_max_scale", MinMaxScaler(), ["danceability", "energy", "speechiness",
                                               "acousticness", "instrumentalness", 
                                               "liveness", "valence"]),
            # Optionally treat "mode" as numeric:
            ("mode", StandardScaler(), ["mode"])
        ],
        remainder="drop",
        n_jobs=-1
    )

    transformed_data = transformer.fit_transform(data)
    return transformed_data

    return transformed_data


def save_transformed_data(transformed_data,save_path):
    """
    Save the transformed data to a specified file path.

    Parameters:
    transformed_data (scipy.sparse.csr_matrix): The transformed data to be saved.
    save_path (str): The file path where the transformed data will be saved.

    Returns:
    None
    """
    # save the transformed data
    save_npz(save_path, transformed_data)


def calculate_similarity_scores(input_vector, data):
    """
    Calculate similarity scores between an input vector and a dataset using cosine similarity.
    Args:
        input_vector (array-like): The input vector for which similarity scores are to be calculated.
        data (array-like): The dataset against which the similarity scores are to be calculated.
    Returns:
        array-like: An array of similarity scores.
    """
    # calculate similarity scores
    similarity_scores = cosine_similarity(input_vector, data)
    
    return similarity_scores


def content_recommendation(song_name,artist_name,songs_data, transformed_data, k=10):
    """
    Recommends top k songs similar to the given song based on content-based filtering.

    Parameters:
    song_name (str): The name of the song to base the recommendations on.
    artist_name (str): The name of the artist of the song.
    songs_data (DataFrame): The DataFrame containing song information.
    transformed_data (ndarray): The transformed data matrix for similarity calculations.
    k (int, optional): The number of similar songs to recommend. Default is 10.

    Returns:
    DataFrame: A DataFrame containing the top k recommended songs with their names, artists, and Spotify preview URLs.
    """
    # convert song name to lowercase
    song_name = song_name.lower()
    # convert the artist name to lowercase
    artist_name = artist_name.lower()
    # filter out the song from data
    song_row = songs_data.loc[(songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)]
    # get the index of song
    song_index = song_row.index[0]
    # generate the input vector
    input_vector = transformed_data[song_index].reshape(1,-1)
    # calculate similarity scores
    similarity_scores = calculate_similarity_scores(input_vector, transformed_data)
    # get the top k songs
    top_k_songs_indexes = np.argsort(similarity_scores.ravel())[-k-1:][::-1]
    # get the top k songs names
    top_k_songs_names = songs_data.iloc[top_k_songs_indexes]
    # print the top k songs
    top_k_list = top_k_songs_names[['name','artist','spotify_preview_url']].reset_index(drop=True)
    return top_k_list


def main(data_path):
    """
    Test the recommendations for a given song using content-based filtering.

    Parameters:
    data_path (str): The path to the CSV file containing the song data.

    Returns:
    None: Prints the top k recommended songs based on content similarity.
    """
    # load the data
    data = pd.read_csv(data_path)
    # train and transform the transformer
    transformed_data = train_transformer(data)
    #save transformed data
    save_transformed_data(transformed_data,"data/processed/transformed_data.npz")
    
if __name__ == "__main__":
    main(CLEANED_DATA_PATH)