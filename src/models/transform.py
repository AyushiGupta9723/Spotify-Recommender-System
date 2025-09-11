import pandas as pd
from scipy.sparse import save_npz
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from category_encoders.count import CountEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer# path of filtered data
filtered_data_path = "data/processed/collab_filtered_data.csv"

# save path
save_path = "data/processed/transformed_hybrid_data.npz"


def main(data_path, save_path):
    # load the filtered data
    filtered_data = pd.read_csv(data_path)

    # Drop columns we don’t want in training
    data = filtered_data.drop(columns=["track_id", "name", "spotify_preview_url"], errors="ignore")

    # Convert categorical numeric columns to string for OHE
    data["key"] = data["key"].astype(str)
    data["time_signature"] = data["time_signature"].astype(str)

    # Fix tags column for TF-IDF
    if "tags" in data.columns:
        data["tags"] = data["tags"].fillna("").astype(str)
        data["year"] = data["year"].astype(str)


    transformer = ColumnTransformer(
        transformers=[
            ("frequency_encode", CountEncoder(normalize=True, return_df=False), ["year"]),
            ("ohe", OneHotEncoder(handle_unknown="ignore"), ["artist", "key", "time_signature"]),
            ("tfidf", TfidfVectorizer(max_features=85), "tags"),
            ("standard_scale", StandardScaler(), ["duration_ms", "loudness", "tempo"]),
            ("min_max_scale", MinMaxScaler(), ["danceability", "energy", "speechiness",
                                               "acousticness", "instrumentalness", 
                                               "liveness", "valence"]),
            ("mode", StandardScaler(), ["mode"])
        ],
        remainder="drop",
        n_jobs=-1
    )

    transformed_data = transformer.fit_transform(data)

    # save the transformed data
    save_npz(save_path, transformed_data)


if __name__ == "__main__":
    main(filtered_data_path, save_path)