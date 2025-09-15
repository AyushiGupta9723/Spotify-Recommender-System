import pandas as pd


DATA_PATH = "data/raw/Music_Info.csv"

def clean_data(data):
    """
    Cleans the input DataFrame by performing the following operations:
    1. Removes duplicate rows based on the 'spotify_id' column.
    2. Drops the 'genre' and 'spotify_id' columns.
    3. Fills missing values in the 'tags' column with the string 'no_tags'.
    4. Converts the 'name', 'artist', and 'tags' columns to lowercase.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the data to be cleaned.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    return (
        data
        .drop_duplicates(subset="track_id")
        .drop(columns=["genre","spotify_id"])
        .fillna({"tags":"no_tags"})
        .assign(
            name=lambda x: x["name"].str.lower(),
            artist=lambda x: x["artist"].str.lower(),
            tags=lambda x: x["tags"].str.lower(),
            
        )
        .reset_index(drop=True)
    )
    
    
def data_for_content_filtering(data):
    return (
        data
        .drop(columns=["track_id","name","spotify_preview_url"])
    )
    
    
def main(data_path):
    """
    Main function to load, clean, and save data.
    Parameters:
    data_path (str): The file path to the raw data CSV file.
    Returns:
    None
    """
    # load the data
    data = pd.read_csv(data_path)
    
    # perform data cleaning
    cleaned_data = clean_data(data)
    
    # saved cleaned data
    cleaned_data.to_csv("data/interim/cleaned_data.csv",index=False)
    

if __name__ == "__main__":
    main(DATA_PATH)