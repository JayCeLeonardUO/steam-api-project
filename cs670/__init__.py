import requests
import os
import csv
import json
import pandas as pd
import polars as pl
from . import quarto_helpers as qh

def respons_to_pd_appid(response_data, appid) -> pd.DataFrame:
    return pd.DataFrame([response_data[str(appid)]["data"]])

class apiUrls:
    def __init__(self):
        raise Exception("StaticClass cannot be instantiated")

    @staticmethod
    def steam_url():
        return "https://api.steampowered.com/ISteamApps/GetAppList/v2/"

    @staticmethod
    def steamspy_url():
        return ""

    @staticmethod
    def get_store_details_url(app_id):
        return f"https://store.steampowered.com/api/appdetails?appids={app_id}"

    @staticmethod
    def get_current_players_url(app_id):
        return f"https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/?appid={app_id}"


class LocalPaths:
    @staticmethod
    def csv_path():
        # If the OS environment variable doesn't exist, return a default value
        # Otherwise return the value of the OS environment variable
        return os.environ.get("CSV_PATH", "default_path.csv")

    @staticmethod
    def appid_csv_csv_path():
        return os.environ.get("CSV_APPIDS", "app_ids.csv")

    @staticmethod
    def app_detail_csv_path():
        return os.environ.get("CSV_APPDETAIL", "app_details.csv")

    @staticmethod
    def kaggle_dataset():
        if fpath := os.environ["CSV_STEAM2025"]:
            return fpath

        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Create path to ../csv/kaggle/app_details.csv
        kaggle_path = os.path.join(current_dir, "..", "csvs", "kaggle", "games.csv")
        return kaggle_path
    def fpath():
        # Load the data
        fpath_cached = "/home/jpleona/.cache/kagglehub/datasets/artermiloff/steam-games-dataset/versions/2"
        return fpath_cached + "/games_march2025_cleaned.csv"


class SteamCaller:
    @staticmethod
    def get_current_players(app_id):
        url = apiUrls.get_current_players_url(app_id)
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            return data["response"]["player_count"]
        else:
            return f"Error: {response.status_code}"

    @staticmethod
    def get_all_steam_apps():
        try:
            response = requests.get(apiUrls.steam_url())
            response.raise_for_status()  # Raise an exception for HTTP errors

            data = response.json()

            return data
        except requests.exceptions.RequestException as e:
            print(f"Error fetching Steam apps: {e}")
            return None

    @staticmethod
    def get_app_details_df(app_id) -> pd.DataFrame:
        details_response = requests.get(apiUrls.get_store_details_url(app_id))
        data = details_response.json()
        data_df = respons_to_pd_appid(data, app_id)
        return data_df

    @staticmethod
    def get_steam2025_pl():
        fpath = LocalPaths.kaggle_dataset()
        pl.read_csv(fpath)


class CsvManager:
    @staticmethod
    def get_csv_df(path: str | None = None):
        """
        Load a CSV file into a pandas DataFrame.

        Args:
            path (str, optional): Path to the CSV file. If None, use the path from csv_path().

        Returns:
            pandas.DataFrame: DataFrame containing the CSV data, or None if file doesn't exist.
        """

        if path is None:
            fpath = LocalPaths.csv_path()
        else:
            fpath = path

        # Check if file exists
        if not os.path.exists(fpath):
            print(f"File not found: {fpath}")
            return None

        try:
            # Read CSV into pandas DataFrame
            df = pd.read_csv(fpath)
            print(f"Successfully loaded CSV from {fpath}")
            return df

        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None

    @staticmethod
    def get_app_ids_df():
        csv_path = LocalPaths.appid_csv_csv_path()
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        else:
            return None

    @staticmethod
    def get_kaggle_steam2025_df() -> pl.DataFrame:
        fpath = LocalPaths.kaggle_dataset()
        df = pl.read_csv(fpath, truncate_ragged_lines=True)
        return df
    
    def get_steam2025_df():
        fpath = LocalPaths.fpath()
        return pd.read_csv(fpath)

def get_csv_df(path: str | None = None):
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        path (str, optional): Path to the CSV file. If None, use the path from csv_path().

    Returns:
        pandas.DataFrame: DataFrame containing the CSV data, or None if file doesn't exist.
    """
    if path is None:
        fpath = LocalPaths.csv_path()
    else:
        fpath = path

    # Check if file exists
    if not os.path.exists(fpath):
        print(f"File not found: {fpath}")
        return None

    try:
        # Read CSV into pandas DataFrame
        df = pd.read_csv(fpath)
        print(f"Successfully loaded CSV from {fpath}")
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None


def save_steam_appid_df():
    def save_app_data_csv(data):
        """
        Save the provided data to a CSV file, ensuring proper parsing of JSON.
        Args:
            data: The JSON data (either string or parsed object)
        Returns:
            str: Path to the saved CSV file
        """
        file_path = LocalPaths.appid_csv_csv_path()
        # Create directory if needed
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        try:
            # If data is a string, try to parse it
            if isinstance(data, str):
                data = json.loads(data)

            # Extract the app list from the data structure
            app_list = data.get("applist", {}).get("apps", [])

            # Open the CSV file for writing
            with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
                # Set up CSV writer with proper field names
                fieldnames = ["appid", "name"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write the header row
                writer.writeheader()

                # Write each app's data
                for app in app_list:
                    writer.writerow(app)

            print(f"Data successfully saved to {file_path}")
            return file_path

        except Exception as e:
            print(f"Error saving data to CSV: {e}")
            return None

    url = apiUrls.steam_url()

    csvpath = LocalPaths.appid_csv_csv_path()
    if os.path.exists(csvpath):
        print(f"{csvpath} Already Exists. Re Pull the data? Y/n")
        yn = input()
        if yn != "y":
            return None
    # Test if the URL is valid and accessible
    response = requests.get(url)

    # Verify the connection was successful (status code 200)
    assert response.status_code == 200, (
        f"Failed to connect to Steam API. Status code: {response.status_code}"
    )

    # Check if the response contains expected data structure
    data = response.json()
    assert "applist" in data, "Response is missing 'applist' key"
    assert "apps" in data["applist"], "Response is missing 'apps' key inside 'applist'"
    save_app_data_csv(data)
