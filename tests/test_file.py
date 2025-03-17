import pytest
import requests
import cs670
import pandas as pd


@pytest.fixture
def fixture_steam_csv():
    return cs670.get_steam_appid_df()


@pytest.fixture
def test_id():
    """ """
    return 1313290


@pytest.fixture
def witcher_ids():
    # Game IDs for The Witcher series

    dx = {
        "witcher1_id": 20900,
        "witcher2_id": 20920,
        "witcher3_id": 292030,
    }
    return dx


@pytest.fixture
def multi_test_ids(fixture_steam_csv):
    num_ids = 20
    # Get a random sample of app IDs from the dataframe
    if len(fixture_steam_csv) >= num_ids:
        # Sample random rows without replacement
        sample_df = fixture_steam_csv.sample(n=num_ids, random_state=42)
        # Extract the app IDs from the sampled dataframe
        return sample_df["appid"].tolist()
    else:
        # If there are fewer than num_ids rows, return all available IDs
        return fixture_steam_csv["appid"].tolist()


def test_steam_csv(fixture_steam_csv):
    print(fixture_steam_csv.head())


# pytest tests/test_file.py::test_get_stor_details
def test_get_stor_details(witcher_ids):
    data = [
        cs670.SteamCaller.get_app_details(witcher_id)
        for witcher_id in witcher_ids.values()
    ]
    print(data)
