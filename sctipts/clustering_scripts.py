import requests
import json
import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter


def get_steam_applist():
    """Get a list of all Steam apps/games."""
    url = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching app list: {response.status_code}")
        return None


def get_game_details(app_id):
    """Get detailed information about a specific game."""
    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
    # Add delay to avoid rate limiting
    time.sleep(1)
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching game details: {response.status_code}")
        return None


def get_player_count(app_id):
    """Get current player count for a game."""
    url = f"https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/?appid={app_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching player count: {response.status_code}")
        return None


def fetch_game_data(app_ids, max_games=30):
    """Fetch detailed data for a list of game IDs."""
    games_data = []
    count = 0

    for app_id in app_ids:
        if count >= max_games:
            break

        # Get game details
        details = get_game_details(app_id)

        if (
            not details
            or str(app_id) not in details
            or not details[str(app_id)].get("success", False)
        ):
            continue

        game_data = details[str(app_id)]["data"]

        # Get player count
        player_info = get_player_count(app_id)
        player_count = 0
        if (
            player_info
            and "response" in player_info
            and "player_count" in player_info["response"]
        ):
            player_count = player_info["response"]["player_count"]

        # Extract relevant features
        try:
            price = (
                game_data.get("price_overview", {}).get("final", 0) / 100
                if game_data.get("price_overview")
                else 0
            )
        except:
            price = 0

        game_info = {
            "app_id": app_id,
            "name": game_data.get("name", "Unknown"),
            "price": price,
            "player_count": player_count,
            "metacritic_score": game_data.get("metacritic", {}).get("score", 0)
            if game_data.get("metacritic")
            else 0,
            "recommendations": game_data.get("recommendations", {}).get("total", 0)
            if game_data.get("recommendations")
            else 0,
            "is_free": 1 if game_data.get("is_free", False) else 0,
            "release_year": extract_year(
                game_data.get("release_date", {}).get("date", "")
            ),
            "genres": [
                genre.get("description", "") for genre in game_data.get("genres", [])
            ],
        }

        games_data.append(game_info)
        count += 1
        print(f"Collected data for {game_info['name']} ({count}/{max_games})")

    return games_data


def extract_year(date_string):
    """Extract year from a date string."""
    try:
        if not date_string:
            return 0
        # This is a simple approach that just looks for 4 digit numbers
        import re

        match = re.search(r"\b(19\d{2}|20\d{2})\b", date_string)
        if match:
            return int(match.group(1))
        return 0
    except:
        return 0


def perform_clustering(games_df, n_clusters=5):
    """Perform KMeans clustering on game data."""
    # Select numerical features for clustering
    features = [
        "price",
        "player_count",
        "metacritic_score",
        "recommendations",
        "is_free",
        "release_year",
    ]

    # Create feature matrix
    X = games_df[features].copy()

    # Handle missing values
    X.fillna(0, inplace=True)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA for dimensionality reduction and visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Add cluster information to the dataframe
    games_df["cluster"] = clusters

    # Create a DataFrame for the PCA results
    pca_df = pd.DataFrame(
        {
            "x": X_pca[:, 0],
            "y": X_pca[:, 1],
            "cluster": clusters,
            "name": games_df["name"],
        }
    )

    return games_df, pca_df, kmeans.cluster_centers_, features


def analyze_clusters(games_df, cluster_centers, feature_names):
    """Generate statistical analysis for each cluster."""
    cluster_stats = {}

    for cluster_id in games_df["cluster"].unique():
        cluster_games = games_df[games_df["cluster"] == cluster_id]

        # Get most common genres in this cluster
        all_genres = [
            genre for sublist in cluster_games["genres"].tolist() for genre in sublist
        ]
        top_genres = Counter(all_genres).most_common(3)

        stats = {
            "count": len(cluster_games),
            "avg_price": cluster_games["price"].mean(),
            "avg_player_count": cluster_games["player_count"].mean(),
            "avg_metacritic": cluster_games["metacritic_score"].mean(),
            "avg_recommendations": cluster_games["recommendations"].mean(),
            "free_games_pct": (cluster_games["is_free"].sum() / len(cluster_games))
            * 100,
            "avg_release_year": cluster_games["release_year"].mean(),
            "top_genres": top_genres,
            "example_games": cluster_games["name"].tolist()[
                :5
            ],  # 5 example games from this cluster
        }

        cluster_stats[cluster_id] = stats

    return cluster_stats


def main():
    print("===== STEAM API CLUSTERING ANALYSIS =====\n")

    # Number of games to analyze
    num_games = 30

    # Number of clusters to create
    num_clusters = 5

    # 1. Get popular games (we'll use a predefined list for demonstration)
    popular_game_ids = [
        570,  # Dota 2
        730,  # Counter-Strike 2
        440,  # Team Fortress 2
        292030,  # The Witcher 3
        1091500,  # Cyberpunk 2077
        578080,  # PUBG
        1086940,  # Baldur's Gate 3
        252490,  # Rust
        431960,  # Wallpaper Engine
        105600,  # Terraria
        230410,  # Warframe
        218620,  # PAYDAY 2
        1174180,  # Red Dead Redemption 2
        1097150,  # Fall Guys
        359550,  # Rainbow Six Siege
        346110,  # ARK
        238960,  # Path of Exile
        814380,  # Sekiro
        271590,  # GTA V
        311210,  # Call of Duty: Black Ops III
        550,  # Left 4 Dead 2
        1145360,  # Hades
        620,  # Portal 2
        1172470,  # Apex Legends
        648800,  # Raft
        1506830,  # FIFA 23
        289070,  # Civ VI
        552990,  # World of Warships
        1517290,  # Forza Horizon 5
        275850,  # No Man's Sky
        594570,  # Total War: WARHAMMER II
        1811260,  # EA Sports FC 24
        242760,  # The Forest
        391540,  # Undertale
        1158310,  # Crusader Kings III
    ]

    # 2. Fetch detailed data for these games
    print(f"Fetching data for {num_games} popular games...")
    games_data = fetch_game_data(popular_game_ids, max_games=num_games)

    if not games_data:
        print("Failed to collect game data. Exiting.")
        return

    # Convert to DataFrame
    games_df = pd.DataFrame(games_data)
    print(f"\nCollected data for {len(games_df)} games.")
    print("\nData sample:")
    print(games_df[["name", "price", "player_count", "metacritic_score"]].head())

    # 3. Perform clustering
    print("\nPerforming clustering analysis...")
    games_df, pca_df, cluster_centers, feature_names = perform_clustering(
        games_df, n_clusters=num_clusters
    )

    # 4. Analyze clusters
    print("\nAnalyzing clusters...")
    cluster_stats = analyze_clusters(games_df, cluster_centers, feature_names)

    # 5. Display results
    print("\n===== CLUSTERING RESULTS =====")
    for cluster_id, stats in cluster_stats.items():
        print(f"\nCluster #{cluster_id} ({stats['count']} games):")
        print(f"  Average Price: ${stats['avg_price']:.2f}")
        print(f"  Average Player Count: {stats['avg_player_count']:.0f}")
        print(f"  Average Metacritic Score: {stats['avg_metacritic']:.1f}")
        print(f"  Average Recommendations: {stats['avg_recommendations']:.0f}")
        print(f"  Free Games: {stats['free_games_pct']:.1f}%")
        print(f"  Average Release Year: {stats['avg_release_year']:.0f}")

        print("  Top Genres:", end=" ")
        if stats["top_genres"]:
            genres_str = ", ".join(
                [f"{genre[0]} ({genre[1]})" for genre in stats["top_genres"]]
            )
            print(genres_str)
        else:
            print("N/A")

        print("  Example Games:", end=" ")
        if stats["example_games"]:
            print(", ".join(stats["example_games"]))
        else:
            print("N/A")

    print("\n===== CLUSTERING COMPLETE =====")


if __name__ == "__main__":
    main()
