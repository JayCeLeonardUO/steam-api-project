import requests
import time
import csv


def get_steam_popular_games(api_key, count=10000):
    """
    Retrieve the IDs for the most popular games on Steam.

    Args:
        api_key (str): Your Steam Web API key
        count (int): Number of popular games to retrieve (default: 10000)

    Returns:
        list: List of dictionaries containing game IDs and names
    """
    # Base URL for the Steam API
    base_url = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"

    # Get the full list of apps from Steam
    response = requests.get(base_url)

    if response.status_code != 200:
        print(f"Error fetching app list: {response.status_code}")
        return []

    # Parse the response JSON
    data = response.json()

    # Extract all apps
    all_apps = data.get("applist", {}).get("apps", [])
    print(f"Total apps found: {len(all_apps)}")

    # Since the basic API doesn't provide popularity metrics directly,
    # we'll need to get additional data for each app

    popular_games = []
    progress_count = 0

    # To get popularity metrics, we'll use the store API to get details
    # about each app and check player counts, reviews, etc.
    store_api_url = "https://store.steampowered.com/api/appdetails"

    # For getting player counts
    player_count_url = (
        f"https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/"
    )

    # Process apps (this will take time for 10,000 games)
    for app in all_apps:
        app_id = app.get("appid")
        name = app.get("name")

        # Skip non-game apps or apps with no name
        if not name or len(name) == 0:
            continue

        # Get current player count
        params = {"appid": app_id, "key": api_key}
        try:
            player_response = requests.get(player_count_url, params=params)

            if player_response.status_code == 200:
                player_data = player_response.json()
                player_count = player_data.get("response", {}).get("player_count", 0)

                # Add to our list with player count
                popular_games.append(
                    {"appid": app_id, "name": name, "player_count": player_count}
                )

                # Print progress
                progress_count += 1
                if progress_count % 100 == 0:
                    print(f"Processed {progress_count} apps...")

                # Respect rate limits
                time.sleep(0.2)  # 5 requests per second max

                # If we have enough games, stop
                if len(popular_games) >= count:
                    break

        except Exception as e:
            print(f"Error processing app {app_id}: {str(e)}")
            continue

    # Sort by player count (descending)
    popular_games.sort(key=lambda x: x.get("player_count", 0), reverse=True)

    # Return the top games up to the requested count
    return popular_games[:count]


def save_to_csv(games, filename="top_steam_games.csv"):
    """Save the list of games to a CSV file"""
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["appid", "name", "player_count"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for game in games:
            writer.writerow(game)
    print(f"Saved {len(games)} games to {filename}")


def main():
    # Replace with your actual Steam API key
    api_key = "YOUR_STEAM_API_KEY"

    # Get top 10,000 games (or fewer if not enough games with players)
    print("Fetching popular Steam games...")
    popular_games = get_steam_popular_games(api_key)

    # Print the top 10 games
    print("\nTop 10 most popular games:")
    for i, game in enumerate(popular_games[:10]):
        print(
            f"{i + 1}. {game['name']} (App ID: {game['appid']}) - {game['player_count']} players"
        )

    # Save all games to CSV
    save_to_csv(popular_games)

    # Return just the app IDs if needed
    game_ids = [game["appid"] for game in popular_games]
    return game_ids


if __name__ == "__main__":
    main()
