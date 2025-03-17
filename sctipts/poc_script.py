import requests
import json
import time


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


def get_news_for_app(app_id, count=3):
    """Get news articles for a specific game."""
    url = f"https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/?appid={app_id}&count={count}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching news: {response.status_code}")
        return None


def main():
    # Popular game IDs to test with
    test_app_ids = [
        570,  # Dota 2
        730,  # Counter-Strike 2
        440,  # Team Fortress 2
        292030,  # The Witcher 3
        1091500,  # Cyberpunk 2077
    ]

    print("===== STEAM API TEST SCRIPT =====\n")

    # 1. Get and display a portion of the Steam app list
    print("1. Fetching Steam app list...")
    app_list = get_steam_applist()
    if app_list and "applist" in app_list and "apps" in app_list["applist"]:
        sample_apps = app_list["applist"]["apps"][:5]  # Just show first 5 apps
        print(f"Total apps found: {len(app_list['applist']['apps'])}")
        print("Sample apps:")
        for app in sample_apps:
            print(f"  - {app['name']} (ID: {app['appid']})")
    print("\n" + "-" * 50 + "\n")

    # 2. Get and display details for sample games
    print("2. Fetching details for sample games:")
    for app_id in test_app_ids[:2]:  # Just test first 2 to avoid rate limiting
        print(f"\nGetting details for app ID: {app_id}")
        details = get_game_details(app_id)
        if details and str(app_id) in details and details[str(app_id)]["success"]:
            game_data = details[str(app_id)]["data"]
            print(f"  - Name: {game_data.get('name', 'N/A')}")
            print(f"  - Type: {game_data.get('type', 'N/A')}")
            print(
                f"  - Description: {game_data.get('short_description', 'N/A')[:100]}..."
            )
            print(f"  - Developers: {', '.join(game_data.get('developers', ['N/A']))}")
            print(
                f"  - Price: {game_data.get('price_overview', {}).get('final_formatted', 'N/A')}"
            )
    print("\n" + "-" * 50 + "\n")

    # 3. Get and display current player counts
    print("3. Fetching current player counts:")
    for app_id in test_app_ids[:3]:  # Test first 3 games
        player_data = get_player_count(app_id)
        if (
            player_data
            and "response" in player_data
            and "player_count" in player_data["response"]
        ):
            print(
                f"  - App ID {app_id}: {player_data['response']['player_count']} players currently online"
            )
    print("\n" + "-" * 50 + "\n")

    # 4. Get and display news for a game
    print("4. Fetching news for Dota 2 (ID: 570):")
    news_data = get_news_for_app(570, 3)
    if news_data and "appnews" in news_data and "newsitems" in news_data["appnews"]:
        news_items = news_data["appnews"]["newsitems"]
        for i, item in enumerate(news_items, 1):
            print(f"\nNews #{i}:")
            print(f"  - Title: {item.get('title', 'N/A')}")
            print(f"  - URL: {item.get('url', 'N/A')}")
            print(
                f"  - Date: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(item.get('date', 0)))}"
            )
            content = item.get("contents", "N/A")
            print(
                f"  - Content Preview: {content[:100]}..."
                if len(content) > 100
                else f"  - Content: {content}"
            )

    print("\n===== TEST COMPLETE =====")


if __name__ == "__main__":
    main()

