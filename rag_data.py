import requests
import pandas as pd
import time
import os
from datetime import datetime

def get_data_by_date_range(coin_id, start_date, end_date, currency="usd", output_dir="crypto_data"):
    """
    Downloads price history from CoinGecko for a specific date range.
    Dates should be in 'YYYY-MM-DD' format.
    """
    # Convert date strings to UNIX timestamps (in seconds)
    from_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    to_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
    params = {
        "vs_currency": currency,
        "from": from_timestamp,
        "to": to_timestamp
    }
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://www.coingecko.com",
    "Origin": "https://www.coingecko.com"
}


    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 429:
            print(f"⏳ Rate limited for {coin_id}. Waiting 10 seconds...")
            time.sleep(10)
            response = requests.get(url, params=params, headers=headers)

        response.raise_for_status()
        data = response.json()

        if "prices" not in data:
            print(f"❌ No price data found for {coin_id} in range")
            return

        df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.sort_values("timestamp")

        os.makedirs(output_dir, exist_ok=True)
        filename = f"{coin_id}_price_2022.csv"
        df.to_csv(os.path.join(output_dir, filename), index=False)
        print(f"✅ Saved {coin_id} 2022 data to {filename} ({len(df)} rows)")

    except Exception as e:
        print(f"❌ Error for {coin_id}: {e}")

# Example: Get 2022 data for coins you successfully downloaded
successful_coins = ["bitcoin", "ethereum", "solana", "cardano", "ripple", "dogecoin"]

for coin in successful_coins:
    get_data_by_date_range(coin_id=coin, start_date="2022-01-01", end_date="2022-12-31")
    time.sleep(5)  # Delay to avoid rate limits
