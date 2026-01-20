import os
import requests
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from bs4 import BeautifulSoup
import io
import numpy as np
from datetime import datetime, timedelta
import time

# --- Configuration ---
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('CHAT_ID')
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# --- Matplotlib Dark Mode Setup ---
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1a1a1a'
plt.rcParams['axes.facecolor'] = '#1a1a1a'
plt.rcParams['text.color'] = 'lightgrey'
plt.rcParams['axes.labelcolor'] = 'lightgrey'
plt.rcParams['xtick.color'] = 'lightgrey'
plt.rcParams['ytick.color'] = 'lightgrey'
plt.rcParams['grid.color'] = '#333333'
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['figure.titlesize'] = 'large'
plt.rcParams['axes.titlesize'] = 'medium'

# --- Helper Functions ---

def get_sp500_tickers():
    """Fetches the S&P 500 ticker list from Wikipedia."""
    try:
        # We add a User-Agent header so Wikipedia treats us like a browser, not a bot
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(SP500_WIKI_URL, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable sortable'})
        tickers = []
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text.strip()
            tickers.append(ticker.replace('.', '-')) # Handle BRK.B -> BRK-B
        return tickers
    except requests.exceptions.RequestException as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        return []
    except Exception as e:
        print(f"Error parsing S&P 500 tickers: {e}")
        return []

def send_telegram_message(chat_id, message, token):
    """Sends a text message to a specified Telegram chat."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown'
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending Telegram message: {e}")
        return None

def send_telegram_photo(chat_id, photo_stream, caption, token):
    """Sends a photo with a caption to a specified Telegram chat."""
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    files = {'photo': ('chart.png', photo_stream, 'image/png')}
    payload = {
        'chat_id': chat_id,
        'caption': caption,
        'parse_mode': 'Markdown'
    }
    try:
        response = requests.post(url, files=files, data=payload, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending Telegram photo: {e}")
        return None

def get_market_cap(ticker_obj):
    """Fetches and formats market cap for a given ticker."""
    try:
        market_cap = ticker_obj.info.get('marketCap')
        if market_cap is None:
            return "N/A"
        if market_cap >= 1_000_000_000:
            return f"{market_cap / 1_000_000_000:.2f}B"
        elif market_cap >= 1_000_000:
            return f"{market_cap / 1_000_000:.2f}M"
        return str(market_cap)
    except Exception:
        return "N/A"

def plot_chart(df, ticker, trend, market_cap, atr_value):
    """Generates a chart and saves it to a BytesIO object."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot Close Price
    ax.plot(df.index, df['Close'], color='#00ff00', label='Price', linewidth=1.5)
    
    # Plot SMA 150
    ax.plot(df.index, df['SMA_150'], color='cyan', label='SMA 150', linestyle='--', linewidth=1.5)

    # Fill area between price and SMA
    ax.fill_between(df.index, df['Close'], df['SMA_150'], where=(df['Close'] > df['SMA_150']), color='green', alpha=0.1)
    ax.fill_between(df.index, df['Close'], df['SMA_150'], where=(df['Close'] < df['SMA_150']), color='red', alpha=0.1)

    # Styling
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=10)) 
    ax.grid(True, linestyle=':', alpha=0.6)
    fig.autofmt_xdate()

    ax.set_title(f"{ticker} | Trend: {trend} | Cap: {market_cap} | ATR: {atr_value:.2f}", fontsize=14, color='white', fontweight='bold')
    ax.set_ylabel("Price ($)")
    ax.legend(loc='upper left')

    # Save to BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig) 
    return buf

# --- Main Logic ---

def run_scan():
    """Main function to perform the stock scan and send alerts."""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("TELEGRAM_TOKEN or CHAT_ID environment variables are not set.")
        return

    tickers = get_sp500_tickers()
    if not tickers:
        send_telegram_message(CHAT_ID, "❌ Failed to fetch S&P 500 tickers.", TELEGRAM_TOKEN)
        return

    alerts_count = 0
    # Process only first 50 for testing if needed, remove slice for full scan
    # tickers = tickers[:50] 
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2 * 365) 

    print(f"Starting scan for {len(tickers)} tickers...")

    for i, ticker in enumerate(tickers):
        # Progress indicator every 10 stocks
        if i % 10 == 0:
            print(f"Scanning... {i}/{len(tickers)}")
            
        try:
            # Fetch data
            df = yf.Ticker(ticker).history(start=start_date, end=end_date)
            if df.empty or len(df) < 150:
                continue

            # Calculate indicators
            df['SMA_150'] = df['Close'].rolling(window=150).mean()
            
            # ATR Calculation
            df['High-Low'] = df['High'] - df['Low']
            df['High-PrevClose'] = abs(df['High'] - df['Close'].shift(1))
            df['Low-PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
            df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
            df['ATR'] = df['TR'].rolling(window=14).mean()

            # Cleanup
            df.dropna(subset=['SMA_150', 'ATR'], inplace=True)
            if df.empty:
                continue

            # Logic Checks
            current_close = df['Close'].iloc[-1]
            current_sma_150 = df['SMA_150'].iloc[-1]
            prev_sma_150 = df['SMA_150'].iloc[-6] # 5 days ago
            current_atr = df['ATR'].iloc[-1]

            # Determine Trend
            if current_sma_150 > prev_sma_150 * 1.005:
                trend = "Up 🟢"
            elif current_sma_150 < prev_sma_150 * 0.995:
                trend = "Down 🔴"
            else:
                trend = "Flat ⚪"

            # Alert Condition: Price within 5% of SMA 150
            dist_pct = abs(current_close - current_sma_150) / current_sma_150
            if dist_pct <= 0.05:
                alerts_count += 1
                ticker_obj = yf.Ticker(ticker)
                market_cap = get_market_cap(ticker_obj)
                
                alert_message = (
                    f"🔔 *Stock Alert: {ticker}*\n"
                    f"Price: ${current_close:.2f}\n"
                    f"SMA 150: ${current_sma_150:.2f} (Dist: {dist_pct*100:.1f}%)\n"
                    f"Trend: {trend}\n"
                    f"Cap: {market_cap}\n"
                    f"ATR: {current_atr:.2f}"
                )
                
                # Plot and send
                chart_stream = plot_chart(df.tail(100), ticker, trend, market_cap, current_atr)
                send_telegram_photo(CHAT_ID, chart_stream, alert_message, TELEGRAM_TOKEN)
                time.sleep(1) # Respect API limits

        except Exception as e:
            print(f"Skipping {ticker}: {e}")
            continue

    final_message = f"✅ Scan finished. Found {alerts_count} alerts."
    send_telegram_message(CHAT_ID, final_message, TELEGRAM_TOKEN)
    print(final_message)

if __name__ == "__main__":
    run_scan()
