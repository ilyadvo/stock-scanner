import os
import requests
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import numpy as np
from datetime import datetime, timedelta
import time
import concurrent.futures
import json

# --- Configuration ---
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('CHAT_ID')

# --- Matplotlib Dark Mode Setup ---
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1a1a1a'
plt.rcParams['axes.facecolor'] = '#1a1a1a'
plt.rcParams['text.color'] = 'lightgrey'

# --- Helper Functions ---

def get_sp500_tickers():
    """Fetches tickers from a stable CSV source."""
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        return [t.replace('.', '-') for t in df['Symbol'].tolist()]
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return ['AAPL', 'MSFT', 'NVDA']

def send_telegram_media_group(chat_id, media_list, token):
    """Sends a group of photos (up to 10) with individual captions."""
    url = f"https://api.telegram.org/bot{token}/sendMediaGroup"
    files = {}
    media = []
    
    for i, (photo_stream, caption) in enumerate(media_list):
        file_id = f"photo{i}"
        media.append({
            'type': 'photo',
            'media': f'attach://{file_id}',
            'caption': caption,
            'parse_mode': 'Markdown'
        })
        files[file_id] = photo_stream

    payload = {'chat_id': chat_id, 'media': json.dumps(media)}
    try:
        return requests.post(url, data=payload, files=files, timeout=30)
    except Exception as e:
        print(f"Error sending Media Group: {e}")
        return None

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def plot_chart(df, ticker, trend):
    """Creates a combined Price and RSI chart."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})
    
    # Price Axis
    ax1.plot(df.index, df['Close'], color='#00ff00', linewidth=1.5, label='Price')
    ax1.plot(df.index, df['SMA_150'], color='cyan', linestyle='--', linewidth=1.2, label='SMA 150')
    ax1.fill_between(df.index, df['Close'], df['SMA_150'], color='cyan', alpha=0.1)
    ax1.set_title(f"{ticker} | {trend}", color='white', fontweight='bold', fontsize=14)
    ax1.grid(True, linestyle=':', alpha=0.3)
    
    # RSI Axis
    ax2.plot(df.index, df['RSI'], color='yellow', linewidth=1)
    ax2.axhline(30, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle=':', alpha=0.3)
    
    fig.autofmt_xdate()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=90)
    buf.seek(0)
    plt.close(fig)
    return buf

# --- Core Analysis ---

def analyze_stock(ticker):
    """Processes a single stock and returns chart + caption data."""
    try:
        df = yf.Ticker(ticker).history(period="2y")
        if len(df) < 150: return None
        
        df['SMA_150'] = df['Close'].rolling(window=150).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        
        curr = df.iloc[-1]
        prev_sma = df['SMA_150'].iloc[-6]
        dist = abs(curr['Close'] - curr['SMA_150']) / curr['SMA_150']
        
        # סינון: מרחק < 2.5% ו-RSI < 50
        if dist <= 0.025 and curr['RSI'] < 50:
            trend = "Rising 🟢" if curr['SMA_150'] > prev_sma * 1.005 else "Falling 🔴" if curr['SMA_150'] < prev_sma * 0.995 else "Flat ⚪"
            
            # יצירת הגרף
            chart = plot_chart(df.tail(80), ticker, trend)
            
            # יצירת הכיתוב שיוצמד לתמונה
            tv_link = f"https://www.tradingview.com/chart/?symbol={ticker}"
            caption = (
                f"🎯 `{ticker}` (Click to Copy)\n"
                f"Price: ${curr['Close']:.2f} | RSI: {curr['RSI']:.1f}\n"
                f"Dist from SMA: {dist*100:.1f}%\n"
                f"Trend: {trend}\n"
                f"[Open in TradingView]({tv_link})"
            )
            
            return (chart, caption, curr['RSI'])
    except: return None
    return None

# --- Main Logic ---

def run_scan():
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    
    tickers = get_sp500_tickers()
    print(f"Scanning {len(tickers)} tickers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # אוספים את כל התוצאות שעברו את הסינון
        results = [r for r in list(executor.map(analyze_stock, tickers)) if r]

    if not results:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={'chat_id': CHAT_ID, 'text': "➖ No stocks met the criteria today."}, timeout=10)
        return

    # מיון התוצאות לפי RSI מהנמוך לגבוה
    results.sort(key=lambda x: x[2])

    # שליחה בקבוצות (Albums) של עד 10 מניות
    # כל תמונה באלבום תכיל את הטקסט הספציפי שלה
    for i in range(0, len(results), 10):
        batch = results[i:i+10]
        # מכינים רשימת (גרף, כיתוב) עבור הפונקציה
        media_group = [(item[0], item[1]) for item in batch]
        send_telegram_media_group(CHAT_ID, media_group, TELEGRAM_TOKEN)
        time.sleep(2) # השהייה קלה למניעת חסימה

    print(f"Done! Sent {len(results)} alerts.")

if __name__ == "__main__":
    run_scan()
