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

# --- Configuration ---
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('CHAT_ID')

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

# --- Helper Functions ---

def get_sp500_tickers():
    """Fetches S&P 500 tickers from GitHub CSV to avoid blocks."""
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        tickers = df['Symbol'].tolist()
        clean_tickers = [ticker.replace('.', '-') for ticker in tickers]
        print(f"Loaded {len(clean_tickers)} tickers.")
        return clean_tickers
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA']

def send_telegram_message(chat_id, message, token):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': message, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram Text Error: {e}")

def send_telegram_photo(chat_id, photo_stream, caption, token):
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    files = {'photo': ('chart.png', photo_stream, 'image/png')}
    payload = {'chat_id': chat_id, 'caption': caption, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, files=files, data=payload, timeout=15)
    except Exception as e:
        print(f"Telegram Photo Error: {e}")

def get_market_cap(ticker_obj):
    try:
        market_cap = ticker_obj.info.get('marketCap')
        if not market_cap: return "N/A"
        if market_cap >= 1_000_000_000: return f"{market_cap / 1_000_000_000:.1f}B"
        if market_cap >= 1_000_000: return f"{market_cap / 1_000_000:.1f}M"
        return str(market_cap)
    except: return "N/A"

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def plot_chart(df, ticker, trend, market_cap, atr_value, rsi_value):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Price Chart
    ax1.plot(df.index, df['Close'], color='#00ff00', label='Price', linewidth=1.5)
    ax1.plot(df.index, df['SMA_150'], color='cyan', label='SMA 150', linestyle='--', linewidth=1.5)
    ax1.fill_between(df.index, df['Close'], df['SMA_150'], where=(df['Close'] > df['SMA_150']), color='green', alpha=0.1)
    ax1.fill_between(df.index, df['Close'], df['SMA_150'], where=(df['Close'] < df['SMA_150']), color='red', alpha=0.1)
    ax1.set_title(f"{ticker} | Trend: {trend} | Cap: {market_cap}", fontsize=14, color='white', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # RSI Chart
    ax2.plot(df.index, df['RSI'], color='yellow', label='RSI(14)', linewidth=1.5)
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax2.set_title(f"RSI: {rsi_value:.1f}", fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle=':', alpha=0.6)

    fig.autofmt_xdate()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)
    return buf

# --- Core Logic ---

def analyze_stock(ticker):
    """Returns tuple (chart_img, message, rsi_value) if alert found, else None."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        df = yf.Ticker(ticker).history(start=start_date, end=end_date)
        if df.empty or len(df) < 150: return None

        df['SMA_150'] = df['Close'].rolling(window=150).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        
        # ATR
        df['High-Low'] = df['High'] - df['Low']
        df['High-PrevClose'] = abs(df['High'] - df['Close'].shift(1))
        df['Low-PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()

        df.dropna(subset=['SMA_150', 'ATR', 'RSI'], inplace=True)
        if df.empty: return None

        current_close = df['Close'].iloc[-1]
        current_sma = df['SMA_150'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1]
        current_atr = df['ATR'].iloc[-1]
        
        prev_sma = df['SMA_150'].iloc[-6]
        if current_sma > prev_sma * 1.005: trend = "Up 🟢"
        elif current_sma < prev_sma * 0.995: trend = "Down 🔴"
        else: trend = "Flat ⚪"

        dist_pct = abs(current_close - current_sma) / current_sma
        
        # === תנאי סינון ===
        # 1. מרחק עד 2.5% מהממוצע
        # 2. RSI נמוך מ-50 (כדי לא לקנות בשיא)
        if dist_pct <= 0.025 and current_rsi < 50:
            ticker_obj = yf.Ticker(ticker)
            market_cap = get_market_cap(ticker_obj)
            
            chart_img = plot_chart(df.tail(100), ticker, trend, market_cap, current_atr, current_rsi)
            tv_link = f"https://www.tradingview.com/chart/?symbol={ticker}"

            message = (
                f"🔔 *Alert:* `{ticker}`\n"
                f"Price: ${current_close:.2f}\n"
                f"SMA 150: ${current_sma:.2f} (Dist: {dist_pct*100:.1f}%)\n"
                f"RSI(14): {current_rsi:.1f}\n"
                f"Trend: {trend}\n"
                f"Cap: {market_cap}\n"
                f"ATR: {current_atr:.2f}\n"
                f"[View on TradingView]({tv_link})"
            )
            
            # מחזירים גם את ה-RSI (באינדקס 2) כדי שנוכל למיין לפיו אחר כך
            return (chart_img, message, current_rsi)
            
    except Exception:
        return None
    return None

# --- Main Execution ---

def run_scan():
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("Missing Tokens")
        return

    tickers = get_sp500_tickers()
    if not tickers: return

    print(f"Starting scan for {len(tickers)} tickers...")
    
    # רשימה לאגירת התוצאות
    valid_alerts = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(analyze_stock, tickers))

    # סינון תוצאות ריקות ושמירה לרשימה
    for res in results:
        if res:
            valid_alerts.append(res)

    # === כאן המיון מתבצע ===
    # מיון מהנמוך לגבוה לפי ה-RSI (שהוא האיבר השלישי בטאפל, אינדקס 2)
    valid_alerts.sort(key=lambda x: x[2])

    print(f"Found {len(valid_alerts)} alerts. Sending sorted by RSI...")

    # שליחת ההודעות לפי הסדר הממוין
    for alert in valid_alerts:
        chart_img, msg, rsi = alert
        send_telegram_photo(CHAT_ID, chart_img, msg, TELEGRAM_TOKEN)
        time.sleep(1.5) # השהייה קטנה כדי שטלגרם לא יחסום בגלל שליחה מהירה

    final_msg = f"✅ Scan finished. Sent {len(valid_alerts)} alerts (Sorted by RSI Low->High)."
    send_telegram_message(CHAT_ID, final_msg, TELEGRAM_TOKEN)
    print(final_msg)

if __name__ == "__main__":
    run_scan()
