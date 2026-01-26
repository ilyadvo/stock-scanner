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
import concurrent.futures # ספרייה לעבודה במקביל

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
        return ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA'] # Fallback

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
    """Calculates the Relative Strength Index (RSI)."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def plot_chart(df, ticker, trend, market_cap, atr_value, rsi_value):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Main Price Chart
    ax1.plot(df.index, df['Close'], color='#00ff00', label='Price', linewidth=1.5)
    ax1.plot(df.index, df['SMA_150'], color='cyan', label='SMA 150', linestyle='--', linewidth=1.5)
    ax1.fill_between(df.index, df['Close'], df['SMA_150'], where=(df['Close'] > df['SMA_150']), color='green', alpha=0.1)
    ax1.fill_between(df.index, df['Close'], df['SMA_150'], where=(df['Close'] < df['SMA_150']), color='red', alpha=0.1)
    ax1.set_title(f"{ticker} | Trend: {trend} | Cap: {market_cap}", fontsize=14, color='white', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # RSI Chart
    ax2.plot(df.index, df['RSI'], color='yellow', label='RSI(14)', linewidth=1.5)
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5) # Overbought
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5) # Oversold
    ax2.set_title(f"RSI: {rsi_value:.1f}", fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle=':', alpha=0.6)

    fig.autofmt_xdate()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)
    return buf

# --- Core Logic per Stock ---

def analyze_stock(ticker):
    """Downloads and analyzes a single stock. Returns alert data or None."""
    try:
        # Download 2 years of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        df = yf.Ticker(ticker).history(start=start_date, end=end_date)
        if df.empty or len(df) < 150: return None

        # Indicators
        df['SMA_150'] = df['Close'].rolling(window=150).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        
        # ATR Calculation
        df['High-Low'] = df['High'] - df['Low']
        df['High-PrevClose'] = abs(df['High'] - df['Close'].shift(1))
        df['Low-PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()

        df.dropna(subset=['SMA_150', 'ATR', 'RSI'], inplace=True)
        if df.empty: return None

        # Current Values
        current_close = df['Close'].iloc[-1]
        current_sma = df['SMA_150'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1]
        current_atr = df['ATR'].iloc[-1]
        
        # Trend Detection
        prev_sma = df['SMA_150'].iloc[-6]
        if current_sma > prev_sma * 1.005: trend = "Up 🟢"
        elif current_sma < prev_sma * 0.995: trend = "Down 🔴"
        else: trend = "Flat ⚪"

       # Check Condition: Distance <= 2.5% AND RSI Filtering
        dist_pct = abs(current_close - current_sma) / current_sma
        
        # === כאן השינוי ===
        # אנו דורשים שני תנאים:
        # 1. המרחק מהממוצע הוא עד 2.5%
        # 2. ה-RSI נמוך מ-50 (כדי לוודא שאנחנו לא קונים בשיא)
        if dist_pct <= 0.025 and current_rsi < 50:
            
            ticker_obj = yf.Ticker(ticker)
            market_cap = get_market_cap(ticker_obj)
            
            # Create Plot
            chart_img = plot_chart(df.tail(100), ticker, trend, market_cap, current_atr, current_rsi)
            
            # ... המשך הקוד רגיל ...
            tv_link = f"https://www.tradingview.com/chart/?symbol={ticker}"
            
            message = (
                f"🔔 *Alert:* `{ticker}`\n"
                f"Price: ${current_close:.2f}\n"
                f"SMA 150: ${current_sma:.2f} (Dist: {dist_pct*100:.1f}%)\n"
                f"RSI(14): {current_rsi:.1f}\n" # ה-RSI כבר מחושב ומוכן
                f"Trend: {trend}\n"
                f"Cap: {market_cap}\n"
                f"ATR: {current_atr:.2f}\n"
                f"[View on TradingView]({tv_link})"
            )
            
            return (chart_img, message)
            
    except Exception as e:
        return None
    return None

# --- Main Execution ---

def run_scan():
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("Missing Tokens")
        return

    tickers = get_sp500_tickers()
    if not tickers:
        send_telegram_message(CHAT_ID, "❌ Failed to fetch list.", TELEGRAM_TOKEN)
        return

    print(f"Starting parallel scan for {len(tickers)} tickers...")
    alerts_found = 0

    # Parallel Processing using ThreadPoolExecutor
    # This runs 20 downloads simultaneously instead of 1 by 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(analyze_stock, tickers))

    # Process results
    for result in results:
        if result:
            alerts_found += 1
            chart_img, msg = result
            send_telegram_photo(CHAT_ID, chart_img, msg, TELEGRAM_TOKEN)
            time.sleep(1) # Prevent Telegram flooding

    final_msg = f"✅ Scan finished. Found {alerts_found} opportunities."
    send_telegram_message(CHAT_ID, final_msg, TELEGRAM_TOKEN)
    print(final_msg)

if __name__ == "__main__":
    run_scan()
