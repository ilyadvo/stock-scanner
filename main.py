import os
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import concurrent.futures

# --- Configuration ---
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('CHAT_ID')

# רשימת המניות להדגשה
BIG_TECH = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AVGO']

# --- Helper Functions ---

def get_sp500_tickers():
    """Fetches tickers from a stable CSV source."""
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        return [t.replace('.', '-') for t in df['Symbol'].tolist()]
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return BIG_TECH # במקרה של שגיאה, נסרוק לפחות את הגדולות

def send_telegram_message(chat_id, message, token):
    """Sends a single text message to Telegram."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        'chat_id': chat_id, 
        'text': message, 
        'parse_mode': 'Markdown',
        'disable_web_page_preview': True
    }
    try:
        return requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Error sending message: {e}")
        return None

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def check_cup_and_handle(df):
    """
    מתודה היוריסטית לזיהוי תבנית ספל וידית.
    דורשת לפחות 120 ימי מסחר.
    """
    if len(df) < 120:
        return False

    try:
        left_part = df['High'].iloc[-120:-60]   
        middle_part = df['Low'].iloc[-80:-20]   
        right_part = df['High'].iloc[-30:-5]    
        
        left_lip_max = left_part.max()
        cup_bottom = middle_part.min()
        right_lip_max = right_part.max()
        current_price = df['Close'].iloc[-1]

        cup_depth = (left_lip_max - cup_bottom) / left_lip_max
        if not (0.10 <= cup_depth <= 0.40):
            return False

        if abs(right_lip_max - left_lip_max) / left_lip_max > 0.10:
            return False

        handle_pullback = (right_lip_max - current_price) / right_lip_max
        if not (0.02 <= handle_pullback <= 0.15):
            return False
            
        if current_price <= cup_bottom + (left_lip_max - cup_bottom) * 0.5:
            return False

        return True
    except:
        return False

# --- Core Analysis ---

def analyze_stock(ticker):
    """Processes a single stock and returns text data if criteria met."""
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
            tv_link = f"https://www.tradingview.com/symbols/{ticker}/"
            
            # בדיקת תבנית ספל וידית
            is_cnh = check_cup_and_handle(df)
            cnh_flag = " | *☕ C&H Pattern*" if is_cnh else ""
            
            # הדגשת מניות ספציפיות
            if ticker in BIG_TECH:
                prefix = "⭐️ 👑"
                bold_ticker = f"*{ticker}*"
            else:
                prefix = "🎯"
                bold_ticker = f"`{ticker}`"
            
            line = (
                f"{prefix} {bold_ticker} | Price: ${curr['Close']:.2f} | "
                f"RSI: {curr['RSI']:.1f} | Dist: {dist*100:.1f}% | "
                f"Trend: {trend}{cnh_flag} | [TV Link]({tv_link})"
            )
            
            return {"rsi": curr['RSI'], "line": line}
    except: 
        return None
    return None

# --- Main Logic ---

def run_scan():
    if not TELEGRAM_TOKEN or not CHAT_ID: 
        print("Missing Tokens")
        return
    
    tickers = get_sp500_tickers()
    print(f"Starting text-only scan for {len(tickers)} tickers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        results = [r for r in list(executor.map(analyze_stock, tickers)) if r]

    if not results:
        send_telegram_message(CHAT_ID, "➖ לא נמצאו מניות העונות לקריטריונים היום.", TELEGRAM_TOKEN)
        return

    # מיון התוצאות לפי RSI מהנמוך לגבוה
    results.sort(key=lambda x: x['rsi'])

    message_lines = [
        "📊 *דו\"ח סריקה יומי - S&P 500*",
        f"נמצאו {len(results)} הזדמנויות קרובות ל-SMA 150 (RSI < 50):",
        "_ממוין מ-RSI נמוך לגבוה_",
        ""
    ]
    
    for r in results:
        message_lines.append(r['line'])

    full_message = "\n".join(message_lines)

    if len(full_message) > 4000:
        for i in range(0, len(message_lines), 30):
            chunk = "\n".join(message_lines[i:i+30])
            send_telegram_message(CHAT_ID, chunk, TELEGRAM_TOKEN)
            time.sleep(1)
    else:
        send_telegram_message(CHAT_ID, full_message, TELEGRAM_TOKEN)

    print(f"Done! Sent report with {len(results)} stocks.")

if __name__ == "__main__":
    run_scan()
