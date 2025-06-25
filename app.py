import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
import time

st.set_page_config(layout="wide", page_title="Crypto Pivot Analyzer")

def find_turning_points(df_close_prices):
    if len(df_close_prices) < 3:
        return []
    first_diff = np.diff(df_close_prices.values)
    signed_diff = np.sign(first_diff)
    diff_of_signed_diff = np.diff(signed_diff)
    turning_points = []
    for i in range(len(diff_of_signed_diff)):
        original_index = df_close_prices.index[i + 1]
        original_price = df_close_prices.iloc[i + 1]
        if diff_of_signed_diff[i] < 0:
            turning_points.append((original_index, original_price, 1))
        elif diff_of_signed_diff[i] > 0:
            turning_points.append((original_index, original_price, -1))
    filtered_turning_points = []
    if turning_points:
        filtered_turning_points.append(turning_points[0])
        for i in range(1, len(turning_points)):
            if turning_points[i][2] != filtered_turning_points[-1][2]:
                filtered_turning_points.append(turning_points[i])
            else:
                last_point = filtered_turning_points[-1]
                current_point = turning_points[i]
                if current_point[2] == 1:
                    if current_point[1] > last_point[1]:
                        filtered_turning_points[-1] = current_point
                elif current_point[2] == -1:
                    if current_point[1] < last_point[1]:
                        filtered_turning_points[-1] = current_point
    return filtered_turning_points

def get_predefined_symbols(exchange_id):
    """Get predefined symbol lists to avoid API calls that might be blocked"""
    symbols = {
        'binance': [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'SOL/USDT:USDT',
            'XRP/USDT:USDT', 'ADA/USDT:USDT', 'AVAX/USDT:USDT', 'DOT/USDT:USDT',
            'MATIC/USDT:USDT', 'LINK/USDT:USDT', 'UNI/USDT:USDT', 'LTC/USDT:USDT',
            'BCH/USDT:USDT', 'NEAR/USDT:USDT', 'ICP/USDT:USDT', 'FTM/USDT:USDT',
            'ATOM/USDT:USDT', 'XLM/USDT:USDT', 'ALGO/USDT:USDT', 'VET/USDT:USDT',
            'DOGE/USDT:USDT', 'SHIB/USDT:USDT', 'TRX/USDT:USDT', 'ETC/USDT:USDT'
        ],
        'bybit': [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'BNB/USDT:USDT',
            'XRP/USDT:USDT', 'ADA/USDT:USDT', 'AVAX/USDT:USDT', 'DOT/USDT:USDT',
            'MATIC/USDT:USDT', 'LINK/USDT:USDT', 'UNI/USDT:USDT', 'LTC/USDT:USDT',
            'BCH/USDT:USDT', 'NEAR/USDT:USDT', 'ICP/USDT:USDT', 'FTM/USDT:USDT',
            'ATOM/USDT:USDT', 'XLM/USDT:USDT', 'ALGO/USDT:USDT', 'VET/USDT:USDT',
            'DOGE/USDT:USDT', 'SHIB/USDT:USDT', 'TRX/USDT:USDT', 'ETC/USDT:USDT'
        ],
        'coinbasepro': [
            'BTC/USD:USD', 'ETH/USD:USD', 'SOL/USD:USD', 'XRP/USD:USD',
            'ADA/USD:USD', 'AVAX/USD:USD', 'DOT/USD:USD', 'MATIC/USD:USD',
            'LINK/USD:USD', 'UNI/USD:USD', 'LTC/USD:USD', 'BCH/USD:USD',
            'NEAR/USD:USD', 'ICP/USD:USD', 'FTM/USD:USD', 'ATOM/USD:USD'
        ]
    }
    return symbols.get(exchange_id, symbols['binance'])

@st.cache_data(ttl=3600)
def fetch_symbols_safe(exchange_id):
    """Safely fetch symbols with fallback to predefined lists"""
    try:
        # First try with minimal configuration
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'enableRateLimit': True,
            'timeout': 10000,  # Shorter timeout
            'headers': {
                'User-Agent': 'ccxt/python'
            }
        })
        
        # Add a small delay to be respectful
        time.sleep(0.5)
        
        markets = exchange.load_markets()
        future_symbols = []
        
        for market_id, market in markets.items():
            is_future = market.get('spot', False) is False and (
                market.get('linear', False) or market.get('inverse', False) or market.get('contract', False))
            is_usdt_usd_settled = market.get('settleId') in ['USDT', 'USD']
            if is_future and is_usdt_usd_settled and market['active']:
                future_symbols.append(market['symbol'])
        
        future_symbols.sort()
        
        if future_symbols:
            st.success(f"✅ Successfully loaded {len(future_symbols)} symbols from {exchange_id}")
            return future_symbols
        else:
            st.warning(f"⚠️ No futures symbols found for {exchange_id}, using predefined list")
            return get_predefined_symbols(exchange_id)
            
    except Exception as e:
        if "403" in str(e) or "Forbidden" in str(e):
            st.warning(f"🚫 {exchange_id.title()} API blocked from your location. Using predefined symbols.")
        else:
            st.warning(f"⚠️ Error fetching symbols from {exchange_id}: {str(e)[:100]}... Using predefined symbols.")
        
        return get_predefined_symbols(exchange_id)

@st.cache_data(ttl=300)
def fetch_ohlcv_safe(exchange_id, symbol, timeframe, limit):
    """Safely fetch OHLCV data with better error handling"""
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'enableRateLimit': True,
            'timeout': 20000,
            'headers': {
                'User-Agent': 'ccxt/python'
            }
        })
        
        # Add delay to be respectful to the API
        time.sleep(1)
        
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        if not ohlcv:
            return None, "No data returned from exchange"
        
        return ohlcv, None
        
    except ccxt.NetworkError as e:
        if "403" in str(e) or "Forbidden" in str(e):
            return None, f"🚫 {exchange_id.title()} blocked your location. Try a different exchange or deploy from another region."
        elif "429" in str(e):
            return None, f"⏰ Rate limited by {exchange_id}. Please wait and try again."
        else:
            return None, f"🌐 Network error: {str(e)[:100]}..."
    except ccxt.ExchangeError as e:
        return None, f"🏛️ Exchange error: {str(e)[:100]}..."
    except Exception as e:
        return None, f"❌ Unexpected error: {str(e)[:100]}..."

# Main UI
st.title("🚀 Crypto Pivot Analyzer")

# Alert box for geo-blocking issues
st.error("""
🚨 **IMPORTANT: Geographical Restrictions**
Many crypto exchanges block certain regions. If you see 403 errors:
1. **Try Binance first** (usually more accessible)
2. **Use longer timeframes** (4h, 1d work better)
3. **Deploy from US/EU regions** if possible
4. **Consider using a VPN-enabled hosting service**
""")

st.sidebar.header("⚙️ Configuration")

# Exchange selection with better ordering
exchange_options = {
    'binance': '🟡 Binance (Recommended)',
    'bybit': '🟠 Bybit (May be blocked)',
    'coinbasepro': '🔵 Coinbase Pro'
}

selected_display = st.sidebar.selectbox(
    "Select Exchange",
    list(exchange_options.values()),
    index=0,
    help="Binance typically has the fewest geographical restrictions"
)

# Get the actual exchange ID
selected_exchange_id = [k for k, v in exchange_options.items() if v == selected_display][0]

# Load symbols immediately (no caching issues)
with st.spinner(f"Loading symbols for {selected_exchange_id}..."):
    all_available_symbols = fetch_symbols_safe(selected_exchange_id)

# Symbol selection
if all_available_symbols:
    # Find default symbol
    default_symbol = 'BTC/USDT:USDT'
    if 'BTC/USD:USD' in all_available_symbols:
        default_symbol = 'BTC/USD:USD'
    
    try:
        default_index = all_available_symbols.index(default_symbol)
    except ValueError:
        default_index = 0
    
    symbol = st.sidebar.selectbox(
        "Select Trading Pair",
        options=all_available_symbols,
        index=default_index,
        help=f"Loaded {len(all_available_symbols)} symbols"
    )
else:
    symbol = st.sidebar.text_input(
        "Enter Symbol (e.g., BTC/USDT:USDT)", 
        "BTC/USDT:USDT",
        help="No symbols loaded - enter manually"
    )

# Timeframe selection
timeframe_options = {
    '1m': '1 Minute',
    '5m': '5 Minutes', 
    '15m': '15 Minutes',
    '1h': '1 Hour (Recommended)',
    '4h': '4 Hours (Recommended)',
    '1d': '1 Day (Most Reliable)'
}

selected_tf_display = st.sidebar.selectbox(
    "Select Timeframe",
    list(timeframe_options.values()),
    index=3,  # Default to 1h
    help="Longer timeframes are more reliable and less likely to be blocked"
)

timeframe = [k for k, v in timeframe_options.items() if v == selected_tf_display][0]

# Limit selection
limit = st.sidebar.slider(
    "Number of Candles", 
    min_value=50, 
    max_value=200, 
    value=100,
    help="Fewer candles = faster loading, less likely to timeout"
)

# Analysis parameters
st.sidebar.subheader("📊 Analysis Settings")
st.sidebar.info("Turning points are detected using price direction changes")

# Add status indicator
status_placeholder = st.sidebar.empty()

# Troubleshooting
with st.sidebar.expander("🔧 Troubleshooting 403 Errors"):
    st.markdown("""
    **If you get blocked (403 errors):**
    
    1. **Switch to Binance** - Usually works better
    2. **Use 4h or 1d timeframes** - More reliable
    3. **Reduce candle count** to 50-100
    4. **Try different symbols** (BTC/ETH usually work)
    5. **Deploy from US/EU** if possible
    6. **Wait 5-10 minutes** then retry
    
    **Alternative hosting solutions:**
    - Railway.app (multiple regions)
    - Render.com (US-based)
    - Fly.io (global regions)
    - Heroku (US/EU regions)
    """)

# Define timeframe duration mapping
timeframe_duration_map = {
    '1m': 1, '5m': 5, '15m': 15, 
    '1h': 60, '4h': 240, '1d': 1440
}

# Main analysis button
if st.sidebar.button("🚀 Fetch Data & Analyze", type="primary"):
    status_placeholder.info("🔄 Starting analysis...")
    
    # Create main content area
    st.subheader(f"📊 Analysis for {symbol} on {selected_exchange_id.title()}")
    
    # Fetch data
    with st.spinner(f"Fetching {limit} candles of {symbol} ({timeframe}) from {selected_exchange_id}..."):
        ohlcv_data, error_msg = fetch_ohlcv_safe(selected_exchange_id, symbol, timeframe, limit)
    
    if error_msg:
        st.error(f"**Data Fetch Failed:** {error_msg}")
        
        if "403" in error_msg or "blocked" in error_msg.lower():
            st.info("""
            **🔧 Quick Fixes:**
            1. Switch to **Binance** in the sidebar
            2. Try **4h or 1d timeframe**
            3. Use **BTC/USDT:USDT** symbol
            4. Deploy your app from a **US or EU region**
            """)
        
        status_placeholder.error("❌ Analysis failed")
        st.stop()
    
    if not ohlcv_data or len(ohlcv_data) == 0:
        st.error("No data received from the exchange")
        status_placeholder.error("❌ No data")
        st.stop()
    
    # Process data
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    st.success(f"✅ Successfully fetched {len(df)} candles from {selected_exchange_id.title()}")
    status_placeholder.success("✅ Data loaded")
    
    # Show recent data
    with st.expander("📋 Recent Price Data"):
        st.dataframe(df.tail(10))
    
    # Find turning points
    st.subheader("🔄 Turning Point Detection")
    turning_points = find_turning_points(df['Close'])
    
    if not turning_points:
        st.warning("⚠️ No turning points detected. Try:")
        st.markdown("- Increasing the number of candles")
        st.markdown("- Using a longer timeframe")
        st.markdown("- Selecting a more volatile trading pair")
        st.stop()
    
    st.success(f"🎯 Found {len(turning_points)} turning points")
    
    # Display turning points
    if turning_points:
        col1, col2 = st.columns(2)
        
        # Sort and filter points
        sorted_turning_points = sorted(turning_points, key=lambda x: x[0], reverse=True)
        recent_points = sorted_turning_points[:20]
        recent_points.sort(key=lambda x: x[0])
        
        minima = [p for p in recent_points if p[2] == -1]
        maxima = [p for p in recent_points if p[2] == 1]
        
        with col1:
            st.write("#### 🔴 Local Minima (Buy Signals)")
            if minima:
                minima_df = pd.DataFrame(minima, columns=['Timestamp', 'Price', 'Type'])
                minima_df['Timestamp'] = minima_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                minima_df['Price'] = minima_df['Price'].apply(lambda x: f"{x:.4f}")
                minima_df = minima_df.drop('Type', axis=1)
                st.dataframe(minima_df.set_index('Timestamp'))
            else:
                st.info("No minima in recent data")
        
        with col2:
            st.write("#### 🟢 Local Maxima (Sell Signals)")
            if maxima:
                maxima_df = pd.DataFrame(maxima, columns=['Timestamp', 'Price', 'Type'])
                maxima_df['Timestamp'] = maxima_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                maxima_df['Price'] = maxima_df['Price'].apply(lambda x: f"{x:.4f}")
                maxima_df = maxima_df.drop('Type', axis=1)
                st.dataframe(maxima_df.set_index('Timestamp'))
            else:
                st.info("No maxima in recent data")
    
    # Trading signals
    st.subheader("💰 Trading Signals")
    if turning_points:
        signals = []
        for point in sorted(turning_points, key=lambda x: x[0])[-10:]:  # Last 10 signals
            signal_type = "🟢 BUY" if point[2] == -1 else "🔴 SELL"
            signals.append({
                'Time': point[0].strftime('%Y-%m-%d %H:%M'),
                'Price': f"{point[1]:.4f}",
                'Signal': signal_type
            })
        
        if signals:
            signals_df = pd.DataFrame(signals)
            st.dataframe(signals_df.set_index('Time'))
    
    # Cycle analysis
    st.subheader("📈 Cycle Analysis")
    if len(turning_points) > 1:
        cycle_data = []
        current_timeframe_duration = timeframe_duration_map.get(timeframe, 1)
        
        for i in range(1, len(turning_points)):
            prev_point = turning_points[i-1]
            current_point = turning_points[i]
            
            time_diff = current_point[0] - prev_point[0]
            duration_minutes = time_diff.total_seconds() / 60
            duration_candles = duration_minutes / current_timeframe_duration
            
            price_change = current_point[1] - prev_point[1]
            price_change_pct = (price_change / prev_point[1]) * 100 if prev_point[1] != 0 else 0
            
            cycle_type = "📈 Min→Max" if prev_point[2] == -1 and current_point[2] == 1 else "📉 Max→Min"
            
            cycle_data.append({
                'Type': cycle_type,
                'Duration (Candles)': f"{duration_candles:.1f}",
                'Duration (Hours)': f"{duration_minutes/60:.1f}",
                'Price Change %': f"{price_change_pct:.2f}%"
            })
        
        if cycle_data:
            cycle_df = pd.DataFrame(cycle_data)
            st.dataframe(cycle_df)
            
            # Calculate averages for prediction
            numeric_cycles = pd.DataFrame([
                {
                    'duration_candles': float(c['Duration (Candles)']),
                    'duration_hours': float(c['Duration (Hours)']),
                    'price_change_pct': float(c['Price Change %'].replace('%', ''))
                }
                for c in cycle_data
            ])
            
            avg_duration = numeric_cycles['duration_hours'].mean()
            avg_price_change = abs(numeric_cycles['price_change_pct']).mean()
            
            # Prediction
            st.subheader("🔮 Next Turning Point Prediction")
            if turning_points:
                last_point = turning_points[-1]
                last_time = last_point[0]
                last_price = last_point[1]
                last_type = last_point[2]
                
                predicted_time = last_time + pd.Timedelta(hours=avg_duration)
                
                if last_type == -1:  # Last was minimum
                    predicted_price = last_price * (1 + avg_price_change/100)
                    predicted_type = "📈 Maximum (Sell Signal)"
                else:  # Last was maximum
                    predicted_price = last_price * (1 - avg_price_change/100)
                    predicted_type = "📉 Minimum (Buy Signal)"
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🕐 Predicted Time", predicted_time.strftime('%Y-%m-%d %H:%M'))
                with col2:
                    st.metric("💰 Predicted Price", f"{predicted_price:.4f}")
                with col3:
                    st.metric("📊 Signal Type", predicted_type)
    
    # Chart
    st.subheader("📊 Price Chart with Turning Points")
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot price
    ax.plot(df.index, df['Close'], label='Close Price', color='#1f77b4', linewidth=2)
    
    # Plot turning points
    if turning_points:
        min_points = [(p[0], p[1]) for p in turning_points if p[2] == -1]
        max_points = [(p[0], p[1]) for p in turning_points if p[2] == 1]
        
        if min_points:
            min_x, min_y = zip(*min_points)
            ax.scatter(min_x, min_y, color='red', s=100, marker='o', label='Buy Points (Minima)', zorder=5)
        
        if max_points:
            max_x, max_y = zip(*max_points)
            ax.scatter(max_x, max_y, color='green', s=100, marker='o', label='Sell Points (Maxima)', zorder=5)
    
    # Add prediction line if available
    if 'predicted_time' in locals() and 'predicted_price' in locals():
        ax.axvline(x=predicted_time, color='purple', linestyle='--', alpha=0.7, label='Predicted Next Pivot')
        ax.scatter([predicted_time], [predicted_price], color='purple', s=150, marker='X', label='Predicted Point', zorder=6)
    
    ax.set_title(f"{symbol} - {timeframe} - {selected_exchange_id.title()}", fontsize=16, fontweight='bold')
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Price", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    status_placeholder.success("✅ Analysis complete!")

# Footer
st.markdown("---")
st.markdown("""
### 💡 Pro Tips:
- **Binance** works best globally
- **4h/1d timeframes** are most reliable  
- **BTC/ETH pairs** have best data availability
- **Deploy from US/EU** for best API access
- **Reduce candles** if you get timeouts

### 🌍 Hosting Recommendations:
- **Railway.app** - Multiple regions, good for geo-blocked APIs
- **Render.com** - US-based, reliable for crypto APIs  
- **Fly.io** - Global deployment options
""")
