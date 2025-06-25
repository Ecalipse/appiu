import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta

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

@st.cache_data(ttl=3600)
def fetch_all_future_symbols(exchange_id):
    try:
        # Create exchange with sandbox mode and different configuration
        exchange_config = {
            'enableRateLimit': True,
            'timeout': 30000,
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        }
        
        # Add proxy configuration if needed (uncomment and configure if you have a proxy)
        # exchange_config['proxies'] = {
        #     'http': 'your-proxy-url',
        #     'https': 'your-proxy-url'
        # }
        
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class(exchange_config)
        
        # Try to load markets with error handling
        try:
            markets = exchange.load_markets()
        except ccxt.NetworkError as e:
            if "403" in str(e) or "Forbidden" in str(e):
                st.warning(f"⚠️ {exchange_id.title()} is blocking requests from your location. Trying fallback method...")
                return get_fallback_symbols(exchange_id)
            else:
                raise e
        
        future_symbols = []
        for market_id, market in markets.items():
            is_future = market.get('spot', False) is False and (
                market.get('linear', False) or market.get('inverse', False) or market.get('contract', False))
            is_usdt_usd_settled = market.get('settleId') in ['USDT', 'USD']
            if is_future and is_usdt_usd_settled and market['active']:
                future_symbols.append(market['symbol'])
        future_symbols.sort()
        return future_symbols
    except ccxt.NetworkError as e:
        if "403" in str(e) or "Forbidden" in str(e):
            st.warning(f"⚠️ {exchange_id.title()} is blocking requests from your location. Using fallback symbols...")
            return get_fallback_symbols(exchange_id)
        else:
            st.error(f"Network error fetching symbols from {exchange_id}: {e}")
            return get_fallback_symbols(exchange_id)
    except ccxt.ExchangeError as e:
        st.error(f"Exchange error fetching symbols from {exchange_id}: {e}")
        return get_fallback_symbols(exchange_id)
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching symbols: {e}")
        return get_fallback_symbols(exchange_id)

def get_fallback_symbols(exchange_id):
    """Return common futures symbols when API is blocked"""
    common_symbols = {
        'bybit': [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'BNB/USDT:USDT',
            'XRP/USDT:USDT', 'ADA/USDT:USDT', 'AVAX/USDT:USDT', 'DOT/USDT:USDT',
            'MATIC/USDT:USDT', 'LINK/USDT:USDT', 'UNI/USDT:USDT', 'LTC/USDT:USDT',
            'BCH/USDT:USDT', 'NEAR/USDT:USDT', 'ICP/USDT:USDT', 'FTM/USDT:USDT',
            'ATOM/USDT:USDT', 'XLM/USDT:USDT', 'ALGO/USDT:USDT', 'VET/USDT:USDT'
        ],
        'binance': [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'SOL/USDT:USDT',
            'XRP/USDT:USDT', 'ADA/USDT:USDT', 'AVAX/USDT:USDT', 'DOT/USDT:USDT',
            'MATIC/USDT:USDT', 'LINK/USDT:USDT', 'UNI/USDT:USDT', 'LTC/USDT:USDT',
            'BCH/USDT:USDT', 'NEAR/USDT:USDT', 'ICP/USDT:USDT', 'FTM/USDT:USDT',
            'ATOM/USDT:USDT', 'XLM/USDT:USDT', 'ALGO/USDT:USDT', 'VET/USDT:USDT'
        ],
        'coinbasepro': [
            'BTC/USD:USD', 'ETH/USD:USD', 'SOL/USD:USD', 'XRP/USD:USD',
            'ADA/USD:USD', 'AVAX/USD:USD', 'DOT/USD:USD', 'MATIC/USD:USD',
            'LINK/USD:USD', 'UNI/USD:USD', 'LTC/USD:USD', 'BCH/USD:USD',
            'NEAR/USD:USD', 'ICP/USD:USD', 'FTM/USD:USD', 'ATOM/USD:USD'
        ]
    }
    return common_symbols.get(exchange_id, ['BTC/USDT:USDT', 'ETH/USDT:USDT'])

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_ohlcv_data(exchange_id, symbol, timeframe, limit):
    """Fetch OHLCV data with error handling for geo-blocking"""
    try:
        exchange_config = {
            'enableRateLimit': True,
            'timeout': 30000,
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        }
        
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class(exchange_config)
        
        # Try to fetch data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        return ohlcv, None
        
    except ccxt.NetworkError as e:
        if "403" in str(e) or "Forbidden" in str(e):
            error_msg = f"🚫 {exchange_id.title()} is blocking requests from your deployment location due to geographical restrictions."
            return None, error_msg
        else:
            error_msg = f"Network error: {e}"
            return None, error_msg
    except ccxt.ExchangeError as e:
        error_msg = f"Exchange error: {e}"
        return None, error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        return None, error_msg

st.title("🚀 Crypto Pivot Analyzer")

# Add information about geo-blocking
st.info("""
**Note:** Some exchanges may block requests from certain locations due to geographical restrictions. 
If you encounter 403 errors, try:
1. Using a different exchange (Binance usually has fewer restrictions)
2. Deploying your app from a different region
3. Using a VPN-enabled hosting service
""")

st.sidebar.header("Configuration")

# Exchange selection with better error handling
selected_exchange_id = st.sidebar.selectbox(
    "Select Exchange",
    ('binance', 'bybit', 'coinbasepro'),  # Put binance first as it's more reliable
    index=0,
    key='exchange_selected',
    help="Binance typically has fewer geographical restrictions"
)

# Get symbols with fallback
all_available_symbols = fetch_all_future_symbols(selected_exchange_id)

# Symbol selection
default_symbol_index = 0
if 'BTC/USDT:USDT' in all_available_symbols:
    default_symbol_index = all_available_symbols.index('BTC/USDT:USDT')
elif 'BTC/USD:USD' in all_available_symbols:
    default_symbol_index = all_available_symbols.index('BTC/USD:USD')
elif all_available_symbols:
    default_symbol_index = 0
else:
    default_symbol_index = None

if all_available_symbols:
    symbol = st.sidebar.selectbox(
        "Select Symbol",
        options=all_available_symbols,
        index=default_symbol_index if default_symbol_index is not None else 0,
        key='symbol_selected'
    )
else:
    symbol = st.sidebar.text_input(
        "Enter Symbol (e.g., BTC/USDT:USDT)", 
        "BTC/USDT:USDT", 
        help="No symbols fetched. Using fallback - enter manually if needed."
    )

timeframe = st.sidebar.selectbox(
    "Select Timeframe", 
    ('1m', '5m', '15m', '1h', '4h', '1d'), 
    index=3,  # Default to 1h for better reliability
    key='timeframe_selected'
)

limit = st.sidebar.slider("Number of Candles (Data Points)", 50, 200, 100)

st.sidebar.subheader("Turning Point Parameters")
st.sidebar.info("Turning points are identified using the sign of the first differences on the raw price data.")

# Add troubleshooting section
with st.sidebar.expander("🔧 Troubleshooting"):
    st.write("**If you get 403 errors:**")
    st.write("1. Try Binance (usually more accessible)")
    st.write("2. Use longer timeframes (1h, 4h, 1d)")
    st.write("3. Reduce the number of candles")
    st.write("4. Deploy from a different region")

# Define timeframe_duration_map for calculating candle durations
timeframe_duration_map = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}

if st.sidebar.button("Fetch Data and Run Analysis", type="primary"):
    st.subheader("📊 Data Fetching")
    
    with st.spinner(f"Fetching data for {symbol} from {selected_exchange_id}..."):
        ohlcv_data, error_msg = fetch_ohlcv_data(selected_exchange_id, symbol, timeframe, limit)
    
    if error_msg:
        st.error(error_msg)
        if "403" in error_msg or "Forbidden" in error_msg:
            st.error("**Geographical Restriction Detected!**")
            st.info("""
            **Possible Solutions:**
            1. **Try Binance**: Switch to Binance exchange (usually has fewer restrictions)
            2. **Use VPN**: Deploy your app through a VPN-enabled hosting service
            3. **Different Region**: Deploy from a different geographical region
            4. **Alternative Hosting**: Consider using hosting services that support VPN or proxy
            5. **Manual Data**: Use a CSV upload feature (would need to be implemented)
            """)
        st.stop()
    
    if not ohlcv_data:
        st.error("No data received. Please try different parameters.")
        st.stop()
    
    # Process the data
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    if df.empty:
        st.error("No data fetched. Please check symbol, timeframe, or exchange.")
        st.stop()
    
    st.success(f"✅ Successfully fetched {len(df)} candles from {selected_exchange_id.title()}")
    st.dataframe(df.tail())
    
    # Continue with the rest of your analysis code...
    st.subheader("🔄 Turning Point Calculation")
    turning_points = find_turning_points(df['Close'])
    
    if not turning_points:
        st.warning("No turning points confirmed with the current method. Ensure sufficient data points.")
        confirmed_turning_point_indices = []
    else:
        confirmed_turning_point_indices = [p[0] for p in turning_points]
        st.success(f"Found {len(confirmed_turning_point_indices)} confirmed turning points.")
    
    st.subheader("📈 Last 20 Turning Points")
    if turning_points:
        sorted_turning_points = sorted(turning_points, key=lambda x:x[0], reverse=True)
        minima_display = []
        maxima_display = []
        recent_points = sorted_turning_points[:20]
        recent_points.sort(key=lambda x:x[0])
        for p in recent_points:
            if p[2] == -1:
                minima_display.append(p)
            elif p[2] == 1:
                maxima_display.append(p)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### 🔴 Local Minima (Last 20)")
            if minima_display:
                minima_df = pd.DataFrame(minima_display, columns=['Timestamp','Price','Type'])
                minima_df['Type'] = 'Minima'
                minima_df['Timestamp'] = minima_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                st.dataframe(minima_df.set_index('Timestamp'))
            else:
                st.info("No local minima detected in the last 20 turning points.")
        
        with col2:
            st.write("#### 🟢 Local Maxima (Last 20)")
            if maxima_display:
                maxima_df = pd.DataFrame(maxima_display, columns=['Timestamp','Price','Type'])
                maxima_df['Type'] = 'Maxima'
                maxima_df['Timestamp'] = maxima_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                st.dataframe(maxima_df.set_index('Timestamp'))
            else:
                st.info("No local maxima detected in the last 20 turning points.")
    else:
        st.info("No turning points to display in the table.")
    
    st.subheader("💰 Buy/Sell Signals from Turning Points")
    if turning_points:
        signals_data = []
        for point in sorted(turning_points, key=lambda x:x[0]):
            signal_type = "🟢 Buy" if point[2] == -1 else "🔴 Sell"
            signals_data.append({
                'Timestamp': point[0].strftime('%Y-%m-%d %H:%M:%S'),
                'Price': f"{point[1]:.4f}",
                'Signal': signal_type
            })
        if signals_data:
            signals_df = pd.DataFrame(signals_data)
            st.dataframe(signals_df.tail(20).set_index('Timestamp'))
        else:
            st.info("No buy/sell signals generated from turning points.")
    else:
        st.info("No turning points detected to generate buy/sell signals.")

    st.subheader("📊 Turning Point Cycle Metrics")
    cycle_metrics_data = []
    current_timeframe_duration = timeframe_duration_map.get(timeframe, 1)

    if len(turning_points) > 1:
        for i in range(1, len(turning_points)):
            prev_point = turning_points[i-1]
            current_point = turning_points[i]

            time_diff_timedelta = current_point[0] - prev_point[0]
            duration_in_minutes = time_diff_timedelta.total_seconds() / 60
            duration_in_candles = duration_in_minutes / current_timeframe_duration

            price_change_abs = current_point[1] - prev_point[1]
            price_change_pct = (price_change_abs / prev_point[1]) * 100 if prev_point[1] != 0 else 0

            cycle_metrics_data.append({
                'Start Timestamp': prev_point[0],
                'End Timestamp': current_point[0],
                'Start Price': prev_point[1],
                'End Price': current_point[1],
                'Type': f"{'Min to Max' if prev_point[2] == -1 and current_point[2] == 1 else 'Max to Min' if prev_point[2] == 1 and current_point[2] == -1 else 'N/A'}",
                'Duration (Candles)': duration_in_candles,
                'Duration (Minutes)': duration_in_minutes,
                'Price Change %': price_change_pct
            })
        
        cycle_metrics_df = pd.DataFrame(cycle_metrics_data)
        # Format columns for display
        cycle_metrics_df_display = cycle_metrics_df.copy()
        cycle_metrics_df_display['Start Timestamp'] = cycle_metrics_df_display['Start Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        cycle_metrics_df_display['End Timestamp'] = cycle_metrics_df_display['End Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        cycle_metrics_df_display['Start Price'] = cycle_metrics_df_display['Start Price'].apply(lambda x: f"{x:.4f}")
        cycle_metrics_df_display['End Price'] = cycle_metrics_df_display['End Price'].apply(lambda x: f"{x:.4f}")
        cycle_metrics_df_display['Duration (Candles)'] = cycle_metrics_df_display['Duration (Candles)'].apply(lambda x: f"{x:.1f}")
        cycle_metrics_df_display['Duration (Minutes)'] = cycle_metrics_df_display['Duration (Minutes)'].apply(lambda x: f"{x:.0f}")
        cycle_metrics_df_display['Price Change %'] = cycle_metrics_df_display['Price Change %'].apply(lambda x: f"{x:.2f}%")

        st.dataframe(cycle_metrics_df_display)
    else:
        st.info("Not enough turning points to calculate cycle metrics (need at least 2 points).")

    st.subheader("🔮 Projected Next Turning Point")
    if len(cycle_metrics_data) > 0:
        avg_duration_candles = cycle_metrics_df['Duration (Candles)'].mean()
        avg_duration_minutes = cycle_metrics_df['Duration (Minutes)'].mean()
        avg_price_change_abs = cycle_metrics_df['End Price'].sub(cycle_metrics_df['Start Price']).abs().mean()
        
        last_turning_point = turning_points[-1]
        last_price = last_turning_point[1]
        last_timestamp = last_turning_point[0]
        last_type = last_turning_point[2] # -1 for min, 1 for max

        # Project next time based on average duration
        projected_next_time = last_timestamp + pd.to_timedelta(avg_duration_minutes, unit='m')

        # Project next price based on average price change
        # If last was a minimum, project an increase; if a maximum, project a decrease
        if last_type == -1: # Last was a minimum, next is likely a maximum
            projected_next_price = last_price + avg_price_change_abs
            projected_type_str = "Potential Maxima"
            emoji = "🟢"
        else: # Last was a maximum, next is likely a minimum
            projected_next_price = last_price - avg_price_change_abs
            projected_type_str = "Potential Minima"
            emoji = "🔴"
        
        st.success(f"{emoji} Based on historical cycles, the next potential turning point is a {projected_type_str}:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📅 Projected Time", f"{projected_next_time.strftime('%Y-%m-%d %H:%M:%S')}")
        with col2:
            st.metric("💰 Projected Price", f"{projected_next_price:.4f}")
        with col3:
            st.metric("⏱️ Cycle Duration", f"{avg_duration_candles:.1f} candles")
    else:
        st.info("Not enough historical cycles to project the next turning point.")

    st.subheader("📈 Backtested Turning Point Extrapolation Performance")
    if len(cycle_metrics_data) > 1: # Need at least two full cycles to evaluate a prediction
        time_deviations_minutes = []
        price_deviations_pct = []

        for i in range(len(cycle_metrics_data) - 1): # Iterate through cycles, predicting the next one
            # Calculate average metrics up to this point (non-repainting for this backtest)
            historical_cycles_for_avg = cycle_metrics_df.iloc[:i+1]
            avg_hist_duration_minutes = historical_cycles_for_avg['Duration (Minutes)'].mean()
            
            current_cycle_end_point_type = turning_points[i+1][2] # Type of the second point in the current cycle_metrics_data row
            avg_hist_price_change_abs = historical_cycles_for_avg['End Price'].sub(historical_cycles_for_avg['Start Price']).abs().mean()

            # 'Forecast' the next turning point
            forecast_start_time = cycle_metrics_df.iloc[i]['End Timestamp']
            forecasted_end_time = forecast_start_time + pd.to_timedelta(avg_hist_duration_minutes, unit='m')
            
            forecast_start_price = cycle_metrics_df.iloc[i]['End Price']
            if current_cycle_end_point_type == -1: # Current cycle ended at a minimum (prev was max)
                # Next cycle will go up, so expect a positive price change
                forecasted_end_price = forecast_start_price + avg_hist_price_change_abs
            else: # Current cycle ended at a maximum (prev was min)
                # Next cycle will go down, so expect a negative price change
                forecasted_end_price = forecast_start_price - avg_hist_price_change_abs

            # Compare with the actual next turning point
            actual_next_cycle_start_time = cycle_metrics_df.iloc[i+1]['Start Timestamp']
            actual_next_cycle_start_price = cycle_metrics_df.iloc[i+1]['Start Price']

            time_deviation_minutes = abs((forecasted_end_time - actual_next_cycle_start_time).total_seconds() / 60)
            price_deviation_pct = abs((forecasted_end_price - actual_next_cycle_start_price) / actual_next_cycle_start_price) * 100 if actual_next_cycle_start_price != 0 else 0

            time_deviations_minutes.append(time_deviation_minutes)
            price_deviations_pct.append(price_deviation_pct)
        
        if time_deviations_minutes:
            avg_time_deviation_minutes = np.mean(time_deviations_minutes)
            avg_price_deviation_pct = np.mean(price_deviations_pct)
            
            st.success("📊 Average historical accuracy of extrapolating turning points:")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("⏰ Average Time Deviation", f"{avg_time_deviation_minutes:.0f} minutes", f"{avg_time_deviation_minutes / current_timeframe_duration:.1f} candles")
            with col2:
                st.metric("📈 Average Price Deviation", f"{avg_price_deviation_pct:.2f}%")
            
            st.info("This indicates how consistently historical turning point patterns (duration and price change) repeat themselves.")
        else:
            st.info("Not enough historical cycles to perform backtested extrapolation analysis.")
    else:
        st.info("Not enough historical cycles to perform backtested extrapolation analysis (need at least 2 complete cycles).")

    st.write("---")
    st.subheader("📊 Crypto Price Chart with Turning Points")
    fig,ax1 = plt.subplots(figsize=(15,8))
    ax1.plot(df.index,df['Close'],label='Close Price',color='blue',linewidth=1.5)
    if turning_points and len(turning_points)>0:
        min_points_x=[p[0]for p in turning_points if p[2]==-1]
        min_points_y=[p[1]for p in turning_points if p[2]==-1]
        max_points_x=[p[0]for p in turning_points if p[2]==1]
        max_points_y=[p[1]for p in turning_points if p[2]==1]
        if min_points_x:
            ax1.plot(min_points_x,min_points_y,'o',color='red',markersize=6,label='Local Minimums')
        if max_points_x:
            ax1.plot(max_points_x,max_points_y,'o',color='green',markersize=6,label='Local Maximums')
    else:
        st.warning("No turning points detected to plot. Chart will only show price data.")
    
    # Plot projected next turning point
    if 'projected_next_time' in locals() and 'projected_next_price' in locals():
        ax1.axvline(x=projected_next_time, color='purple', linestyle=':', linewidth=1.5, label='Projected Next Pivot')
        ax1.plot(projected_next_time, projected_next_price, 'X', color='purple', markersize=10, label='Projected Pivot Point')
        ax1.annotate(f'Projected {projected_type_str}\n@ {projected_next_price:.4f}',
                     xy=(projected_next_time, projected_next_price),
                     xytext=(projected_next_time, projected_next_price + (df['Close'].max() - df['Close'].min()) * 0.05),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                     horizontalalignment='center', verticalalignment='bottom', color='purple', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="b", lw=1, alpha=0.6))

    ax1.set_title(f"{symbol} Price Chart with Turning Points ({timeframe}) - {selected_exchange_id.title()}")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)

# Add footer with additional information
st.markdown("---")
st.markdown("""
### 💡 Tips for Better Performance:
- **Binance** typically has fewer geographical restrictions
- Use **longer timeframes** (1h, 4h, 1d) for more reliable data
- **Reduce candle count** if you encounter timeout errors
- Consider deploying from regions with fewer restrictions
""")
