import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import logging

# Configure logging to suppress verbose output from libraries if needed
# logging.basicConfig(level=logging.INFO) # Set to DEBUG for more verbose, INFO for less.
# logging.getLogger('ccxt').setLevel(logging.WARNING) # Suppress CCXT debug logs if too noisy

st.set_page_config(layout="wide", page_title="Crypto Pivot Analyzer")

# --- Helper Functions ---

def find_turning_points(df_close_prices):
    """
    Finds local minimums and maximums in a series using the sign of the first differences.
    This method is robust to noise by focusing on the change in slope direction.
    Returns a list of tuples (timestamp, price, type), where type is 1 for max, -1 for min.
    """
    if len(df_close_prices) < 3: # Need at least 3 points for meaningful diff and sign changes
        return []

    # Calculate the first difference of the close prices
    first_diff = np.diff(df_close_prices.values)
    
    # Take the sign of the first difference. This transforms slopes into -1, 0, or 1.
    # A change from positive to negative sign (1 to -1) indicates a peak.
    # A change from negative to positive sign (-1 to 1) indicates a trough.
    signed_diff = np.sign(first_diff)
    
    # Calculate the difference of the signed differences.
    # This will be non-zero at points where the slope changes direction significantly.
    # For a peak, it will typically be -2 (1 - (-1)).
    # For a trough, it will typically be 2 (-1 - 1).
    diff_of_signed_diff = np.diff(signed_diff)

    turning_points = []
    
    # Iterate through the array of differences of signed differences
    # The index in diff_of_signed_diff corresponds to index + 1 in the original signed_diff
    # and index + 2 in the original df_close_prices due to two diff operations.
    for i in range(len(diff_of_signed_diff)):
        # Ensure the index aligns with the original df_close_prices
        original_index = df_close_prices.index[i + 1] 
        original_price = df_close_prices.iloc[i + 1]

        if diff_of_signed_diff[i] < 0: 
            turning_points.append((original_index, original_price, 1)) # 1 for local maximum
        elif diff_of_signed_diff[i] > 0:
            turning_points.append((original_index, original_price, -1)) # -1 for local minimum
    
    # Additional refinement: Ensure that consecutive turning points alternate between min and max.
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

def add_lagged_features(df, lags):
    df_copy = df.copy()
    for lag in lags:
        df_copy[f'Close_Lag_{lag}'] = df_copy['Close'].shift(lag)
        df_copy[f'High_Lag_{lag}'] = df_copy['High'].shift(lag)
        df_copy[f'Low_Lag_{lag}'] = df_copy['Low'].shift(lag)
        df_copy[f'Volume_Lag_{lag}'] = df_copy['Volume'].shift(lag)
    return df_copy

def add_technical_indicators(df):
    df_copy = df.copy()
    # Simple Moving Average (SMA)
    df_copy['SMA_10'] = df_copy['Close'].rolling(window=10).mean()
    df_copy['SMA_20'] = df_copy['Close'].rolling(window=20).mean()

    # Relative Strength Index (RSI)
    delta = df_copy['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_copy['RSI'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    exp1 = df_copy['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df_copy['Close'].ewm(span=26, adjust=False).mean()
    df_copy['MACD'] = exp1 - exp2
    df_copy['Signal_Line'] = df_copy['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df_copy['BB_Middle'] = df_copy['Close'].rolling(window=20).mean()
    df_copy['BB_Upper'] = df_copy['BB_Middle'] + (df_copy['Close'].rolling(window=20).std() * 2)
    df_copy['BB_Lower'] = df_copy['BB_Middle'] - (df_copy['Close'].rolling(window=20).std() * 2)

    # Price Rate of Change (ROC)
    df_copy['ROC'] = ((df_copy['Close'] - df_copy['Close'].shift(10)) / df_copy['Close'].shift(10)) * 100

    # True Range (TR) and Average True Range (ATR)
    df_copy['High-Low'] = df_copy['High'] - df_copy['Low']
    df_copy['High-PrevClose'] = abs(df_copy['High'] - df_copy['Close'].shift(1))
    df_copy['Low-PrevClose'] = abs(df_copy['Low'] - df_copy['Close'].shift(1))
    df_copy['TR'] = df_copy[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    df_copy['ATR'] = df_copy['TR'].rolling(window=14).mean()

    # On-Balance Volume (OBV)
    df_copy['OBV'] = (np.sign(df_copy['Close'].diff()) * df_copy['Volume']).fillna(0).cumsum()

    # Return
    df_copy['Return'] = df_copy['Close'].pct_change()

    return df_copy.drop(columns=['High-Low', 'High-PrevClose', 'Low-PrevClose', 'TR'])

def create_features_and_labels(df, turning_points, future_horizon=5):
    df_features = df.copy()

    # Add 'is_pivot' label (0 or 1)
    df_features['is_pivot'] = 0
    df_features['pivot_type'] = 0 # -1 for low, 1 for high, 0 for not a pivot
    
    # Map turning points to the DataFrame
    turning_point_data = pd.DataFrame(turning_points, columns=['timestamp', 'price', 'type'])
    turning_point_data['timestamp'] = pd.to_datetime(turning_point_data['timestamp'])
    
    # Convert df_features index to datetime if it's not already
    df_features.index = pd.to_datetime(df_features.index)

    # Use a more robust method to find the closest index without 'method' keyword
    df_index_values = df_features.index.values

    for _, row in turning_point_data.iterrows():
        target_timestamp = row['timestamp'].to_datetime64() # Convert to numpy datetime64 for comparison
        
        # Find the insertion point for the target timestamp in the sorted index
        idx_pos = np.searchsorted(df_index_values, target_timestamp)
        
        closest_idx = -1
        # Check if the insertion point is within bounds
        if idx_pos == 0:
            closest_idx = 0
        elif idx_pos == len(df_index_values):
            closest_idx = len(df_index_values) - 1
        else:
            # Compare distances to the two nearest indices
            left_diff = abs(df_index_values[idx_pos - 1] - target_timestamp)
            right_diff = abs(df_index_values[idx_pos] - target_timestamp)
            
            if left_diff <= right_diff:
                closest_idx = idx_pos - 1
            else:
                closest_idx = idx_pos
        
        # Apply the pivot label to the found index
        df_features.loc[df_features.index[closest_idx], 'is_pivot'] = 1
        df_features.loc[df_features.index[closest_idx], 'pivot_type'] = row['type']

    # Add 'future_pivot' label (1 if a pivot occurs in the next 'future_horizon' bars)
    df_features['future_pivot'] = 0
    for i in range(len(df_features) - future_horizon):
        if df_features['is_pivot'].iloc[i+1 : i+1+future_horizon].sum() > 0:
            df_features.loc[df_features.index[i], 'future_pivot'] = 1

    # Encode pivot_type (for multi-class classification)
    df_features['pivot_type_encoded'] = df_features['pivot_type'].map({-1: 0, 0: 1, 1: 2}) # Low=0, None=1, High=2

    return df_features

def train_xgboost_model(df_features, feature_columns, target_column='future_pivot'):
    df_model = df_features.dropna(subset=feature_columns + [target_column]).copy()

    if df_model.empty or len(df_model) < 50:
        st.warning("Insufficient data after dropping NaNs for XGBoost training. Skipping.")
        return None, None, None, None, None

    X = df_model[feature_columns]
    y = df_model[target_column]

    if y.nunique() < 2:
        st.warning(f"Not enough diverse data points for XGBoost pivot detection training (target has only {y.nunique()} unique values). Skipping.")
        return None, None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.success(f"XGBoost Pivot Detection Model Accuracy: {accuracy:.2f}")
    st.dataframe(pd.DataFrame(report).transpose())

    return model, X_test, y_test, accuracy, df_model.index[-1]

def predict_next_pivots(model, df_latest_features, feature_columns, future_horizon, prediction_threshold=0.3):
    """
    Predicts future pivots based on the trained model.
    df_latest_features is expected to be the DataFrame with all features already calculated.
    """
    if model is None:
        return [], 0

    # Ensure we get the last row where all relevant feature columns are non-NaN
    # This addresses the "Features for prediction contain NaN values" error
    X_pred_latest = df_latest_features[feature_columns].dropna().tail(1).copy()

    if X_pred_latest.empty:
        st.warning("No valid data points with complete features available for prediction. Cannot make prediction.")
        return [], 0
    
    # No scaler needed as SVM is removed, so X_pred_latest_scaled is just X_pred_latest
    probabilities = model.predict_proba(X_pred_latest)[0]

    if len(probabilities) == 2:
        pivot_prob = probabilities[1]
        predicted_class = model.predict(X_pred_latest)[0]
    else:
        st.error("Unexpected number of classes in prediction probabilities.")
        return [], 0

    st.info(f"Latest candle's predicted pivot probability: {pivot_prob:.4f}")

    future_pivots = []
    num_future_pivots = 0
    if pivot_prob >= prediction_threshold:
        # Get the timestamp of the actual last candle in the original dataframe
        # This is critical to correctly place the prediction in time
        last_index_dt = df_latest_features.index[-1]
            
        timeframe_map = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}
        interval_minutes = timeframe_map.get(st.session_state.timeframe_selected, 1)
        
        # Predict the pivot time 'future_horizon' steps into the future from the last data point
        predicted_pivot_time = pd.to_datetime(last_index_dt) + pd.to_timedelta(future_horizon * interval_minutes, unit='m')
        # Use the price of the last data point from which prediction was made
        predicted_price = df_latest_features['Close'].iloc[-1] 
        pivot_type_str = ""

        if predicted_class == 1:
            pivot_type_str = "Predicted Future Pivot"

        future_pivots.append({
            'time': predicted_pivot_time,
            'price': predicted_price,
            'type': pivot_type_str,
            'probability': pivot_prob
        })
        num_future_pivots = len(future_pivots)

    return future_pivots, num_future_pivots

@st.cache_data(ttl=3600) # Cache for 1 hour to reduce API calls
def fetch_all_future_symbols(exchange_id):
    """Fetches all future (perpetual) symbols for a given exchange."""
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'enableRateLimit': True,
        })
        markets = exchange.load_markets()
        
        future_symbols = []
        for market_id, market in markets.items():
            # Check for common indicators of future/perpetual contracts
            # This logic might need adjustment based on specific exchange nuances
            is_future = (
                market.get('linear', False) or market.get('inverse', False) or market.get('contract', False)
            )
            # Filter for USDT or USD settled futures as common use case
            is_usdt_usd_settled = market.get('settleId') in ['USDT', 'USD']
            
            if is_future and is_usdt_usd_settled and market['active']:
                future_symbols.append(market['symbol'])
        
        future_symbols.sort() # Sort alphabetically for easy selection
        return future_symbols
    except ccxt.NetworkError as e:
        st.error(f"Network error fetching symbols from {exchange_id}: {e}")
        return []
    except ccxt.ExchangeError as e:
        st.error(f"Exchange error fetching symbols from {exchange_id}: {e}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching symbols: {e}")
        return []

# --- Streamlit UI ---
st.title("Crypto Pivot Analyzer with ML Forecasting")

st.sidebar.header("Configuration")
# Use st.session_state to store and update selected exchange
selected_exchange_id = st.sidebar.selectbox(
    "Select Exchange", 
    ('bybit', 'binance', 'coinbasepro'), 
    index=0,
    key='exchange_selected'
)

# Fetch symbols based on the selected exchange
all_available_symbols = fetch_all_future_symbols(selected_exchange_id)

# Set a default symbol, or use the first available one if the previous default is not in the list
default_symbol_index = 0
if 'BTC/USDT:USDT' in all_available_symbols: # Common Bybit/Binance perpetual
    default_symbol_index = all_available_symbols.index('BTC/USDT:USDT')
elif all_available_symbols:
    default_symbol_index = 0 # Fallback to the first symbol if common ones not found
else:
    default_symbol_index = None # No symbols available

if all_available_symbols:
    symbol = st.sidebar.selectbox(
        "Select Symbol", 
        options=all_available_symbols, 
        index=default_symbol_index if default_symbol_index is not None else 0,
        key='symbol_selected'
    )
else:
    symbol = st.sidebar.text_input("Enter Symbol (e.g., BTC/USDT)", "BTC/USDT", help="No symbols fetched for the selected exchange. Please enter manually.")

timeframe = st.sidebar.selectbox("Select Timeframe", ('1m', '5m', '15m', '1h', '4h', '1d'), index=0, key='timeframe_selected')

limit = st.sidebar.slider("Number of Candles (Data Points)", 150, 200, 100)

st.sidebar.subheader("Turning Point Parameters")
st.sidebar.info("Turning points are identified using the sign of the first differences on the raw price data.")

st.sidebar.subheader("Machine Learning Parameters")
future_horizon = st.sidebar.slider("Future Horizon (Candles)", 1, 10, 5)
prediction_threshold = st.sidebar.slider("Prediction Probability Threshold", 0.05, 0.95, 0.3, 0.05)

st.sidebar.info("Adjust 'Future Horizon' for how far ahead to predict pivots. 'Prediction Probability Threshold' controls the confidence level for showing predictions.")

# --- Main Logic ---
if st.sidebar.button("Fetch Data and Run Analysis"):
    st.subheader("Data Fetching")
    try:
        # Use the selected exchange from the selectbox
        exchange = getattr(ccxt, selected_exchange_id)({
            'enableRateLimit': True,
        })
        st.write(f"Fetching data for {symbol} from {selected_exchange_id}...")

        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        if df.empty:
            st.error("No data fetched. Please check symbol, timeframe, or exchange.")
            st.stop()

        st.success(f"Successfully fetched {len(df)} candles.")
        st.dataframe(df.tail())

        st.subheader("Turning Point Calculation")
        turning_points = find_turning_points(df['Close'])

        if not turning_points:
            st.warning("No turning points confirmed with the current method. Ensure sufficient data points.")
            confirmed_turning_point_indices = []
        else:
            confirmed_turning_point_indices = [p[0] for p in turning_points]
            st.success(f"Found {len(confirmed_turning_point_indices)} confirmed turning points.")

        # --- Display Last 20 Turning Points ---
        st.subheader("Last 20 Turning Points")
        if turning_points:
            sorted_turning_points = sorted(turning_points, key=lambda x: x[0], reverse=True) 
            
            minima_display = []
            maxima_display = []

            recent_points = sorted_turning_points[:20]
            recent_points.sort(key=lambda x: x[0]) 

            for p in recent_points:
                if p[2] == -1:
                    minima_display.append(p)
                elif p[2] == 1:
                    maxima_display.append(p)

            st.write("#### Local Minima (Last 20)")
            if minima_display:
                minima_df = pd.DataFrame(minima_display, columns=['Timestamp', 'Price', 'Type'])
                minima_df['Type'] = 'Minima' 
                minima_df['Timestamp'] = minima_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                st.dataframe(minima_df.set_index('Timestamp'))
            else:
                st.info("No local minima detected in the last 20 turning points.")

            st.write("#### Local Maxima (Last 20)")
            if maxima_display:
                maxima_df = pd.DataFrame(maxima_display, columns=['Timestamp', 'Price', 'Type'])
                maxima_df['Type'] = 'Maxima' 
                maxima_df['Timestamp'] = maxima_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                st.dataframe(maxima_df.set_index('Timestamp'))
            else:
                st.info("No local maxima detected in the last 20 turning points.")
        else:
            st.info("No turning points to display in the table.")

        # --- Buy/Sell Signals Table ---
        st.subheader("Buy/Sell Signals from Turning Points")
        if turning_points:
            signals_data = []
            for point in sorted(turning_points, key=lambda x: x[0]): 
                signal_type = "Buy" if point[2] == -1 else "Sell"
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

        # --- Feature Engineering ---
        st.subheader("Feature Engineering")
        Lags = [1, 2, 3, 5, 10]
        df_lags = add_lagged_features(df, Lags)
        df_indicators = add_technical_indicators(df_lags)
        df_features_labels = create_features_and_labels(df_indicators, turning_points, future_horizon)

        st.write("Generated features and labels:")
        st.dataframe(df_features_labels.tail())

        feature_columns = [col for col in df_features_labels.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'is_pivot', 'pivot_type', 'future_pivot', 'pivot_type_encoded', 'BB_Middle', 'BB_Upper', 'BB_Lower']]
        feature_columns = [col for col in feature_columns if not df_features_labels[col].isnull().all()] 
                                
                                # --- Machine Learning Models ---
        st.subheader("Machine Learning Model Training")
        st.info("Training XGBoost for 'future_pivot' prediction (Is there a pivot in the next X candles?).")
        xgboost_model, _, _, _, _ = train_xgboost_model(df_features_labels, feature_columns, 'future_pivot')
        st.info("SVM model removed as per request.")

        # --- Predictions ---
        st.subheader("Forecasting Future Pivots")
        future_pivots_plot = []
        num_future_pivots_xgboost = 0

        if xgboost_model:
            xgboost_predicted_pivots, num_future_pivots_xgboost = predict_next_pivots(
                xgboost_model, df_features_labels, feature_columns, future_horizon, prediction_threshold
            )
            future_pivots_plot.extend(xgboost_predicted_pivots)
            st.info(f"XGBoost predicted {num_future_pivots_xgboost} future pivot events.")
        else:
            st.warning("XGBoost model not trained. Cannot make future pivot predictions.")
        
        total_predicted_pivots = len(future_pivots_plot)
        if total_predicted_pivots == 0:
            st.info("No potential future pivots predicted based on current data and threshold.")
        else:
            st.success(f"Found {total_predicted_pivots} potential future pivots.")

        # --- Visualization ---
        st.write("---")
        st.subheader("Crypto Price Chart with Turning Points and ML Predicted Pivots")

        st.write(f"Confirmed Turning Point Indices count: {len(confirmed_turning_point_indices)}")
        st.write(f"Future Pivots to Plot count: {len(future_pivots_plot)}")

        fig, ax1 = plt.subplots(figsize=(15, 8))

        ax1.plot(df.index, df['Close'], label='Close Price', color='blue', linewidth=1.5)

        if turning_points and len(turning_points) > 0:
            min_points_x = [p[0] for p in turning_points if p[2] == -1]
            min_points_y = [p[1] for p in turning_points if p[2] == -1]
            max_points_x = [p[0] for p in turning_points if p[2] == 1]
            max_points_y = [p[1] for p in turning_points if p[2] == 1]
            
            if min_points_x:
                ax1.plot(min_points_x, min_points_y, 'o', color='red', markersize=6, label='Local Minimums')

            if max_points_x:
                ax1.plot(max_points_x, max_points_y, 'o', color='green', markersize=6, label='Local Maximums')

        else:
            st.warning("No turning points detected to plot. Chart will only show price data.")

        if future_pivots_plot:
            for pivot in future_pots_plot:
                time_dt = pivot['time']
                price = pivot['price']
                pivot_type = pivot['type']
                probability = pivot['probability']

                ax1.axvline(x=time_dt, color='purple', linestyle=':', linewidth=1.5, label='Predicted Pivot' if pivot == future_pivots_plot[0] else "")
                ax1.annotate(f'{pivot_type}\nProb: {probability:.2f}',
                                    xy=(time_dt, price),
                                    xytext=(time_dt, price + (df['Close'].max() - df['Close'].min()) * 0.05),
                                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                                    horizontalalignment='center', verticalalignment='bottom', color='purple', fontsize=9,
                                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="b", lw=1, alpha=0.6))
            st.success("Future pivot predictions plotted.")
        else:
            st.info("No future pivots to plot.")

        ax1.set_title(f"{symbol} Price Chart with Turning Points and ML Predicted Pivots ({timeframe})")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Price")
        ax1.grid(True)
        ax1.legend()
        fig.autofmt_xdate()

        st.pyplot(fig)
        
        # --- Chart with Line of Best Fit and Error ---
        st.write("---")
        st.subheader("Turning Points with Line of Best Fit and Residuals")

        if turning_points:
            turning_point_df = pd.DataFrame(turning_points, columns=['timestamp', 'price', 'type'])
            
            X_tp = (turning_point_df['timestamp'].astype(np.int64) // 10**9).values.reshape(-1, 1)
            y_tp = turning_point_df['price'].values

            if len(X_tp) > 1:
                model_lr = LinearRegression()
                model_lr.fit(X_tp, y_tp)
                
                y_pred_lr = model_lr.predict(X_tp)

                fig_lr, ax_lr = plt.subplots(figsize=(15, 8))

                x_plot_dt = pd.to_datetime(X_tp.flatten() * 10**9)

                min_points_x_plot = [p_dt for p_dt, p_type in zip(x_plot_dt, turning_point_df['type']) if p_type == -1]
                min_points_y_plot = [p_y for p_y, p_type in zip(turning_point_df['price'], turning_point_df['type']) if p_type == -1]
                max_points_x_plot = [p_dt for p_dt, p_type in zip(x_plot_dt, turning_point_df['type']) if p_type == 1]
                max_points_y_plot = [p_y for p_y, p_type in zip(turning_point_df['price'], turning_point_df['type']) if p_type == 1]

                if min_points_x_plot:
                    ax_lr.plot(min_points_x_plot, min_points_y_plot, 'o', color='red', markersize=6, label='Local Minimums')
                if max_points_x_plot:
                    ax_lr.plot(max_points_x_plot, max_points_y_plot, 'o', color='green', markersize=6, label='Local Maximums')

                ax_lr.plot(x_plot_dt, y_pred_lr, color='blue', linestyle='--', label='Line of Best Fit')

                for i in range(len(turning_point_df)):
                    tp_time_dt = turning_point_df['timestamp'].iloc[i]
                    tp_price = turning_point_df['price'].iloc[i]
                    predicted_price_at_tp = y_pred_lr[i]
                    
                    ax_lr.plot([tp_time_dt, tp_time_dt], [tp_price, predicted_price_at_tp], color='gray', linestyle=':', linewidth=0.8)

                ax_lr.set_title("Turning Points with Line of Best Fit and Residuals") 
                ax_lr.set_xlabel("Time")
                ax_lr.set_ylabel("Price")
                ax_lr.grid(True)
                ax_lr.legend()
                fig_lr.autofmt_xdate()
                st.pyplot(fig_lr)
            else:
                st.info("Not enough turning points to compute a line of best fit (need at least 2 points).")
        else:
            st.info("No turning points detected to plot the line of best fit.")

    except ccxt.NetworkError as e:
        st.error(f"Network error: {e}. Please check your internet connection or try again later.")
    except ccxt.ExchangeError as e:
        st.error(f"Exchange error: {e}. Please check the symbol, timeframe, or exchange ID.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.exception(e)
