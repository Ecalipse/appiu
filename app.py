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
    """Fetches all future (perpetual) symbols for a given exchange, avoiding spot markets."""
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'enableRateLimit': True,
        })
        
        future_symbols_dict = {}

        if exchange_id == 'bybit':
            # For Bybit, specifically request 'linear' and 'inverse' categories to avoid 'spot'
            # If the Bybit API or CCXT's Bybit implementation still makes an internal spot call
            # that gets geo-blocked, this specific handling might still lead to the error.
            # The issue is likely the server's geographical location being blocked by Bybit's CDN for spot data.
            try:
                linear_markets = exchange.fetch_markets({'params': {'category': 'linear'}})
                for market in linear_markets:
                    if market.get('settleId') == 'USDT' and market['active'] and market.get('swap'): # ensure it's a perpetual swap
                        future_symbols_dict[market['symbol']] = market
            except Exception as e:
                # Catch the specific 403 error for Bybit and provide a more targeted message
                if "403 Forbidden" in str(e) and "category=spot" in str(e) and exchange_id == 'bybit':
                    st.error(f"**Geo-blocking detected for Bybit:** The server's location appears to be blocked by Bybit's CDN for fetching market data, specifically for 'spot' instruments. Even though the code attempts to fetch only futures, Bybit's API or CCXT's internal logic may still trigger a 'spot' market data request which is being denied. Please try selecting a different exchange (e.g., Binance) or consider running this application from a different network/location (e.g., using a VPN locally if this were your client).")
                else:
                    logging.warning(f"Could not fetch linear perpetual markets for Bybit: {e}")
                return [] # Return empty list if this crucial step fails for Bybit

            try:
                inverse_markets = exchange.fetch_markets({'params': {'category': 'inverse'}})
                for market in inverse_markets:
                    if market.get('settleId') == 'USD' and market['active'] and market.get('swap'): # ensure it's a perpetual swap
                        future_symbols_dict[market['symbol']] = market
            except Exception as e:
                if "403 Forbidden" in str(e) and "category=spot" in str(e) and exchange_id == 'bybit':
                    st.error(f"**Geo-blocking detected for Bybit:** The server's location appears to be blocked by Bybit's CDN for fetching market data, specifically for 'spot' instruments. Even though the code attempts to fetch only futures, Bybit's API or CCXT's internal logic may still trigger a 'spot' market data request which is being denied. Please try selecting a different exchange (e.g., Binance) or consider running this application from a different network/location (e.g., using a VPN locally if this were your client).")
                else:
                    logging.warning(f"Could not fetch inverse perpetual markets for Bybit: {e}")
                return [] # Return empty list if this crucial step fails for Bybit
            
            # If no markets found via category-specific fetch, something is wrong, or no such markets exist.
            if not future_symbols_dict:
                st.warning(f"No perpetual futures markets found for {exchange_id} using category-specific fetching. This might be expected or an issue.")

        else:
            # For other exchanges, use load_markets and filter, which is generally reliable.
            # ccxt's load_markets might try to fetch all types, but typically handles
            # market types more gracefully for non-geo-restricted scenarios.
            markets = exchange.load_markets() 
            for market_id, market in markets.items():
                is_future_or_swap = (
                    market.get('linear', False) or 
                    market.get('inverse', False) or 
                    market.get('contract', False) or 
                    market.get('swap', False)
                )
                is_usdt_usd_settled = market.get('settleId') in ['USDT', 'USD']
                # Ensure it's not a spot market explicitly if load_markets includes them
                is_not_spot = not market.get('spot', False) 
                
                if is_future_or_swap and is_usdt_usd_settled and is_not_spot and market['active']:
                    future_symbols_dict[market['symbol']] = market
        
        future_symbols = list(future_symbols_dict.keys())
        future_symbols.sort() # Sort alphabetically for easy selection
        return future_symbols
    except ccxt.NetworkError as e:
        # Generic network error for all exchanges
        st.error(f"Network error fetching symbols from {exchange_id}: {e}. This might be due to general network issues or geographical restrictions. Please check your network/location or try a different exchange.")
        return []
    except ccxt.ExchangeError as e:
        # Generic exchange error for all exchanges
        st.error(f"Exchange error fetching symbols from {exchange_id}: {e}. Please check the exchange status or your API permissions.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching symbols: {e}")
        return []
