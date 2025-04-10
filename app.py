import yfinance as yf
import pandas as pd
import numpy as np
import re
import xgboost as xgb
import pytz
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

# Auth setup
bcrypt = Bcrypt(app)
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# === Utility Functions ===
def savitzky_golay_filter(y, window_size, order=3):
    return savgol_filter(y, window_size, order)

@app.route('/api/stock_ticker')
def stock_ticker():
    tickers = [
        # Indian Stocks First
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
        'SBIN.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'ITC.NS', 'WIPRO.NS',
        # US Stocks
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE', 'INTC',
        # Indices
        '^NSEI', '^BSESN', '^GSPC', '^IXIC',
    ]

    data = {}
    try:
        # Get last 2 days of daily closing price
        stock_data = yf.download(tickers=tickers, period="2d", interval="1d", group_by='ticker', threads=True)

        for symbol in tickers:
            try:
                if symbol in stock_data:
                    closes = stock_data[symbol]['Close'].dropna()
                else:
                    closes = stock_data['Close'][symbol].dropna()

                if len(closes) >= 2:
                    latest = closes.iloc[-1]
                    prev = closes.iloc[-2]
                elif len(closes) == 1:
                    latest = prev = closes.iloc[0]
                else:
                    continue

                data[symbol] = {
                    "price": round(latest, 2),
                    "change": round(latest - prev, 2)
                }
            except Exception as e:
                print(f"Error with {symbol}: {e}")
                continue

    except Exception as e:
        print("Error fetching stock data:", e)

    return jsonify(data)
 	
@app.route('/api/market_summary')
def market_summary():
    try:
        gainers = ['RELIANCE.NS', 'LT.NS', 'SBIN.NS']
        losers = ['TCS.NS', 'ICICIBANK.NS', 'INFY.NS']
        result = {'gainers': [], 'losers': []}

        for symbol in gainers:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="3d")
            if not data.empty:
                close = data["Close"].iloc[-1]
                prev = data["Close"].iloc[0]
                change = round(((close - prev) / prev) * 100, 2)
                result['gainers'].append({
                    'symbol': symbol.replace('.NS', ''),
                    'price': round(close, 2),
                    'change': change
                })

        for symbol in losers:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="3d")
            if not data.empty:
                close = data["Close"].iloc[-1]
                prev = data["Close"].iloc[0]
                change = round(((close - prev) / prev) * 100, 2)
                result['losers'].append({
                    'symbol': symbol.replace('.NS', ''),
                    'price': round(close, 2),
                    'change': change
                })

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/index_summary')
def index_summary():
    try:
        symbols = {
            "nifty": "^NSEI",
            "sensex": "^BSESN",
            "banknifty": "^NSEBANK",
            "niftyit": "^CNXIT",
            "midcap": "^NSEMDCP50",
            "sp500": "^GSPC",
            "nasdaq": "^IXIC",
            "dowjones": "^DJI",
            "russell2000": "^RUT",
            "ftse100": "^FTSE",
            "dax": "^GDAXI",
            "nikkei": "^N225",
            "hangseng": "^HSI",
            "shanghai": "000001.SS"
        }
        index_data = {}

        for key, sym in symbols.items():
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="2d")
            if not hist.empty and len(hist) >= 2:
                prev_close = hist["Close"].iloc[-2]
                last_close = hist["Close"].iloc[-1]
                change = round(last_close - prev_close, 2)
                percent = round((change / prev_close) * 100, 2)

                index_data[key] = {
                    "value": round(last_close, 2),
                    "change": change,
                    "percent": percent
                }

        return jsonify(index_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_interval(timeframe):
    if timeframe == "1wk":
        return "30m"
    elif timeframe == "1mo":
        return "1h"
    elif timeframe == "3mo":
        return "1h"
    elif timeframe == "6mo":
        return "1d"
    elif timeframe == "1y":
        return "1d"
    elif timeframe == "2y":
        return "1wk"
    else:
        return "1d"

def get_num_prediction_days(timeframe):
    mapping = {
        "1wk": 7,
        "1mo": 30,
        "3mo": 60,
        "6mo": 90,
        "1y": 180,
        "2y": 365
    }
    return mapping.get(timeframe, 30)

def get_stock_data(symbol, timeframe):
    end_date = datetime.today() - timedelta(days=1)

    if timeframe == "1wk":
        start_date = end_date - timedelta(days=7)
    elif timeframe == "1mo":
        start_date = end_date - relativedelta(months=1)
    elif timeframe == "3mo":
        start_date = end_date - relativedelta(months=3)
    elif timeframe == "6mo":
        start_date = end_date - relativedelta(months=6)
    elif timeframe == "1y":
        start_date = end_date - relativedelta(years=1)
    elif timeframe == "2y":
        start_date = end_date - relativedelta(years=2)
    else:
        start_date = end_date - relativedelta(months=6)

    interval = get_interval(timeframe)
    print(f"Fetching {symbol} from {start_date.date()} to {end_date.date()} with interval={interval}")
    df = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
    print(df)
    if df.empty:
        print(f"No stock data found for {symbol}")
        return None

    if df is None or df.empty:
        return jsonify({'error': f'No data found for {symbol}'}), 404

    df["Date"] = df.index
    df.reset_index(drop=True, inplace=True)
    print("DEBUG: Data points fetched ->", len(df))
    return df

def generate_future_dates(df, num_days):
    last_known_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(start=last_known_date, periods=num_days + 1, freq='B')[1:]
    return future_dates.strftime('%Y-%m-%d').tolist()

def polynomial_regression_prediction(df, num_days, degree=3):
    series = df["Close"].squeeze()
    series.index = np.arange(len(series))

    # Smoothing
    smooth = int(2 * (series.shape[0] // 30 or 1) + 3)
    pts = savitzky_golay_filter(series.to_numpy(), smooth, 3)

    x = np.arange(len(series)).reshape(-1, 1)
    y = pts

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Create polynomial regression pipeline
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = r2_score(y_test, y_pred)
    print(f"Polynomial Regression (deg {degree}) R² Score: {accuracy:.4f}")

    # Future prediction
    future_x = np.arange(len(series), len(series) + num_days).reshape(-1, 1)
    future_predictions = model.predict(future_x)

    return {
        "future_dates": generate_future_dates(df, num_days),
        "future_predictions": future_predictions.tolist(),
        "r2_score": round(accuracy, 4)
    }


def lstm_prediction(df, num_days):
    data = df[['Close']].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    seq_length = 10

    for i in range(len(data_scaled) - seq_length - 1):
        X.append(data_scaled[i:i+seq_length])
        y.append(data_scaled[i+seq_length])

    if len(X) == 0:
        print("Not enough data for LSTM training.")
        return {
            "future_dates": generate_future_dates(df, num_days),
            "future_predictions": [None] * num_days,
            "train_loss": None
        }

    X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X, y, epochs=10, batch_size=16, verbose=0)

    final_loss = history.history['loss'][-1]
    print(f"LSTM Final Training Loss: {final_loss:.6f}")


    y_pred_scaled = model.predict(X_test, verbose=0)
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred_scaled)
    r2 = r2_score(y_test_inv, y_pred_inv)
    print(f"LSTM R² Score: {r2:.4f}")

    last_seq = data_scaled[-seq_length:].reshape(1, seq_length, 1)
    lstm_predictions = []

    for _ in range(num_days):
        next_pred = model.predict(last_seq, verbose=0)[0][0]
        lstm_predictions.append(next_pred)
        last_seq = np.append(last_seq[:, 1:, :], [[[next_pred]]], axis=1)

    lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1)).flatten().tolist()

    return {
        "future_dates": generate_future_dates(df, num_days),
        "future_predictions": lstm_predictions,
        "train_loss": round(float(final_loss), 6),
        "lstm_r2_score": round(r2, 4)
    }

def create_lag_features(series, lag=20):
    X, y = [], []
    for i in range(lag, len(series)):
        X.append(series[i-lag:i])
        y.append(series[i])
    return np.array(X), np.array(y)
  

def xgboost_prediction(df, num_days):
    print("Training XGBoost Regressor...")  # ✅ Log message updated

    series = df["Close"].squeeze()
    series.index = np.arange(len(series))

    lag = 20
    X, y = create_lag_features(series.to_numpy(), lag)

    X = X.reshape(X.shape[0], -1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # ✅ Changed from RandomForest to XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, objective='reg:squarederror', random_state=42)
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    print(f"XGBoost R² Score: {accuracy:.4f}")  # ✅ Log message updated

    # ✅ Future prediction loop same as before
    last_sequence = series.to_numpy()[-lag:]
    future_predictions = []
    for _ in range(num_days):
        input_seq = last_sequence.reshape(1, -1)
        next_val = xgb_model.predict(input_seq)[0]
        future_predictions.append(next_val)
        last_sequence = np.append(last_sequence[1:], next_val)

    return {
        "future_dates": generate_future_dates(df, num_days),
        "future_predictions": [round(float(p), 2) for p in future_predictions],
        "r2_score": round(accuracy, 4)
    }


def get_stock_prediction(symbol, timeframe):
    df = get_stock_data(symbol, timeframe)
    if df is None or df.empty:
        print(f"DEBUG: No stock data found for {symbol}")
        return None

    print(f"DEBUG: Successfully fetched stock data for {symbol}")
    latest_price = round(float(df["Close"].iloc[-1]), 2)
    num_days = get_num_prediction_days(timeframe)

    poly_pred = polynomial_regression_prediction(df, num_days, degree=3)
    lstm_pred = lstm_prediction(df, num_days)
    xgb_pred = xgboost_prediction(df, num_days)

    prediction_data = {
        "latest_price": latest_price,
        "dates": df["Date"].dt.strftime('%Y-%m-%d').tolist(),
        "actual_prices": df["Close"].squeeze().astype(float).round(2).tolist(),

        "linear_future_dates": poly_pred["future_dates"],
        "linear_future_predictions": [round(float(p), 2) for p in poly_pred["future_predictions"]],
        "linear_r2_score": poly_pred["r2_score"],

        "lstm_future_dates": lstm_pred["future_dates"],
        "lstm_future_predictions": [round(float(p), 2) for p in lstm_pred["future_predictions"]],
        "lstm_train_loss": lstm_pred["train_loss"],
        "lstm_r2_score": lstm_pred["lstm_r2_score"],

        "xgb_future_dates": xgb_pred["future_dates"],
        "xgb_future_predictions": xgb_pred["future_predictions"],
        "xgb_r2_score": xgb_pred["r2_score"]
    }

    return prediction_data


@app.route('/')
def index():
    return redirect(url_for('signup'))

@app.route('/home')
@login_required
def home():
    return render_template('home.html',user=current_user.username)

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        print("DEBUG: Received request data ->", data)

        symbol = data.get("symbol", "").upper()
        timeframe = data.get("timeframe", "6mo")

        print("Received timeframe:", timeframe)
        if not symbol:
            return jsonify({"error": "Stock symbol is required!"}), 400

        prediction_data = get_stock_prediction(symbol, timeframe)

        if prediction_data is None:
            return jsonify({"error": f"No data found for {symbol}!"}), 404

        return jsonify(prediction_data)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print("DEBUG: Full error traceback:\n", error_details)
        return jsonify({"error": "Internal Server Error", "details": error_details}), 500

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        # ✅ Gmail validation
        if not re.match(r'^[\w\.-]+@gmail\.com$', username):
            flash('Only Gmail addresses are allowed.', 'danger')
            return redirect(url_for('signup'))

        # ✅ Empty field check
        if not username or not password:
            flash('Username and password are required.', 'danger')
            return redirect(url_for('signup'))

        # ✅ Existing user check
        if User.query.filter_by(username=username).first():
            flash('This Gmail is already registered.', 'warning')
            return redirect(url_for('signup'))

        # ✅ Password strength check
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'danger')
            return redirect(url_for('signup'))

        # ✅ Create new user
        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        flash('Signup successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()

        # ✅ Empty field check
        if not username or not password:
            flash('Username and password are required.', 'danger')
            return redirect(url_for('login'))

        # ✅ User auth
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid Gmail or password.', 'danger')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully.', 'info')  # 👈 Toast message
    return redirect(url_for('login'))

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(use_reloader=True,debug=True)
