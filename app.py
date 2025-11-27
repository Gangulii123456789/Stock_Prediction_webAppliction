import os
import secrets
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, flash
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import base64
import razorpay
from functools import wraps
from alpha_vantage.timeseries import TimeSeries

# Flask app setup
app = Flask(__name__)

# Generate a random secret key for session management
app.secret_key = secrets.token_hex(16)

# Razorpay API Keys (Replace with your own)
RAZORPAY_KEY_ID = "rzp_test_e664V0FP0zQy7N"
RAZORPAY_KEY_SECRET = "QdnuRxUHrPGeiJc9lDTXYPO7"

# Admin credentials (You can replace this with a secure DB check)
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'admin123'

# Initialize Razorpay Client
razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

# SQLite Database setup
DATABASE = 'users.db'

def init_db():
    """Initialize the SQLite database."""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                username TEXT NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                login_timestamp TEXT NOT NULL
            )
        ''')
        conn.commit()

init_db()

# SMTP Configuration for OTP and Password Reset
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SMTP_USE_TLS = True
SENDER_EMAIL = 'kambarv23@gmail.com'  # Replace with your email
SENDER_PASSWORD = 'hdiy bpip abke pezi'  # Replace with your email password

# Token store for password reset
reset_tokens = {}  # token: (email, expiry_time)

# Initialize Alpha Vantage API Client (Replace with your Alpha Vantage API Key)
ALPHA_VANTAGE_API_KEY = "54HMXTPTECEPQVWU"  # Replace with your Alpha Vantage API key
ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')

# ---------------- STOCK PRICE PREDICTION FUNCTIONS --------------
def fetch_stock_data(stock_symbol):
    """Fetch historical stock data from Alpha Vantage."""
    try:
        data, meta_data = ts.get_daily(symbol=stock_symbol, outputsize='full')
        df = data[['4. close']]
        df.columns = ['Close']
        
        if df.empty:
            raise ValueError("Invalid stock symbol or no data available.")
        
        return df
    except Exception as e:
        raise ValueError(f"Error fetching stock data: {str(e)}")

def prepare_data(df):
    """Normalize stock price data."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

def create_sequences(data, time_step=50):
    """Create time series sequences for training the model."""
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

def build_and_train_model(X, Y):
    """Train an XGBoost regression model."""
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X, Y)
    return model

def predict_stock_price(stock_symbol):
    """Predict future stock prices using trained XGBoost model for the years 2025 to 2030."""
    df = fetch_stock_data(stock_symbol)
    data, scaler = prepare_data(df)
    X, Y = create_sequences(data)

    if len(X) == 0 or len(Y) == 0:
        raise ValueError("Insufficient data to train the model.")

    model = build_and_train_model(X, Y)

    future_dates = pd.date_range(start="2025-01-01", end="2030-12-31", freq='D')
    future_prices = []

    test_data = data[-50:].reshape(1, -1)

    for _ in range(len(future_dates)):
        prediction = model.predict(test_data)
        predicted_price = scaler.inverse_transform(np.array(prediction).reshape(-1, 1))
        future_prices.append(predicted_price[0][0])
        test_data = np.roll(test_data, -1)
        test_data[0, -1] = prediction[0]

    return future_prices, future_dates, df

def plot_stock_data(df, future_dates, future_prices):
    """Generate a stock price trend plot for the years 2025 to 2030."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(df.index, df['Close'], label="Historical Price", color='blue')
    ax.plot(future_dates, future_prices, label="Predicted Price", linestyle='dashed', color='red')
    ax.set_title("Stock Price Trend (2025-2030)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price (USD)")
    ax.legend()

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)  # Close the figure after saving to free memory
    return plot_url

# ---------------- DATABASE UTILITIES ----------------
def get_user_by_email(email):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, email, username, password FROM users WHERE email=?", (email,))
        return cursor.fetchone()

def insert_user(email, username, password):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (email, username, password) VALUES (?, ?, ?)", (email, username, password))
        conn.commit()

def update_user_password(email, new_password):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET password=? WHERE email=?", (new_password, email))
        conn.commit()

def get_all_users():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT email, username FROM users")
        return cursor.fetchall()

def log_user_login(email):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO logins (user_email, login_timestamp) VALUES (?, ?)", (email, datetime.now().isoformat()))
        conn.commit()

def get_all_logins():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_email, login_timestamp FROM logins ORDER BY login_timestamp DESC")
        return cursor.fetchall()

# ---------------- USER AUTHENTICATION ROUTES ----------------
def send_otp_email(receiver_email):
    """Send OTP to the user's email address."""
    otp = random.randint(100000, 999999)
    session['otp'] = otp
    session['otp_expiry'] = datetime.now() + timedelta(minutes=5)
    
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = receiver_email
    msg['Subject'] = 'OTP for Your Account Verification'

    body = f"Your OTP code is {otp}. It will expire in 5 minutes."
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, receiver_email, text)
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending OTP: {e}")
        return False

def send_password_reset_email(receiver_email, reset_token):
    """Send password reset link to the user's email."""
    reset_link = url_for('reset_password', token=reset_token, _external=True)

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = receiver_email
    msg['Subject'] = 'Password Reset Request'

    body = f"Click the following link to reset your password: {reset_link}"
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, receiver_email, text)
        server.quit()
    except Exception as e:
        print(f"Error sending reset email: {e}")

# ---------------- ROUTE DECORATORS ----------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'email' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'is_admin' not in session:
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

# ---------------- ROUTES ----------------
@app.route('/')
@login_required
def home():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')

        if get_user_by_email(email):
            flash("User already exists!", "error")
            return redirect(url_for('signup'))

        insert_user(email, username, password)
        flash("Signup successful! Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = get_user_by_email(email)

        if user and user[3] == password:  # user[3] is password
            session['email'] = email
            log_user_login(email)  # Record login event
            flash("Login successful!", "success")
            return redirect(url_for('home'))
        else:
            flash("Invalid email or password.", "error")
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for('login'))

# ---------------- PASSWORD RESET ROUTES ----------------
@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')

        if get_user_by_email(email):
            reset_token = secrets.token_hex(16)
            reset_tokens[reset_token] = (email, datetime.now() + timedelta(minutes=15))
            send_password_reset_email(email, reset_token)
            flash("Password reset link has been sent to your email.", "info")
            return redirect(url_for('login'))
        else:
            flash("Email not found.", "error")
            return redirect(url_for('forgot_password'))

    return render_template('forgot_password.html')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    info = reset_tokens.get(token)
    if not info:
        flash("Invalid or expired token.", "error")
        return redirect(url_for('login'))
    email, expiry = info
    if datetime.now() > expiry:
        flash("Token has expired.", "error")
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        new_password = request.form.get('new_password')
        if get_user_by_email(email):
            update_user_password(email, new_password)
            flash("Password updated successfully!", "success")
            reset_tokens.pop(token, None)
            return redirect(url_for('login'))

    return render_template('reset_password.html', token=token)

# ---------------- PREDICTION AND PAYMENT ROUTES ----------------
@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        stock_symbol = request.form.get('symbol', 'AAPL').strip().upper()
        session['stock_symbol'] = stock_symbol
        session['paid'] = False
        return redirect(url_for('payment_page', symbol=stock_symbol))

    stock_symbol = session.get('stock_symbol', 'AAPL')
    if 'paid' not in session or not session['paid']:
        return redirect(url_for('payment_page', symbol=stock_symbol))

    try:
        future_prices, future_dates, df = predict_stock_price(stock_symbol)
        plot_url = plot_stock_data(df, future_dates, future_prices)
        predictions = {str(future_dates[i].date()): round(future_prices[i], 2) for i in range(len(future_dates))}

        history = session.get('search_history', [])
        if stock_symbol not in history:
            history.append(stock_symbol)
        session['search_history'] = history

        user_email = session.get('email')
        if user_email:
            send_prediction_email(user_email, stock_symbol, predictions)

        return render_template('result.html', symbol=stock_symbol, predictions=predictions, plot_url=plot_url)

    except Exception as e:
        return render_template('index.html', error=str(e))

def send_prediction_email(receiver_email, stock_symbol, predictions):
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = receiver_email
    msg['Subject'] = f"Stock Predictions for {stock_symbol}"

    body = f"Here are the predicted prices for {stock_symbol}:\n\n"
    for date, price in predictions.items():
        body += f"{date}: ${price}\n"
    
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, receiver_email, text)
        server.quit()
    except Exception as e:
        print(f"Error sending prediction email: {e}")

# ---------------- PAYMENT INTEGRATION ----------------
@app.route('/payment')
@login_required
def payment_page():
    stock_symbol = request.args.get('symbol', 'AAPL')
    session['stock_symbol'] = stock_symbol
    order_amount = 100 * 100  # ₹100 in paise
    order_currency = 'INR'
    order_receipt = f"order_{stock_symbol}_{datetime.now().timestamp()}"

    razorpay_order = razorpay_client.order.create({
        'amount': order_amount,
        'currency': order_currency,
        'receipt': order_receipt,
        'payment_capture': '1'
    })

    return render_template('payment.html', order=razorpay_order, key=RAZORPAY_KEY_ID, symbol=stock_symbol)

from razorpay.errors import SignatureVerificationError

@app.route('/verify_payment', methods=['POST'])
@login_required
def verify_payment():
    try:
        params_dict = {
            'razorpay_order_id': request.form['razorpay_order_id'],
            'razorpay_payment_id': request.form['razorpay_payment_id'],
            'razorpay_signature': request.form['razorpay_signature']
        }

        razorpay_client.utility.verify_payment_signature(params_dict)

        flash("Payment verified successfully!", "success")
        session['paid'] = True

        return redirect(url_for('predict'))

    except SignatureVerificationError:
        flash("Payment verification failed!", "error")
        return redirect(url_for('payment_failure'))

@app.route('/payment-success')
@login_required
def payment_success():
    return render_template('payment_success.html')

@app.route('/payment-failure')
@login_required
def payment_failure():
    return render_template('payment_failure.html')

# ---------------- ADMIN ROUTES ----------------
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['is_admin'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template('admin_login.html', error="Invalid credentials.")
    return render_template('admin_login.html')

@app.route('/admin')
@admin_login_required
def admin_dashboard():
    users = get_all_users()  # [(email, username), ...]
    logins = get_all_logins()  # [(user_email, login_timestamp), ...]
    search_history = session.get('search_history', [])
    return render_template('admin_dashboard.html', users=users, logins=logins, history=search_history)

@app.route('/admin/logout')
def admin_logout():
    session.pop('is_admin', None)
    return redirect(url_for('home'))

# ---------------- RUN FLASK APP ----------------
if __name__ == '__main__':
    app.run(debug=True)


