👨‍💻 Author
Gangadhar Badiger


# Stock_Prediction_webAppliction
Predict your stock and know the stock price after 5 years 
Features

User Signup / Login / Logout

OTP Email Verification & Password Reset

Payment Gateway (Razorpay)

Stock Data from Alpha Vantage

ML Prediction using XGBoost (future prices: 2025–2030)

Matplotlib graph generation

Prediction email sent to user

Admin Panel (users, logins, search history)

SQLite database

📦 Install Requirements
pip install flask numpy pandas xgboost scikit-learn matplotlib razorpay alpha_vantage


⚠️ Use Python 3.10 or 3.11 (3.14 not supported)



Get API Keys:
• Alpha Vantage: https://www.alphavantage.co

• Razorpay: https://razorpay.com

• Gmail App Passwords: https://myaccount.google.com/apppasswords

▶️ Run the Project
python app.py


App starts at:
👉 http://127.0.0.1:5000/

🛡 Admin Login
Username: admin
Password: admin123

📊 How Prediction Works (Short)

Fetch daily stock data

Use Close price only

Normalize using MinMaxScaler

Create time sequences (50 days)

Train XGBoost model

Predict 2025–2030 daily prices

Plot graph + send email

📁 Project Structure
app.py
users.db
templates/
static/

