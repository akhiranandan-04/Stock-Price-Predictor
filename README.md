# Stock Price Predictor (Python Application)

A simple and interactive Python-based stock price predictor that forecasts short-term stock prices. Supports both Indian (NSE/BSE) and international (NASDAQ/NYSE) stocks, with outputs formatted in INR for Indian stocks and USD for global stocks.

---

## ⚙️ Features

- **Daily price retrieval**: Automatically fetches the latest stock data.
- **Currency handling**: Displays Indian stocks in INR (₹) and international stocks in USD ($).
- **Graphical display**: Renders historical price charts with daily granularity covering the past 5 years.
- **ML prediction model** *(optional)*: Integrates a machine learning model to forecast short-term price movements.

---

## 📦 Installation

```bash
git clone https://github.com/akhiranandan-04/stock-price-predictor-PY-Application-.git
cd stock-price-predictor-PY-Application-
```

(Recommended) Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
python main.py
```

When prompted, enter a stock symbol:
- Indian stocks (e.g. `RELIANCE.NS`) → shown in **INR**
- International stocks (e.g. `AAPL`, `GOOGL`) → shown in **USD**

You will see:
- Daily adjusted close prices
- Graph with past 5 years' data
- (If implemented) Predicted prices for upcoming days

---

## 🧠 Model Training *(optional)*

If using a machine learning model:

```bash
python train_model.py
```

Ensure historical stock data is stored in the `data/` folder.

---

## 💾 Data

- Pulled from APIs like `yfinance` or `Alpha Vantage`
- Covers the **last 5 years** with **daily frequency**

---

## 🔧 Customization

- Modify currency formatting in `utils/format_currency.py`
- Change graph behavior in `main.py` or config files
- Add or switch ML models under `models/`

---

## ✅ To Do

- [ ] Improve error handling
- [ ] Add a simple web UI using Streamlit or Flask
- [ ] Deploy as a web app with scheduled updates
- [ ] Add support for additional exchanges

---

## 🛠️ Contribution

```bash
# Fork the repository
# Create a new branch for your feature
git checkout -b feature/your-feature

# Make your changes and commit
git commit -m "Add your feature"

# Push the changes and open a pull request
git push origin feature/your-feature
```

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## ✉️ Contact

Feel free to reach out via GitHub Issues or email for questions, suggestions, or collaboration.

---

*Happy Coding! 📈*
