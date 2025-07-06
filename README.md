# 💹 Financial Assistant using Agentic Workflows and Long-Term Memory (MCP)

> 👨‍💻 **Type:** Educational AI/ML Prototype  
> 🎯 **Focus:** Building AI workflows for financial reasoning

---

## 🔍 Overview

This project demonstrates a **modular pipeline** for building a financial assistant using modern machine learning tools and simulations. From classic NumPy calculations to predictive ML models and exploratory visualizations, it brings together the key layers of an intelligent financial reasoning system — structured using the **MCP (Model Context Protocol)** philosophy of **agentic workflows + long-term memory**.

---

## 🧠 Tech Stack

| Tool                 | Purpose                                        | Use in Notebook                              |
|----------------------|------------------------------------------------|----------------------------------------------|
| `NumPy`              | Vectorized math, simulations                   | Returns, volatility, Monte Carlo, correlation|
| `Pandas`             | Structured data handling                       | Cleaning, indexing, rolling mean, grouping   |
| `Matplotlib`         | Charting                                       | Stock plots, simulation paths                |
| `Seaborn`            | Stats visualization                            | Correlation matrix, regression scatter plots |
| `Scikit-learn`       | ML modeling & metrics                          | Linear regression, clustering, evaluation    |
| `yFinance`           | Real-world stock data                          | HDFC / RELIANCE stock price ingestion        |
| `PyTorch / TensorFlow` | Deep learning for time series, agents        | LSTM, Transformer (conceptual design)        |

---

## 📈 Features (Detailed)

### 1. 📊 Stock Price Simulation – Monte Carlo

- Simulates 10,000+ stock price paths using daily return volatility.
- Uses NumPy to model random walk behavior with normal distribution.
- Helps visualize risk and range of outcomes over 252 trading days.

### 2. 🧮 Financial Metrics – Daily Returns & Correlation

- Vectorized return calculations.
- Correlation matrix generation between simulated stock returns.
- Risk diversification insights from high/low correlation.

### 3. 📉 Machine Learning Models

- Real-time stock data from Yahoo Finance.
- Feature engineering: Open, High, Low, Volume, Day index.
- Model training, evaluation (R², MSE), and scatter plot visualization.

### 4. 🧠 Agentic Workflows – Modular & Context-Aware

- Layered design: data prep → ML modeling → insights → memory.
- Envisioned memory context for long-term adaptation (MCP).
- Combines procedural logic + learning agents.

### 5. 📅 Time Series Handling – Resampling & Rolling Windows

- Pandas used for rolling means, time-based smoothing.
- Useful for momentum analysis and financial signal extraction.

---

## 🔬 Deep Learning in Financial Assistants – PyTorch & TensorFlow

Although not directly implemented, the notebook outlines future directions using:

| Use Case                      | Model Type             | Framework         |
|-------------------------------|------------------------|-------------------|
| Time Series Forecasting       | LSTM / Transformer     | TensorFlow / PyTorch |
| Sentiment Analysis            | BERT, FinBERT          | TensorFlow / HuggingFace |
| Goal-Based Planning           | Reinforcement Learning | PyTorch           |
| Agentic Reasoning             | Multi-agent with memory| PyTorch           |

Deep learning is key for:
- Capturing temporal dependencies.
- Building agents that learn via feedback loops.
- Creating adaptive strategies in volatile markets.

---

## 🧪 Running Locally

```bash
pip install numpy pandas matplotlib seaborn scikit-learn yfinance
jupyter notebook EndGame_Analysts_05_07_2025.ipynb
```
---

# 👥 Team
## EndGame Analysts
### Authors:
- Joshua Vinu Koshy
- Poorva Raut
- Aakanksh Mishra 
- Sujit Lendave
