# 🚖 Uber Trip Analysis & Prediction Dashboard  
*A Data Analytics + Machine Learning Dashboard built with Streamlit*

This project was developed as part of my **AI–ML Internship with Edunet Foundation in collaboration with Microsoft**.  
The dashboard provides **interactive data analysis**, **visual insights**, and **machine learning–based trip demand prediction** using the Uber FOIL dataset.

---

## 📌 Project Overview

This project converts raw Uber trip data into meaningful insights.  
It performs:

- Data cleaning & preprocessing  
- Feature engineering (day, hour, month, day-of-week)  
- Exploratory Data Analysis (EDA)  
- Visualization of trip patterns  
- Machine Learning model training  
- Demand prediction based on date, base, and active vehicles  
- A fully interactive, multi-page Streamlit dashboard with light & dark themes  

The main goal is to understand **when Uber demand increases** and to make **future trip predictions**.

---

## 🌟 Key Features

### 🔹 1. Data Upload & Preprocessing  
- Upload CSV dataset  
- Automatic date-time conversion  
- Extraction of features: Day, Month, Hour, DayOfWeek  
- Label encoding for dispatching base numbers  

### 🔹 2. Exploratory Data Analysis  
Interactive charts including:

- Trips per hour  
- Trips per day of week  
- Heatmap (Hour × DayOfWeek)  
- Trend analysis over time  
- Base-wise trip patterns  

### 🔹 3. Machine Learning Model  
- **Random Forest Regressor**  
- High accuracy (R² ≈ 0.98)  
- Model evaluation metrics: MAE, MSE, RMSE  
- Feature importance visualization  

### 🔹 4. Trip Prediction  
Predict expected number of trips based on:

- Base  
- Date  
- Active vehicles  

### 🔹 5. User Interface  
- Multi-page dashboard  
- Sidebar navigation  
- Light and Dark themes  
- Clean UI with pastel design  
- Footer: “Created with 💙 by Riya Roy”  

---

## 🗂 Dataset

Dataset used: **Uber FOIL Dataset (Jan–Feb)**  
Contains:

- Dispatching base number  
- Date/Time  
- Number of active vehicles  
- Trips  
- Time-based patterns  

Due to license restrictions, the dataset is not included in this repository.  
You may upload your own Uber FOIL CSV file.

---

## 🛠 Tech Stack

- **Python**
- **Pandas, NumPy**
- **Matplotlib, Seaborn, Plotly**
- **Scikit-Learn**
- **Streamlit**
- **HTML + CSS (for custom themes)**

---
