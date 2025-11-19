# Command to start the project
#         cd "C:\Users\royri\OneDrive\Desktop\Riya Project\Riya Project"
#         .\.venv\Scripts\Activate
#         streamlit run app.py
# ==============================================================
# UBER TRIP ANALYSIS & PREDICTION DASHBOARD
# --------------------------------------------------------------
# Technologies Used:
#   - Python
#   - Pandas, NumPy
#   - Matplotlib, Seaborn, Plotly
#   - Scikit-learn (RandomForestRegressor)
#   - Streamlit (for interactive web dashboard)
#
# Features:
#   1. CSV Upload and Data Preprocessing
#   2. Multi-page Navigation (Home, Dataset, Visuals, ML, Predictions, Insights, About)
#   3. Light/Dark pastel themes with custom CSS
#   4. Descriptive visualizations (bar plots, heatmap, interactive line chart)
#   5. Machine Learning model training and evaluation
#   6. Trip demand prediction module
#   7. Automatic insights generation
#   8. Downloadable text report
#   9. Footer: "Created with 💙 by Riya Roy"
#
# ==============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import numpy as np

# --------------------------------------------------------------
# 0. STREAMLIT PAGE CONFIGURATION (OPTIONAL BUT NICE)
# --------------------------------------------------------------
# You can customize the page title and icon that appears in the browser tab.
st.set_page_config(
    page_title="Uber Trip Analysis & Prediction Dashboard",
    page_icon="🚖",
    layout="wide",
)

# --------------------------------------------------------------
# 1. SESSION STATE INITIALIZATION
# --------------------------------------------------------------
# We store the key objects (dataframe, model, metrics, etc.) in
# st.session_state so that they persist across pages and reruns.

if "df" not in st.session_state:
    st.session_state.df = None

if "model" not in st.session_state:
    st.session_state.model = None

if "metrics" not in st.session_state:
    st.session_state.metrics = None

if "feature_importance" not in st.session_state:
    st.session_state.feature_importance = None

if "label_encoder" not in st.session_state:
    st.session_state.label_encoder = None

# --------------------------------------------------------------
# 2. SIDEBAR: GLOBAL CONTROLS (THEME + PAGE NAVIGATION)
# --------------------------------------------------------------

st.sidebar.title("⚙ Controls")

# Theme toggle (Light / Dark)
theme_choice = st.sidebar.radio("🌗 Theme", ["Light", "Dark"], index=0)

# Navigation for pages
page_choice = st.sidebar.selectbox(
    "📂 Go to",
    [
        "Home",
        "Dataset",
        "Visualizations",
        "ML Model",
        "Predictions",
        "Insights",
        "About",
    ],
)

# --------------------------------------------------------------
# 3. GLOBAL STYLING (CSS FOR LIGHT & DARK THEMES)
# --------------------------------------------------------------
# We inject raw CSS to override Streamlit's default styling,
# including sidebar, cards, file uploader, text colors, etc.

if theme_choice == "Dark":
    # ---------------------------
    # DARK THEME STYLING
    # ---------------------------
    st.markdown(
        """
        <style>
        /* Main app background and text color */
        .stApp {
            background-color: #1B2430 !important;   /* Deep blue-gray */
            color: #E5E5E5 !important;             /* Soft light text */
            font-family: 'Segoe UI', sans-serif;
        }

        /* Headings */
        h1, h2, h3, h4, h5, h6 {
            color: #FFB3C6 !important;             /* Pastel pink headings */
        }

        /* Generic text */
        p, span, label, li, div {
            color: #E5E5E5 !important;
        }

        /* Sidebar background and text */
        section[data-testid="stSidebar"] {
            background-color: #161A22 !important;
        }
        section[data-testid="stSidebar"] * {
            color: #E5E5E5 !important;
        }

        /* Dataframe background */
        div[data-testid="stDataFrame"] {
            background-color: #2C2C3E !important;
            color: #E5E5E5 !important;
        }

        /* Generic card styling for dark theme */
        .card {
            background-color: #2C2C3E !important;
            padding: 18px;
            border-radius: 12px;
            margin-bottom: 1rem;
            box-shadow: 0 0 12px rgba(0,0,0,0.3);
        }

        /* Footer styling */
        #footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            padding: 8px 0;
            background-color: #1B2430;
            color: #FFB3C6;
            text-align: center;
            font-size: 15px;
            font-weight: 600;
        }
        </style>

        <div id="footer">
            Created with 💙 by Riya Roy
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    # ---------------------------
    # LIGHT THEME STYLING (Palette A)
    # ---------------------------
    st.markdown(
        """
        <style>
        /* Main background and base text color */
        .stApp {
            background-color: #F2EBE3 !important;   /* Pastel beige */
            color: #2B2B2B !important;             /* Dark charcoal text */
            font-family: 'Segoe UI', sans-serif;
        }

        /* Headings (titles, subtitles, etc.) */
        h1, h2, h3, h4, h5, h6 {
            color: #364F6B !important;             /* Pastel navy headings */
        }

        /* Base text everywhere */
        p, span, label, li, div {
            color: #2B2B2B !important;             /* Ensure readability */
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #F7EEDD !important;  /* Soft beige */
        }
        section[data-testid="stSidebar"] * {
            color: #2B2B2B !important;             /* Dark text in sidebar */
            font-weight: 600;
        }

        /* Fix selectbox / dropdown / radio text in sidebar */
        .stSelectbox div, .stRadio label {
            color: #2B2B2B !important;
        }

        /* File uploader styling in light theme */
        div[data-testid="stFileUploader"] > div {
            background-color: #E3F4F4 !important;  /* Soft mint box */
            border-radius: 12px !important;
            padding: 14px !important;
        }
        div[data-testid="stFileUploader"] * {
            color: #2B2B2B !important;             /* Make text visible */
        }

        /* Dataframe background and text */
        div[data-testid="stDataFrame"] {
            background-color: #E3F4F4 !important;  /* Mint background */
            color: #2B2B2B !important;
        }

        /* Generic "card" container styling */
        .card {
            background-color: #E3F4F4 !important;  /* Card background */
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 1rem;
            box-shadow: 0 0 12px rgba(0,0,0,0.1);
        }

        /* Footer bar at bottom for light theme */
        #footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            padding: 8px 0;
            background-color: #F2EBE3;
            color: #2B2B2B;
            text-align: center;
            font-size: 15px;
            font-weight: 600;
        }
        </style>

        <div id="footer">
            Created with 💙 by Riya Roy
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------
# 4. DATA + MODEL PIPELINE
# --------------------------------------------------------------
# This function loads the CSV, processes the data, and trains
# a RandomForestRegressor model to predict Uber trip counts.

def process_and_train(uploaded_file):
    """
    Load Uber CSV file, perform feature engineering, encode categorical
    variables, split data, train the Random Forest model, and compute metrics.
    """
    # 1) Load data
    df = pd.read_csv(uploaded_file)

    # 2) Convert 'date' column to datetime and derive new features
    df["date"] = pd.to_datetime(df["date"])
    df["Day"] = df["date"].dt.day
    df["DayOfWeek"] = df["date"].dt.dayofweek  # 0=Mon ... 6=Sun
    df["Month"] = df["date"].dt.month
    df["Hour"] = df["date"].dt.hour

    # 3) Encode dispatching base number
    le = LabelEncoder()
    df["Base"] = le.fit_transform(df["dispatching_base_number"])

    # 4) Define feature matrix X and target y
    X = df[["Base", "active_vehicles", "Day", "DayOfWeek", "Month"]]
    y = df["trips"]

    # 5) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 6) Model training
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # 7) Predictions on test set
    y_pred = model.predict(X_test)

    # 8) Metrics calculation
    r2 = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    metrics_dict = {
        "R2": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
    }

    # 9) Feature importance
    feature_importance = (
        pd.DataFrame(
            {
                "Feature": X.columns,
                "Importance": model.feature_importances_,
            }
        )
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )

    # Return all useful objects
    return df, model, metrics_dict, feature_importance, le


# --------------------------------------------------------------
# 5. FILE UPLOADER (VISIBLE FROM ALL PAGES)
# --------------------------------------------------------------
# This card and uploader appear at the top, so the user can upload
# the dataset once and then navigate across pages.

st.markdown(
    """
<div class="card">
    <h3>📂 Upload Uber Dataset</h3>
    <p>
        Upload the CSV file (for example, 
        <code>Uber-Jan-Feb-FOIL (1).csv</code>) to enable all analysis,
        visualizations, machine learning predictions, and insights.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("", type="csv")

if uploaded_file is not None:
    # Process and train model as soon as file is uploaded
    df, model, metrics_dict, fi_df, le = process_and_train(uploaded_file)

    # Save into session state for reuse on all pages
    st.session_state.df = df
    st.session_state.model = model
    st.session_state.metrics = metrics_dict
    st.session_state.feature_importance = fi_df
    st.session_state.label_encoder = le

# Convenience variables (read from session state)
df = st.session_state.df
model = st.session_state.model
metrics_dict = st.session_state.metrics
feature_importance = st.session_state.feature_importance
label_encoder = st.session_state.label_encoder


# --------------------------------------------------------------
# Helper: show warning if dataset not loaded
# --------------------------------------------------------------
def need_data_warning():
    st.warning("⚠ Please upload a valid Uber CSV file at the top to use this section.")


# ==============================================================
# 6. PAGES IMPLEMENTATION
# ==============================================================

# --------------------------------------------------------------
# PAGE: HOME
# --------------------------------------------------------------
if page_choice == "Home":

    # Main title for the dashboard
    st.title("🚖 Uber Trip Analysis & Prediction Dashboard")

    # Overview card describing the project
    st.markdown(
        """
<div class="card">
    <h4>🔍 Project Overview</h4>
    <p>
        This dashboard is built to analyze historical Uber trip data and uncover ride demand
        patterns across different days, hours, and base locations. It also integrates a
        machine learning model (Random Forest Regression) to predict the expected number of
        trips based on engineered features such as active vehicles, day of week, and month.
        The interface is designed for interactive exploration and is suitable for academic
        presentations and viva discussions.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    # Quick metrics snapshot if data is available
    if df is not None and metrics_dict is not None:
        st.markdown(
            """
<div class="card">
    <h4>📊 Quick Snapshot of Model & Data</h4>
    <p>
        The metrics below provide a quick overview of the model performance and dataset size.
    </p>
</div>
""",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² Score", f"{metrics_dict['R2']:.3f}")
        c2.metric("MAE", f"{metrics_dict['MAE']:.0f}")
        c3.metric("RMSE", f"{metrics_dict['RMSE']:.0f}")
        c4.metric("Rows in Dataset", f"{len(df)}")

        st.write(
            """
A higher R² score indicates that the model explains a large proportion of the variance
in trip counts. Lower MAE and RMSE values mean that the prediction errors are relatively small.
"""
        )
    else:
        need_data_warning()

# --------------------------------------------------------------
# PAGE: DATASET
# --------------------------------------------------------------
elif page_choice == "Dataset":
    st.header("📁 Dataset View")

    if df is None:
        need_data_warning()
    else:
        # Preview of the top rows
        st.subheader("🔹 Data Preview (First 20 Rows)")
        st.dataframe(df.head(20))
        st.caption(
            """
This preview allows a quick check of the structure of the processed dataset,
including engineered columns such as `Day`, `DayOfWeek`, `Month`, and `Hour`.
"""
        )

        # Column information
        st.subheader("🔹 Column Information")
        col_info = pd.DataFrame(
            {
                "Column": df.columns,
                "Data Type": [str(t) for t in df.dtypes],
                "Missing Values": df.isna().sum().values,
            }
        )
        st.dataframe(col_info)
        st.caption(
            """
This summary describes the data type and number of missing values for each column.
It is useful for assessing data quality before modeling.
"""
        )

        # Statistical summary
        st.subheader("🔹 Statistical Summary")
        st.dataframe(df.describe())
        st.caption(
            """
Basic descriptive statistics such as mean, minimum, maximum, and quartiles give
an idea of the distribution and scale of numerical variables like `trips` and
`active_vehicles`.
"""
        )

        # Downloadable text report
        if metrics_dict is not None and feature_importance is not None:
            report_lines = []
            report_lines.append("UBER TRIP ANALYSIS REPORT\n")
            report_lines.append("-------------------------------------\n")
            report_lines.append(f"Total Rows: {len(df)}\n\n")
            report_lines.append("MODEL METRICS\n")
            report_lines.append(f"R² Score: {metrics_dict['R2']:.4f}\n")
            report_lines.append(f"MAE:      {metrics_dict['MAE']:.2f}\n")
            report_lines.append(f"MSE:      {metrics_dict['MSE']:.2f}\n")
            report_lines.append(f"RMSE:     {metrics_dict['RMSE']:.2f}\n\n")
            report_lines.append("FEATURE IMPORTANCE:\n")
            for _, row in feature_importance.iterrows():
                report_lines.append(f"{row['Feature']}: {row['Importance']:.4f}\n")

            report_text = "".join(report_lines)

            st.download_button(
                "⬇ Download Text Report",
                data=report_text,
                file_name="uber_analysis_report.txt",
                mime="text/plain",
            )
            st.caption(
                "This text report can be attached as an appendix to your project documentation."
            )

# --------------------------------------------------------------
# PAGE: VISUALIZATIONS
# --------------------------------------------------------------
elif page_choice == "Visualizations":

    st.header("📊 Visualizations")

    if df is None:
        need_data_warning()
    else:
        # --- Trips by Hour of the Day ---
        st.subheader("⏰ Total Trips by Hour of the Day")
        fig1, ax1 = plt.subplots()
        sns.barplot(x="Hour", y="trips", data=df, estimator=sum, errorbar=None, ax=ax1)
        ax1.set_xlabel("Hour of the Day")
        ax1.set_ylabel("Total Trips")
        ax1.set_title("Total Uber Trips by Hour of the Day")
        st.pyplot(fig1)
        st.caption(
            """
This bar chart shows how the total number of Uber trips varies by hour. It helps identify
which times of day experience the highest demand, such as morning and evening rush hours.
"""
        )

        # --- Trips by Day of Week ---
        st.subheader("📅 Total Trips by Day of the Week")
        fig2, ax2 = plt.subplots()
        sns.barplot(
            x="DayOfWeek",
            y="trips",
            data=df,
            estimator=sum,
            errorbar=None,
            ax=ax2,
        )
        ax2.set_xlabel("Day of the Week (0 = Monday, 6 = Sunday)")
        ax2.set_ylabel("Total Trips")
        ax2.set_title("Total Uber Trips by Day of the Week")
        st.pyplot(fig2)
        st.caption(
            """
This chart aggregates trip counts across each day of the week. It helps identify which
days are typically busiest and is useful for weekly planning and staffing decisions.
"""
        )

        # --- Heatmap of Trips by Hour vs DayOfWeek ---
        st.subheader("🔥 Heatmap: Trips by Hour vs Day of Week")
        pivot = df.pivot_table(
            values="trips",
            index="DayOfWeek",
            columns="Hour",
            aggfunc="sum",
            fill_value=0,
        )
        fig3, ax3 = plt.subplots()
        sns.heatmap(pivot, cmap="YlGnBu", ax=ax3)
        ax3.set_xlabel("Hour of the Day")
        ax3.set_ylabel("Day of the Week (0 = Monday)")
        ax3.set_title("Heatmap of Trips by Hour and Day of Week")
        st.pyplot(fig3)
        st.caption(
            """
The heatmap combines day of week and hour of day into a single visualization, highlighting
demand hotspots. Darker cells represent periods with higher trip volumes.
"""
        )

        # --- Interactive line chart using Plotly ---
        st.subheader("🌈 Interactive Trend: Trips Over Time by Base")
        df_sorted = df.sort_values("date")
        fig4 = px.line(
            df_sorted,
            x="date",
            y="trips",
            color="dispatching_base_number",
            title="Trips Over Time by Dispatching Base",
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.caption(
            """
This interactive line chart shows how trip volumes change over time, separated by base.
You can hover, zoom, and pan to inspect specific periods or compare demand across bases.
"""
        )

# --------------------------------------------------------------
# PAGE: ML MODEL
# --------------------------------------------------------------
elif page_choice == "ML Model":

    st.header("🤖 Machine Learning Model Details")

    if df is None or model is None or metrics_dict is None or feature_importance is None:
        need_data_warning()
    else:
        st.subheader("📌 Model Used: Random Forest Regressor")
        st.write(
            """
The Random Forest Regressor is an ensemble learning algorithm that builds multiple
decision trees and averages their predictions. This typically improves accuracy and
reduces overfitting compared to using a single decision tree.
"""
        )

        st.subheader("📊 Model Performance Metrics")
        st.write(f"**R² Score:** {metrics_dict['R2']:.4f}")
        st.write(f"**Mean Absolute Error (MAE):** {metrics_dict['MAE']:.2f}")
        st.write(f"**Mean Squared Error (MSE):** {metrics_dict['MSE']:.2f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {metrics_dict['RMSE']:.2f}")
        st.caption(
            """
These metrics assess how well the model predicts the number of trips. A higher R² score and
lower error values indicate a better-performing model.
"""
        )

        st.subheader("⭐ Feature Importance")
        fig_fi, ax_fi = plt.subplots()
        sns.barplot(
            x="Importance",
            y="Feature",
            data=feature_importance,
            errorbar=None,
            ax=ax_fi,
        )
        ax_fi.set_xlabel("Importance Score")
        ax_fi.set_ylabel("Feature")
        ax_fi.set_title("Feature Importance for Trip Prediction")
        st.pyplot(fig_fi)
        st.caption(
            """
Feature importance scores show how much each input variable contributes to the model’s predictions.
This helps explain the model and identify which features (such as `DayOfWeek` or `active_vehicles`)
are most influential.
"""
        )

# --------------------------------------------------------------
# PAGE: PREDICTIONS
# --------------------------------------------------------------
elif page_choice == "Predictions":

    st.header("📈 Trip Demand Prediction")

    if df is None or model is None or label_encoder is None:
        need_data_warning()
    else:
        st.write(
            """
Use this section to simulate expected trip demand by specifying a base,
a date, and the number of active vehicles. The trained model will then
estimate the number of Uber trips.
"""
        )

        # Input widgets
        base_options = sorted(df["dispatching_base_number"].unique())
        selected_base = st.selectbox("Select Dispatching Base", base_options)

        selected_date = st.date_input("Select Date")
        active_vehicles = st.number_input(
            "Number of Active Vehicles",
            min_value=0,
            step=1,
            value=10,
        )

        # Convert selected_date into features
        date_obj = pd.to_datetime(selected_date)
        day = date_obj.day
        day_of_week = date_obj.dayofweek
        month = date_obj.month

        if st.button("🔮 Predict Trips"):
            # Encode base string into numeric code
            base_encoded = label_encoder.transform([selected_base])[0]

            # Build input row for prediction
            input_data = pd.DataFrame(
                {
                    "Base": [base_encoded],
                    "active_vehicles": [active_vehicles],
                    "Day": [day],
                    "DayOfWeek": [day_of_week],
                    "Month": [month],
                }
            )

            # Perform prediction
            predicted_trips = model.predict(input_data)[0]

            st.success(
                f"Estimated number of trips for {selected_base} on {selected_date} "
                f"with {active_vehicles} active vehicles: "
                f"**{predicted_trips:.0f} trips**"
            )
            st.caption(
                """
This prediction is based on historical patterns learned from the dataset.
It serves as an approximate estimate of demand and can support operational planning.
"""
            )

# --------------------------------------------------------------
# PAGE: INSIGHTS
# --------------------------------------------------------------
elif page_choice == "Insights":

    st.header("🧠 Data Insights & Observations")

    if df is None:
        need_data_warning()
    else:
        st.markdown(
            """
<div class="card">
    <h4>📌 Automatically Generated Insights</h4>
    <p>
        The bullet points below summarize some key observations extracted from the dataset.
        You can directly use these insights in your viva or project report.
    </p>
</div>
""",
            unsafe_allow_html=True,
        )

        # 1) Top day of week in terms of total trips
        dow_totals = df.groupby("DayOfWeek")["trips"].sum()
        top_dow = int(dow_totals.idxmax())
        top_dow_value = int(dow_totals.max())

        # 2) Top hour of day
        hour_totals = df.groupby("Hour")["trips"].sum()
        top_hour = int(hour_totals.idxmax())
        top_hour_value = int(hour_totals.max())

        # 3) Correlation between active_vehicles and trips
        corr_av_trips = df["active_vehicles"].corr(df["trips"])

        st.subheader("1️⃣ Weekly Demand Pattern")
        st.write(
            f"- The highest total trips occur on **DayOfWeek = {top_dow}** "
            f"(0 = Monday, 6 = Sunday), with approximately **{top_dow_value} trips**."
        )
        st.write(
            "- This suggests that on this day, Uber usage is consistently higher, possibly "
            "due to recurring work schedules, social events, or typical travel behavior."
        )

        st.subheader("2️⃣ Peak Hour Analysis")
        st.write(
            f"- The busiest time of day appears to be **Hour = {top_hour}:00**, "
            f"with about **{top_hour_value} trips** recorded at that hour."
        )
        st.write(
            "- This time window may correspond to rush hours or key commuting periods. "
            "Allocating additional drivers during this hour can help meet demand efficiently."
        )

        st.subheader("3️⃣ Relationship Between Active Vehicles and Trips")
        st.write(
            f"- The correlation between **active_vehicles** and **trips** is "
            f"**{corr_av_trips:.3f}**."
        )
        if corr_av_trips > 0.5:
            st.write(
                "- This indicates a strong positive relationship: as more vehicles are "
                "active, the number of trips tends to increase significantly."
            )
        elif corr_av_trips > 0.2:
            st.write(
                "- This indicates a moderate positive relationship: vehicle availability "
                "does influence trips, but other factors also play an important role."
            )
        else:
            st.write(
                "- This indicates a weak relationship: increasing active vehicles alone "
                "may not guarantee more trips; demand is likely influenced by time, "
                "location, and external conditions."
            )

        st.subheader("4️⃣ Operational Recommendation")
        st.write(
            f"- Based on the peak day (**DayOfWeek = {top_dow}**) and peak hour "
            f"(**{top_hour}:00**), Uber could prioritize driver availability during "
            "these specific time windows."
        )
        st.write(
            "- Aligning supply (drivers) with high-demand periods can reduce passenger "
            "waiting times and increase driver earnings and platform efficiency."
        )

# --------------------------------------------------------------
# PAGE: ABOUT
# --------------------------------------------------------------
elif page_choice == "About":

    st.header(" About This Project")

    st.subheader("👤 Developed by Riya Roy")
    st.write(
        """
This Uber Trip Analysis & Prediction Dashboard was developed by me as part of my 
internship project under the Edunet Foundation, co-partnered with Microsoft. 
The project focuses on applying data analysis, visualization, and machine 
learning techniques to understand and forecast Uber trip demand patterns.

The dashboard showcases the complete workflow of a real-world data analytics project. 
It includes:

• **Data Ingestion & Cleaning** – Uploading and preprocessing the raw Uber dataset.  
• **Feature Engineering** – Creating additional insights such as day, month, hour, and 
  day-of-week patterns from the timestamp data.  
• **Interactive Visualizations** – Using bar charts, heatmaps, trend lines, and other 
  graphical tools to explore trip demand across time periods and base locations.  
• **Machine Learning Model** – Training a Random Forest Regression model to predict 
  trip volumes based on engineered features.  
• **Prediction Interface** – Allowing users to input base location, date, and active 
  vehicles to estimate future trip counts.  
• **Insights & Data Storytelling** – Automatically summarizing important patterns such 
  as peak hours, busiest days, and demand correlations.

This project reflects practical experience in data science and dashboard development, 
aligning with real-industry applications supported by Edunet Foundation and Microsoft.

"""
    )

    st.write(
        """
The project follows good software engineering practices including:
- Use of virtual environments (venv) and requirements management.
- Separation of logic into data processing, modeling, and visualization sections.
- Reusability via Streamlit's session state and multi-page layout.
"""
    )
