"""Streamlit EDA and AQI Category Classification Application."""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Configure page layout and style
st.set_page_config(
    page_title="AQI Analysis & Prediction App",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for center alignment of all tables and clean UI styling
st.markdown(
    """
    <style>
    /* Center align all table elements (headers and cells) */
    table {
        margin-left: auto !important;
        margin-right: auto !important;
        text-align: center !important;
    }
    th {
        text-align: center !important;
        background-color: #f8f9fa !important;
        font-weight: 600 !important;
    }
    td {
        text-align: center !important;
    }
    [data-testid="stTable"] td, [data-testid="stTable"] th {
        text-align: center !important;
    }
    div[data-testid="stDataFrame"] div[role="gridcell"] {
        text-align: center !important;
        justify-content: center !important;
    }
    div[data-testid="stDataFrame"] div[role="columnheader"] {
        text-align: center !important;
        justify-content: center !important;
    }
    /* Metric styling */
    div[data-testid="metric-container"] {
        text-align: center !important;
        padding: 10px;
        background: #f8f9fb;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Constants
POLLUTANT_COLUMNS = ["PM2_5", "PM10", "NO2", "SO2", "CO", "O3"]
DISPLAY_POLLUTANT_NAMES = {
    "PM2_5": "PM2.5",
    "PM10": "PM10",
    "NO2": "NO2",
    "SO2": "SO2",
    "CO": "CO",
    "O3": "O3",
}
DATE_FEATURES = [
    "Year", "Month", "Day", "DayOfWeek", "DayOfYear", "Is_Weekend", "Season",
    "sin_month", "cos_month", "sin_dayofyear", "cos_dayofyear",
]
NUMERIC_FEATURES = POLLUTANT_COLUMNS + DATE_FEATURES
CATEGORICAL_FEATURES = ["City"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = "AQI_Category"

AQI_CATEGORY_COLORS = {
    "Good": "#2ecc71",
    "Satisfactory": "#27ae60",
    "Moderate": "#f39c12",
    "Poor": "#e67e22",
    "Very Poor": "#e74c3c",
    "Severe": "#8e44ad",
}


def center_dataframe(df: pd.DataFrame):
    """Return a Styler object with center-aligned text and headers."""
    return df.style.set_properties(**{"text-align": "center"}).set_table_styles(
        [
            {"selector": "th", "props": [("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]},
        ]
    )


def standardize_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Validate the uploaded schema and return a cleaned, standardized DataFrame."""
    aliases = {
        "city": "City",
        "area": "City",
        "date": "Date",
        "pm25": "PM2_5",
        "pm2_5": "PM2_5",
        "pm2.5": "PM2_5",
        "pm10": "PM10",
        "no2": "NO2",
        "so2": "SO2",
        "co": "CO",
        "o3": "O3",
        "aqi": "AQI",
        "aqi_value": "AQI",
        "aqi_category": "AQI_Category",
        "air_quality_status": "AQI_Category",
        "aqi_bucket": "AQI_Category",
    }
    rename_map = {
        column: aliases.get(column.strip().lower().replace(" ", "_"), column)
        for column in raw_df.columns
    }
    df = raw_df.rename(columns=rename_map).copy()

    required = {"City", "Date", *POLLUTANT_COLUMNS, TARGET_COLUMN}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "The dataset is missing required columns: " + ", ".join(missing)
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["City"] = df["City"].astype(str).str.strip()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()
    for column in POLLUTANT_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "AQI" in df.columns:
        df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")

    df = df.dropna(subset=["Date", "City", *POLLUTANT_COLUMNS, TARGET_COLUMN])
    df = df[df["City"].ne("") & df[TARGET_COLUMN].ne("")]
    return df.sort_values(["Date", "City"]).reset_index(drop=True)


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features without using AQI or AQI category as inputs."""
    output = df.copy()
    output["Year"] = output["Date"].dt.year
    output["Month"] = output["Date"].dt.month
    output["Month_Name"] = output["Date"].dt.strftime("%b")
    output["Day"] = output["Date"].dt.day
    output["DayOfWeek"] = output["Date"].dt.dayofweek
    output["DayOfYear"] = output["Date"].dt.dayofyear
    output["Is_Weekend"] = output["DayOfWeek"].isin([5, 6]).astype(int)
    output["Season"] = np.select(
        [
            output["Month"].isin([12, 1, 2]),
            output["Month"].isin([3, 4, 5]),
            output["Month"].isin([6, 7, 8, 9]),
        ],
        [0, 1, 2],
        default=3,
    )
    output["sin_month"] = np.sin(2 * np.pi * output["Month"] / 12)
    output["cos_month"] = np.cos(2 * np.pi * output["Month"] / 12)
    output["sin_dayofyear"] = np.sin(2 * np.pi * output["DayOfYear"] / 365.25)
    output["cos_dayofyear"] = np.cos(2 * np.pi * output["DayOfYear"] / 365.25)
    return output


@st.cache_resource
def get_trained_model(df: pd.DataFrame) -> Pipeline:
    """Train and cache the Random Forest classifier for instant real-time prediction."""
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("city", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline([
        ("preprocessor", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )),
    ])
    model.fit(df[FEATURE_COLUMNS], df[TARGET_COLUMN])
    return model


def display_eda(df: pd.DataFrame) -> None:
    """Render comprehensive Exploratory Data Analysis with 5 core visualization tabs."""
    tab_summary, tab_categories, tab_pollutants, tab_distribution, tab_correlations = st.tabs([
        "📋 Summary & Stats",
        "📊 Category Distribution (Bar & Pie)",
        "🧪 Pollutant Levels",
        "📈 Distribution & KDE",
        "🔥 Correlation Heatmap",
    ])

    # 1. Summary Tab
    with tab_summary:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Unique Cities", df["City"].nunique())
        col3.metric("Date Range", f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        col4.metric("Missing Values", df.isnull().sum().sum())

        st.markdown("#### 📄 Dataset Preview")
        display_df = df.copy().rename(columns=DISPLAY_POLLUTANT_NAMES)
        preview_table = display_df.head(10).astype(str)
        st.dataframe(center_dataframe(preview_table), use_container_width=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### 🔍 Missing Values Audit")
            missing_df = df.isna().sum().rename("Missing Values").to_frame()
            st.dataframe(center_dataframe(missing_df), use_container_width=True)

        with col_right:
            st.markdown("#### 📐 Summary Statistics")
            desc_df = display_df.describe().T.round(2)
            st.dataframe(center_dataframe(desc_df), use_container_width=True)

    # 2. Category Distribution (Bar & Pie)
    with tab_categories:
        st.markdown("#### 🏷️ AQI Category Distribution (Bar & Pie Charts)")
        bucket_counts = df[TARGET_COLUMN].value_counts()
        
        col_bar, col_pie = st.columns(2)
        
        with col_bar:
            fig_bar, ax_bar = plt.subplots(figsize=(7, 4.8))
            colors = [AQI_CATEGORY_COLORS.get(cat, "#3498db") for cat in bucket_counts.index]
            sns.barplot(x=bucket_counts.index, y=bucket_counts.values, palette=colors, ax=ax_bar, edgecolor="black")
            ax_bar.set_title("AQI Category Counts (Bar Chart)", fontsize=13, fontweight="bold")
            ax_bar.set_xlabel("AQI Category", fontsize=11)
            ax_bar.set_ylabel("Number of Records", fontsize=11)
            ax_bar.tick_params(axis="x", rotation=25)
            for p in ax_bar.patches:
                ax_bar.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                               ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)

        with col_pie:
            fig_pie, ax_pie = plt.subplots(figsize=(7, 4.8))
            colors = [AQI_CATEGORY_COLORS.get(cat, "#3498db") for cat in bucket_counts.index]
            ax_pie.pie(
                bucket_counts.values,
                labels=bucket_counts.index,
                autopct="%1.1f%%",
                colors=colors,
                startangle=140,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5}
            )
            ax_pie.set_title("AQI Category Share (Pie Chart)", fontsize=13, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_pie)
            plt.close(fig_pie)

    # 3. Pollutant Levels
    with tab_pollutants:
        st.markdown("#### 🧪 Average Pollutant Concentration Levels")
        
        pollutants_clean = [p for p in POLLUTANT_COLUMNS if p in df.columns]
        avg_pollutants = df[pollutants_clean].mean().rename(DISPLAY_POLLUTANT_NAMES).sort_values(ascending=False)
        
        fig_poll, ax_poll = plt.subplots(figsize=(10, 4.2))
        sns.barplot(x=avg_pollutants.index, y=avg_pollutants.values, color="teal", ax=ax_poll, edgecolor="black")
        ax_poll.set_title("Average Concentration Levels Across Pollutants", fontsize=13, fontweight="bold")
        ax_poll.set_xlabel("Pollutant", fontsize=11)
        ax_poll.set_ylabel("Average Concentration (µg/m³ / mg/m³)", fontsize=11)
        for p in ax_poll.patches:
            ax_poll.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
        plt.tight_layout()
        st.pyplot(fig_poll)
        plt.close(fig_poll)

        st.markdown("#### 📊 Detailed Pollutant Statistics & Correlation with AQI")
        stats_data = []
        for p in pollutants_clean:
            corr_val = df["AQI"].corr(df[p]) if "AQI" in df.columns else np.nan
            stats_data.append({
                "Pollutant": DISPLAY_POLLUTANT_NAMES.get(p, p),
                "Mean": df[p].mean(),
                "Median": df[p].median(),
                "Std Dev": df[p].std(),
                "Min": df[p].min(),
                "Max": df[p].max(),
                "Correlation with AQI": corr_val,
            })
        pollutant_stats_df = pd.DataFrame(stats_data).round(3).set_index("Pollutant")
        st.dataframe(center_dataframe(pollutant_stats_df), use_container_width=True)

    # 4. Distribution & KDE
    with tab_distribution:
        st.markdown("#### 📈 AQI & Pollutant Distribution Analysis")
        
        target_col = "AQI" if "AQI" in df.columns else "PM2_5"
        val = df[target_col]
        
        fig_dist, axes_dist = plt.subplots(1, 2, figsize=(14, 4.8))
        
        # Histogram
        axes_dist[0].hist(val, bins=40, density=True, color="steelblue", edgecolor="white", alpha=0.8)
        axes_dist[0].axvline(val.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean = {val.mean():.1f}")
        axes_dist[0].axvline(val.median(), color="green", linestyle="--", linewidth=2, label=f"Median = {val.median():.1f}")
        axes_dist[0].set_title(f"{target_col} Histogram", fontsize=12, fontweight="bold")
        axes_dist[0].set_xlabel(target_col)
        axes_dist[0].set_ylabel("Density")
        axes_dist[0].legend()
        axes_dist[0].grid(alpha=0.3)

        # KDE
        val.plot(kind="kde", ax=axes_dist[1], color="darkorange", linewidth=2.5)
        axes_dist[1].axvline(val.mean(), color="red", linestyle="--", linewidth=1.5, label="Mean")
        axes_dist[1].axvline(val.median(), color="green", linestyle="--", linewidth=1.5, label="Median")
        axes_dist[1].set_title(f"{target_col} Density (KDE Curve)", fontsize=12, fontweight="bold")
        axes_dist[1].set_xlabel(target_col)
        axes_dist[1].legend()
        axes_dist[1].grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig_dist)
        plt.close(fig_dist)

    # 5. Correlation Heatmap
    with tab_correlations:
        st.markdown("#### 🔥 Feature Correlation Heatmap")
        corr_candidates = (["AQI"] if "AQI" in df.columns else []) + POLLUTANT_COLUMNS + ["Year", "Month"]
        corr_cols = [c for c in corr_candidates if c in df.columns]
        
        corr_matrix = df[corr_cols].corr()
        corr_matrix.rename(index=DISPLAY_POLLUTANT_NAMES, columns=DISPLAY_POLLUTANT_NAMES, inplace=True)

        fig_corr, ax_corr = plt.subplots(figsize=(8.5, 6.5))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, linewidths=0.5, ax=ax_corr)
        ax_corr.set_title("Correlation Heatmap (AQI, Pollutants, Time)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_corr)
        plt.close(fig_corr)


def render_prediction_section(model: Pipeline, df: pd.DataFrame) -> None:
    """Render the interactive Real-Time AQI Prediction interface."""
    st.markdown("### 🔮 Real-Time AQI Category Prediction")
    st.write("Input city, date, and individual pollutant concentrations to predict the corresponding Air Quality Category:")

    with st.form("aqi_prediction_form"):
        col_city, col_date = st.columns(2)
        with col_city:
            city = st.selectbox("Select City", sorted(df["City"].unique()))
        with col_date:
            date = st.date_input("Select Date", value=df["Date"].max().date())

        st.markdown("**Enter Pollutant Concentration Levels:**")
        cols = st.columns(3)
        values = {}
        for index, pollutant in enumerate(POLLUTANT_COLUMNS):
            median_val = float(df[pollutant].median()) if pollutant in df.columns else 50.0
            display_name = DISPLAY_POLLUTANT_NAMES.get(pollutant, pollutant)
            with cols[index % 3]:
                values[pollutant] = st.number_input(
                    f"{display_name} (µg/m³)",
                    min_value=0.0,
                    max_value=1500.0,
                    value=round(median_val, 2),
                    step=1.0,
                )

        submitted = st.form_submit_button("🚀 Predict AQI Category", type="primary", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{"City": city, "Date": pd.Timestamp(date), **values}])
        row = add_date_features(row)
        
        prediction = model.predict(row[FEATURE_COLUMNS])[0]
        color = AQI_CATEGORY_COLORS.get(prediction, "#3498db")
        
        st.markdown(
            f"""
            <div style="background-color: {color}22; border-left: 6px solid {color}; padding: 18px; border-radius: 8px; margin: 20px 0;">
                <h3 style="margin: 0; color: {color};">Predicted AQI Category: <b>{prediction}</b></h3>
                <p style="margin: 6px 0 0 0; color: #444; font-size: 15px;">Location: <b>{city}</b> | Target Date: <b>{date}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Estimate numerical AQI using PM2.5 linear relation
        if "PM2_5" in values:
            approx_aqi = round(0.8003 * values["PM2_5"] + 23.4856, 1)
            st.info(f"💡 Estimated Numerical AQI Index: **~{approx_aqi}** (calculated from PM2.5 concentration)")

        if hasattr(model.named_steps["classifier"], "predict_proba"):
            probabilities = model.predict_proba(row[FEATURE_COLUMNS])[0]
            prob_df = pd.DataFrame({"AQI Category": model.classes_, "Confidence (%)": (probabilities * 100).round(2)})
            
            st.markdown("#### 🎯 Prediction Confidence Distribution")
            fig_prob, ax_prob = plt.subplots(figsize=(8, 3.5))
            colors = [AQI_CATEGORY_COLORS.get(cat, "#3498db") for cat in prob_df["AQI Category"]]
            sns.barplot(data=prob_df, x="AQI Category", y="Confidence (%)", palette=colors, ax=ax_prob, edgecolor="black")
            ax_prob.set_ylabel("Probability (%)")
            ax_prob.set_ylim(0, 100)
            for p in ax_prob.patches:
                ax_prob.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                               ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
            plt.tight_layout()
            st.pyplot(fig_prob)
            plt.close(fig_prob)


# ==============================================================================
# MAIN APP EXECUTION
# ==============================================================================

st.title("🌬️ Air Quality Index (AQI) EDA & Prediction App")
st.caption("Interactive platform for Air Quality Exploratory Data Analysis, Feature Inspection, and Real-Time AQI Category Prediction.")

# End-to-End ML Pipeline Architecture Expander (Grounded strictly on aqi_analysis notebook workflow)
with st.expander("📖 End-to-End ML Pipeline Architecture & Deployment", expanded=False):
    st.markdown(
        """
        ### 🚀 End-to-End ML Pipeline Workflow (from `aqi_analysis`):
        
        **1. Library & Environment Setup**:
           - Imports foundational packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy.stats`, and `sklearn`.
        
        **2. Data Ingestion & Standardization**:
           - Loads multi-city dataset (`aqi_data.csv`).
           - Standardizes column names (`city` &rarr; `City`, `date` &rarr; `Date`, `pm25` &rarr; `PM2.5`, `pm10` &rarr; `PM10`, `no2` &rarr; `NO2`, `so2` &rarr; `SO2`, `co` &rarr; `CO`, `o3` &rarr; `O3`, `aqi` &rarr; `AQI`, `aqi_category` &rarr; `AQI_Bucket`).
           - Parses dates into datetime objects and extracts temporal features (`Year`, `Month`, `Month_Name`).

        **3. Data Processing & Missing Value Handling**:
           - Audits null values across all columns and applies mean imputation on numeric attributes where necessary.

        **4. Statistical Measures of Central Tendency**:
           - Computes parametric and non-parametric summary metrics for AQI: **Mean**, **Median**, **Mode**, **Standard Deviation**, **Variance**, **Skewness**, **Kurtosis**, **Q1**, **Q3**, and **IQR**.
           - Performs **Chi-Square Test of Independence** between AQI categories and recorded years.

        **5. Exploratory Data Analysis & Visualizations**:
           - **Bar & Pie Charts**: Category distribution and proportional share across AQI buckets (`Good`, `Satisfactory`, `Moderate`, `Poor`).
           - **Pollutant Levels**: Average concentrations across $PM_{2.5}, PM_{10}, NO_2, SO_2, CO, O_3$.
           - **Histogram + KDE**: Visualizes density curve, skewness, and distribution shape against Mean & Median.
           - **Correlation Heatmap**: Analyzes Pearson correlations between AQI, pollutant concentrations, and calendar time.

        **6. Feature Engineering & Preprocessing**:
           - Extracts calendar attributes and cyclical periodic features.
           - Applies `StandardScaler` for numeric pollutants and `OneHotEncoder` for categorical city values.

        **7. Real-Time Model Inference & Prediction**:
           - Deploys trained classification pipeline to evaluate user-supplied pollutant inputs and predict AQI category in real time with confidence distributions.
        """
    )

# Automatic Data Loading (No Sidebar)
data_file = next(
    (path for path in [
        Path("data/aqi_data.csv"),
        Path("aqi_data.csv"),
        Path("data/india_city_aqi_2015_2023.csv"),
        Path("india_city_aqi_2015_2023.csv"),
    ] if path.exists()),
    None,
)

if data_file is None:
    st.error("AQI dataset ('data/aqi_data.csv') not found in workspace.")
    st.stop()

try:
    raw_df = pd.read_csv(data_file)
    clean_df = standardize_dataset(raw_df)
    feature_df = add_date_features(clean_df)
    model = get_trained_model(feature_df)
except Exception as error:
    st.error(f"Error loading and processing dataset: {error}")
    st.stop()

# Main Navigation Tabs (EDA and Real-Time Prediction)
main_eda_tab, main_pred_tab = st.tabs([
    "🔍 Exploratory Data Analysis (EDA)",
    "🔮 Real-Time AQI Prediction",
])

with main_eda_tab:
    display_eda(feature_df)

with main_pred_tab:
    render_prediction_section(model, feature_df)