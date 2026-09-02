# 🌬️ Air Quality Index (AQI) Analysis & ML Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fa1-activity-project-aqi-analysis.streamlit.app/)

> 🚀 **Live Demo**: [fa1-activity-project-aqi-analysis.streamlit.app](https://fa1-activity-project-aqi-analysis.streamlit.app/)

An end-to-end Air Quality Index (AQI) data analysis, exploratory data visualization, and machine learning classification project built using Python, Scikit-Learn, and Streamlit.

---

## 📌 Project Highlights

- **Data Standardization & Cleaning**:
  - Automatically handles column aliases (e.g., `pm25` &rarr; `PM2_5`, `city` &rarr; `City`, `aqi_category` &rarr; `AQI_Category`).
  - Cleans missing records and parses date fields into standardized temporal features.

- **Statistical Analysis & Measures of Central Tendency**:
  - Parametric and non-parametric summary statistics: Mean, Median, Mode, Standard Deviation, Variance, Skewness, Kurtosis, Q1, Q3, and IQR.
  - Chi-Square test of independence between AQI category and calendar years.

- **Exploratory Data Analysis (EDA)**:
  - **Category Distributions**: Bar and Pie charts displaying proportion and counts across AQI categories (`Good`, `Satisfactory`, `Moderate`, `Poor`, `Very Poor`, `Severe`).
  - **Pollutant Levels**: Average concentration rankings across major pollutants ($PM_{2.5}, PM_{10}, NO_2, SO_2, CO, O_3$) and their correlation with overall AQI.
  - **Distribution & KDE**: Histograms and Kernel Density Estimation curves showcasing concentration spreads against Mean and Median baselines.
  - **Correlation Heatmap**: Pearson correlation matrix analyzing interplay between pollutants, time components, and AQI.

- **Feature Engineering & Preprocessing**:
  - Cyclical date transformations using sine and cosine components (`sin_month`, `cos_month`, `sin_dayofyear`, `cos_dayofyear`) to capture seasonality.
  - Calendar markers: `Year`, `Month`, `Day`, `DayOfWeek`, `DayOfYear`, `Is_Weekend`, and `Season`.
  - Feature pipelines using `StandardScaler` for numeric columns and `OneHotEncoder` for categorical city values.

- **Machine Learning Classification**:
  - Honest chronological train-test split (2015–2022 train set vs. 2023 unseen test set) to prevent data leakage.
  - Benchmark evaluation of classification algorithms:
    - **Logistic Regression**
    - **Decision Tree Classifier**
    - **Random Forest Classifier**
  - Performance assessment using Accuracy, Confusion Matrix, and detailed Classification Reports (Precision, Recall, F1-Score).

- **Real-Time Interactive Web App**:
  - Built with **Streamlit** for real-time inference.
  - Input city, date, and individual pollutant levels to receive predicted AQI Category, confidence probability breakdown, and estimated numerical AQI index.

---

## 🗂️ Directory Structure

```text
aqi-project/
├── app.py                         # Streamlit interactive web application
├── 125M1H002_FA1_Activity.ipynb   # Jupyter notebook with complete EDA & ML pipeline
├── data/
│   └── aqi_data.csv               # Air Quality historical dataset
├── models/                        # Serialized ML pipeline artifacts (.pkl)
├── .gitignore                     # Ignored cache, checkpoints, and environment files
└── README.md                      # Project documentation
```

---

## 🚀 Getting Started

### 1. Install Dependencies

From the repository root (or project folder), install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook

Open and execute the analysis notebook:

```bash
jupyter notebook 125M1H002_FA1_Activity.ipynb
```

### 3. Launch the Streamlit Web Application

Access the deployed application directly:
👉 **[Open Live Streamlit App](https://fa1-activity-project-aqi-analysis.streamlit.app/)**

Or launch locally:

```bash
streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 🛠️ Built With

- **Language**: Python 3
- **Data Wrangling**: Pandas, NumPy
- **Visualizations**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-Learn, SciPy
- **Web Deployment**: Streamlit
