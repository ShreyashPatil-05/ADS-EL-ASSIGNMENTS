import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# ─── Page Config (must be first) ───────────────────────────────────────────────
st.set_page_config(
    page_title="ADS Assignment 2",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS: center-align all text ────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Titles, headers, subheaders */
        h1, h2, h3, h4, h5, h6 { text-align: center !important; }

        /* Body / paragraph text */
        p, .stMarkdown p { text-align: center !important; }

        /* Metric container — center everything inside */
        [data-testid="stMetric"] {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            text-align: center !important;
        }
        [data-testid="stMetricLabel"] > div,
        [data-testid="stMetricValue"] > div,
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {
            text-align: center !important;
            justify-content: center !important;
            width: 100% !important;
        }

        /* Caption / small text */
        .stCaption, small { text-align: center !important; }

        /* Write / plain text blocks */
        .stText { text-align: center !important; }

        /* Dataframe / st.dataframe cells and headers */
        [data-testid="stDataFrame"] td,
        [data-testid="stDataFrame"] th,
        [data-testid="stDataFrame"] .dvn-cell,
        [data-testid="stDataFrame"] .dvn-header,
        .stDataFrame td,
        .stDataFrame th { text-align: center !important; justify-content: center !important; }

        /* st.table (static table) */
        table td, table th { text-align: center !important; }

        /* Glide data grid (ag-grid style cells inside dataframe) */
        .gdg-cell { justify-content: center !important; text-align: center !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Helper: resolve CSV paths relative to this file ──────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def csv(name):
    return os.path.join(BASE_DIR, name)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 ADS Assignment 2")
    st.markdown("*Advanced Data Science Lab*")
    st.markdown("---")

    page = st.radio(
        "Choose a Case Study",
        options=[
            "🛍️  Retail Sales Dashboard",
            "🎓  Student Grades Explorer",
            "🎬  Movie Ratings Explorer",
            "🚢  Titanic Survival Analysis",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("Select a case study above to explore the analysis.")


# ══════════════════════════════════════════════════════════════════════════════
# CASE STUDY 1 — Retail Sales Dashboard
# ══════════════════════════════════════════════════════════════════════════════
if page == "🛍️  Retail Sales Dashboard":
    st.title("🛍️ Retail Sales Dashboard")
    st.write("Analyze daily retail sales data and visualize product performance.")

    df = pd.read_csv(csv("sales.csv"))
    df["Date"] = pd.to_datetime(df["Date"])

    st.header("Dataset Preview")
    st.table(df.head())

    st.header("Summary Statistics")
    mean_revenue = df["Revenue"].mean()
    median_revenue = df["Revenue"].median()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Revenue", f"${mean_revenue:.2f}")
    with col2:
        st.metric("Median Revenue", f"${median_revenue:.2f}")

    st.header("Revenue and Units Sold by Category")
    category_summary = (
        df.groupby("ProductCategory")[["UnitsSold", "Revenue"]].sum().reset_index()
    )
    st.table(category_summary)

    st.header("Daily Revenue Trend")
    daily_revenue = df.groupby("Date")["Revenue"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily_revenue["Date"], daily_revenue["Revenue"], marker="o", color="#4C72B0")
    ax.set_title("Daily Revenue Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    plt.xticks(rotation=45)
    fig.tight_layout()
    st.pyplot(fig)

    st.header("Revenue by Product Category")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        category_summary["ProductCategory"],
        category_summary["Revenue"],
        color="#4C72B0",
    )
    ax.set_xlabel("Product Category")
    ax.set_ylabel("Revenue")
    ax.set_title("Revenue by Product Category")
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.caption("Advanced Data Science Lab — Assignment 02")


# ══════════════════════════════════════════════════════════════════════════════
# CASE STUDY 2 — Student Grades Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎓  Student Grades Explorer":
    st.title("🎓 Student Grades Explorer")

    df = pd.read_csv(csv("grades.csv"))

    st.header("Dataset Preview")
    st.table(df.head())

    subjects = df["Subject"].unique()
    selected_subject = st.selectbox("Select Subject", subjects)
    subject_data = df[df["Subject"] == selected_subject]

    mean_marks = subject_data["Final"].mean()
    median_marks = subject_data["Final"].median()
    std_marks = subject_data["Final"].std()

    st.header("Summary Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean", f"{mean_marks:.2f}")
    with col2:
        st.metric("Median", f"{median_marks:.2f}")
    with col3:
        st.metric("Std Deviation", f"{std_marks:.2f}")

    st.header("Box Plot of Final Marks")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.boxplot(subject_data["Final"])
    ax.set_ylabel("Marks")
    ax.set_title(f"Final Marks Distribution — {selected_subject}")
    fig.tight_layout()
    st.pyplot(fig)

    st.header("Test1 vs Final Marks")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(subject_data["Test1"], subject_data["Final"], color="#4C72B0", alpha=0.7)
    ax.set_xlabel("Test1 Marks")
    ax.set_ylabel("Final Marks")
    ax.set_title("Test1 vs Final")
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.caption("Advanced Data Science Lab — Assignment 02")


# ══════════════════════════════════════════════════════════════════════════════
# CASE STUDY 3 — Movie Ratings Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎬  Movie Ratings Explorer":
    st.title("🎬 Movie Ratings Explorer")

    df = pd.read_csv(csv("movies.csv"))
    df = df.dropna()

    genre = st.selectbox("Select Movie Genre", sorted(df["Genre"].unique()))
    filtered_df = df[df["Genre"] == genre]

    st.subheader("Movies")
    st.table(filtered_df)

    st.subheader("Summary Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Rating", f"{filtered_df['Rating'].mean():.2f}")
    col2.metric("Average Votes", f"{filtered_df['Votes'].mean():.0f}")
    col3.metric("Median Release Year", int(filtered_df["Year"].median()))

    st.subheader("Ratings Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(filtered_df["Rating"])
    ax.set_ylabel("Rating")
    ax.set_title(f"Rating Distribution — {genre}")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Votes vs Rating")
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        filtered_df["Votes"],
        filtered_df["Rating"],
        s=filtered_df["Votes"] / 50,
        c=filtered_df["Rating"],
        cmap="viridis",
        alpha=0.8,
    )
    ax.set_xlabel("Votes")
    ax.set_ylabel("Rating")
    ax.set_title(f"Votes vs Rating — {genre}")
    plt.colorbar(scatter, ax=ax, label="Rating")
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.caption("Advanced Data Science Lab — Assignment 02")


# ══════════════════════════════════════════════════════════════════════════════
# CASE STUDY 4 — Titanic Survival Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚢  Titanic Survival Analysis":
    st.title("🚢 Titanic Survival Analysis Dashboard")

    df = pd.read_csv(csv("titanic.csv"))
    df = df.dropna()
    df["SurvivalStatus"] = df["Survived"].replace({0: "Not Survived", 1: "Survived"})
    df["PassengerClass"] = df["Pclass"].replace(
        {1: "First Class", 2: "Second Class", 3: "Third Class"}
    )

    st.header("Dataset Preview")
    st.table(df.head())

    st.header("Survival Percentage by Gender")
    gender_survival = df.groupby("Sex")["Survived"].mean() * 100
    st.table(gender_survival.round(2).rename("Survival %"))

    st.header("Survival Percentage by Passenger Class")
    class_survival = df.groupby("PassengerClass")["Survived"].mean() * 100
    st.table(class_survival.round(2).rename("Survival %"))

    st.header("Average Age by Survival Status")
    average_age = df.groupby("SurvivalStatus")["Age"].mean()
    st.table(average_age.round(2).rename("Average Age"))

    st.header("Age Distribution by Survival Status")
    fig, ax = plt.subplots(figsize=(6, 5))
    df.boxplot(column="Age", by="SurvivalStatus", ax=ax)
    plt.suptitle("")
    ax.set_title("Age Distribution by Survival Status")
    ax.set_xlabel("Survival Status")
    ax.set_ylabel("Age")
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.caption("Advanced Data Science Lab — Assignment 02")
