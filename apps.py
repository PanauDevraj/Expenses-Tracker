import streamlit as st
import pandas as pd
import numpy as np
import os

from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartSpend AI",
    page_icon="💰",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

DATA_FILE = "expenses.csv"

CATEGORIES = [
    "Food",
    "Transport",
    "Shopping",
    "Entertainment",
    "Education",
    "Bills",
    "Health",
    "Other"
]


# ============================================================
# CREATE SAMPLE DATA
# ============================================================

def create_sample_data():

    np.random.seed(42)

    dates = pd.date_range(
        start="2026-01-01",
        end="2026-08-15",
        freq="D"
    )

    data = []

    for d in dates:

        # Random number of transactions per day
        number_of_transactions = np.random.randint(0, 4)

        for _ in range(number_of_transactions):

            category = np.random.choice(
                CATEGORIES,
                p=[
                    0.25,  # Food
                    0.15,  # Transport
                    0.15,  # Shopping
                    0.10,  # Entertainment
                    0.10,  # Education
                    0.10,  # Bills
                    0.05,  # Health
                    0.10   # Other
                ]
            )

            # Normal expense ranges
            ranges = {
                "Food": (80, 600),
                "Transport": (50, 400),
                "Shopping": (200, 2000),
                "Entertainment": (100, 1000),
                "Education": (200, 1500),
                "Bills": (300, 3000),
                "Health": (100, 2000),
                "Other": (50, 1000)
            }

            low, high = ranges[category]

            amount = round(
                np.random.uniform(low, high),
                2
            )

            data.append([
                d.strftime("%Y-%m-%d"),
                category,
                amount,
                f"{category} expense"
            ])

    df = pd.DataFrame(
        data,
        columns=[
            "Date",
            "Category",
            "Amount",
            "Description"
        ]
    )

    # Add a few obvious anomalies
    anomaly_data = pd.DataFrame([
        ["2026-08-05", "Food", 2800, "Large restaurant bill"],
        ["2026-08-08", "Shopping", 6500, "Unusual purchase"],
        ["2026-08-10", "Transport", 2500, "Unusual travel expense"]
    ], columns=df.columns)

    df = pd.concat(
        [df, anomaly_data],
        ignore_index=True
    )

    return df


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not os.path.exists(DATA_FILE):

        df = create_sample_data()

        df.to_csv(
            DATA_FILE,
            index=False
        )

    else:

        df = pd.read_csv(DATA_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    df["Amount"] = pd.to_numeric(
        df["Amount"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date", "Amount"]
    )

    return df


# ============================================================
# SAVE DATA
# ============================================================

def save_data(df):

    df.to_csv(
        DATA_FILE,
        index=False
    )


# ============================================================
# MONTHLY EXPENSE CALCULATION
# ============================================================

def monthly_expenses(df):

    monthly = (
        df.groupby(
            df["Date"].dt.to_period("M")
        )["Amount"]
        .sum()
        .reset_index()
    )

    monthly["Date"] = monthly["Date"].dt.to_timestamp()

    return monthly


# ============================================================
# PREDICTION MODEL
# ============================================================

def predict_next_month(df):

    monthly = monthly_expenses(df)

    if len(monthly) < 2:

        return None

    monthly["MonthNumber"] = np.arange(
        len(monthly)
    )

    X = monthly[
        ["MonthNumber"]
    ]

    y = monthly[
        "Amount"
    ]

    model = LinearRegression()

    model.fit(
        X,
        y
    )

    next_month_number = np.array([
        [len(monthly)]
    ])

    prediction = model.predict(
        next_month_number
    )[0]

    # Prevent negative prediction
    prediction = max(
        0,
        prediction
    )

    return prediction


# ============================================================
# ANOMALY DETECTION
# ============================================================

def detect_anomalies(df):

    result = df.copy()

    if len(result) < 10:

        result["Anomaly"] = 1

        return result

    # Features used by Isolation Forest
    result["DayOfWeek"] = (
        result["Date"].dt.dayofweek
    )

    result["Month"] = (
        result["Date"].dt.month
    )

    result["Day"] = (
        result["Date"].dt.day
    )

    features = result[
        [
            "Amount",
            "DayOfWeek",
            "Month",
            "Day"
        ]
    ]

    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    result["Anomaly"] = model.fit_predict(
        features
    )

    return result


# ============================================================
# RECOMMENDATIONS
# ============================================================

def generate_recommendations(df):

    recommendations = []

    total = df["Amount"].sum()

    category_spending = (
        df.groupby("Category")["Amount"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    if len(category_spending) > 0:

        highest_category = (
            category_spending.index[0]
        )

        highest_amount = (
            category_spending.iloc[0]
        )

        percentage = (
            highest_amount / total
        ) * 100

        if percentage > 30:

            recommendations.append(
                f"💡 {highest_category} accounts for "
                f"{percentage:.1f}% of your spending. "
                f"Consider setting a monthly budget for this category."
            )

    # Food recommendation
    if "Food" in category_spending:

        food = category_spending["Food"]

        if food > total * 0.20:

            recommendations.append(
                "🍔 Your food spending is relatively high. "
                "Reducing food delivery and restaurant expenses "
                "could improve your monthly savings."
            )

    # Shopping recommendation
    if "Shopping" in category_spending:

        shopping = category_spending["Shopping"]

        if shopping > total * 0.15:

            recommendations.append(
                "🛍️ Shopping is taking a significant portion "
                "of your expenses. Consider using a fixed "
                "shopping budget."
            )

    # Transport
    if "Transport" in category_spending:

        transport = category_spending["Transport"]

        if transport > total * 0.15:

            recommendations.append(
                "🚗 Transport expenses are relatively high. "
                "Consider public transport or combining trips."
            )

    # General recommendation
    if len(recommendations) == 0:

        recommendations.append(
            "✅ Your spending pattern looks reasonably balanced. "
            "Continue tracking your expenses regularly."
        )

    return recommendations


# ============================================================
# LOAD DATA
# ============================================================

df = load_data()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("💰 SmartSpend AI")

st.sidebar.write(
    "AI-Powered Personal Expense Tracker"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Add Expense",
        "Import CSV",
        "AI Insights",
        "Anomaly Detection"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title("💰 SmartSpend AI")

    st.subheader(
        "AI-Powered Personal Expense Tracker & Financial Assistant"
    )

    st.write(
        "Track • Analyze • Predict • Detect • Improve"
    )

    st.divider()

    # --------------------------------------------------------
    # KPI CALCULATIONS
    # --------------------------------------------------------

    total_spending = df["Amount"].sum()

    current_month = pd.Timestamp.today().to_period("M")

    current_month_data = df[
        df["Date"].dt.to_period("M")
        == current_month
    ]

    # If current month has no data, use latest month
    if len(current_month_data) == 0:

        latest_month = (
            df["Date"]
            .dt.to_period("M")
            .max()
        )

        current_month_data = df[
            df["Date"].dt.to_period("M")
            == latest_month
        ]

    monthly_spending = (
        current_month_data["Amount"].sum()
    )

    prediction = predict_next_month(df)

    if prediction is None:

        prediction = 0

    category_spending = (
        df.groupby("Category")["Amount"]
        .sum()
    )

    if len(category_spending) > 0:

        highest_category = (
            category_spending.idxmax()
        )

        highest_category_amount = (
            category_spending.max()
        )

    else:

        highest_category = "N/A"
        highest_category_amount = 0


    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Spending",
            f"₹{total_spending:,.0f}"
        )

    with col2:

        st.metric(
            "Current Month",
            f"₹{monthly_spending:,.0f}"
        )

    with col3:

        st.metric(
            "Predicted Next Month",
            f"₹{prediction:,.0f}"
        )

    with col4:

        st.metric(
            "Highest Category",
            highest_category
        )


    st.divider()


    # --------------------------------------------------------
    # MONTHLY SPENDING GRAPH
    # --------------------------------------------------------

    st.subheader(
        "📈 Monthly Spending"
    )

    monthly = monthly_expenses(df)

    fig = px.line(
        monthly,
        x="Date",
        y="Amount",
        markers=True,
        title="Monthly Expense Trend"
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Amount (₹)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # CATEGORY CHART
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "🥧 Spending by Category"
        )

        category_df = (
            df.groupby("Category")["Amount"]
            .sum()
            .reset_index()
        )

        fig2 = px.pie(
            category_df,
            names="Category",
            values="Amount",
            hole=0.4
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )


    # --------------------------------------------------------
    # TOP EXPENSES
    # --------------------------------------------------------

    with col2:

        st.subheader(
            "💸 Top Expenses"
        )

        top_expenses = (
            df.sort_values(
                "Amount",
                ascending=False
            )
            .head(10)
        )

        st.dataframe(
            top_expenses[
                [
                    "Date",
                    "Category",
                    "Amount",
                    "Description"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# ADD EXPENSE
# ============================================================

elif page == "Add Expense":

    st.title(
        "➕ Add New Expense"
    )

    st.write(
        "Record your daily transaction."
    )

    with st.form("expense_form"):

        expense_date = st.date_input(
            "Date",
            value=date.today()
        )

        category = st.selectbox(
            "Category",
            CATEGORIES
        )

        amount = st.number_input(
            "Amount (₹)",
            min_value=1.0,
            step=10.0
        )

        description = st.text_input(
            "Description",
            placeholder="Example: Lunch at restaurant"
        )

        submitted = st.form_submit_button(
            "Add Expense"
        )

        if submitted:

            new_expense = pd.DataFrame(
                [
                    {
                        "Date": expense_date,
                        "Category": category,
                        "Amount": amount,
                        "Description": description
                    }
                ]
            )

            df = pd.concat(
                [
                    df,
                    new_expense
                ],
                ignore_index=True
            )

            save_data(df)

            st.success(
                "✅ Expense added successfully!"
            )

            st.rerun()


# ============================================================
# IMPORT CSV
# ============================================================

elif page == "Import CSV":

    st.title(
        "📂 Import Expenses"
    )

    st.write(
        "Upload a CSV file containing your expenses."
    )

    st.info(
        "CSV must contain: Date, Category, Amount, Description"
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(
                uploaded_file
            )

            required_columns = [
                "Date",
                "Category",
                "Amount"
            ]

            missing_columns = [
                col
                for col in required_columns
                if col not in uploaded_df.columns
            ]

            if missing_columns:

                st.error(
                    "Missing columns: "
                    + ", ".join(missing_columns)
                )

            else:

                uploaded_df["Date"] = pd.to_datetime(
                    uploaded_df["Date"]
                )

                uploaded_df["Amount"] = pd.to_numeric(
                    uploaded_df["Amount"],
                    errors="coerce"
                )

                uploaded_df = uploaded_df.dropna(
                    subset=[
                        "Date",
                        "Amount"
                    ]
                )

                if "Description" not in uploaded_df.columns:

                    uploaded_df["Description"] = ""

                df = pd.concat(
                    [
                        df,
                        uploaded_df
                    ],
                    ignore_index=True
                )

                save_data(df)

                st.success(
                    f"✅ Imported {len(uploaded_df)} transactions."
                )

                st.dataframe(
                    uploaded_df,
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                f"Error reading CSV: {e}"
            )


# ============================================================
# AI INSIGHTS
# ============================================================

elif page == "AI Insights":

    st.title(
        "🤖 AI Financial Insights"
    )

    st.write(
        "SmartSpend AI analyzes your spending behavior "
        "and provides personalized recommendations."
    )

    st.divider()

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    st.subheader(
        "🔮 Expense Prediction"
    )

    prediction = predict_next_month(df)

    if prediction is not None:

        st.metric(
            "Predicted Next Month Spending",
            f"₹{prediction:,.0f}"
        )

        monthly = monthly_expenses(df)

        if len(monthly) >= 2:

            last_month = monthly.iloc[-1]["Amount"]

            difference = (
                prediction - last_month
            )

            percentage_change = (
                difference / last_month
            ) * 100

            if percentage_change > 0:

                st.warning(
                    f"⚠️ Your spending may increase by "
                    f"{percentage_change:.1f}% next month."
                )

            else:

                st.success(
                    f"✅ Your spending may decrease by "
                    f"{abs(percentage_change):.1f}% next month."
                )

    else:

        st.info(
            "Need at least two months of expense data "
            "for prediction."
        )


    st.divider()


    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    st.subheader(
        "💡 Personalized Recommendations"
    )

    recommendations = generate_recommendations(
        df
    )

    for recommendation in recommendations:

        st.info(
            recommendation
        )


    st.divider()


    # --------------------------------------------------------
    # CATEGORY ANALYSIS
    # --------------------------------------------------------

    st.subheader(
        "📊 Category Analysis"
    )

    category_analysis = (
        df.groupby("Category")["Amount"]
        .agg(
            [
                "sum",
                "mean",
                "count"
            ]
        )
        .reset_index()
    )

    category_analysis.columns = [
        "Category",
        "Total Spending",
        "Average Expense",
        "Transactions"
    ]

    st.dataframe(
        category_analysis.sort_values(
            "Total Spending",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ANOMALY DETECTION
# ============================================================

elif page == "Anomaly Detection":

    st.title(
        "🚨 Unusual Expense Detection"
    )

    st.write(
        "Isolation Forest identifies transactions "
        "that look unusual compared with your normal spending."
    )

    st.divider()

    result = detect_anomalies(
        df
    )

    anomalies = result[
        result["Anomaly"] == -1
    ]

    normal = result[
        result["Anomaly"] == 1
    ]

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Total Transactions",
            len(result)
        )

    with col2:

        st.metric(
            "Unusual Transactions",
            len(anomalies)
        )


    st.divider()


    if len(anomalies) > 0:

        st.error(
            "🚨 Unusual expenses detected!"
        )

        display_anomalies = anomalies[
            [
                "Date",
                "Category",
                "Amount",
                "Description"
            ]
        ].sort_values(
            "Amount",
            ascending=False
        )

        st.dataframe(
            display_anomalies,
            use_container_width=True,
            hide_index=True
        )

        # Anomaly chart
        fig = px.scatter(
            result,
            x="Date",
            y="Amount",
            color=result["Anomaly"].map({
                1: "Normal",
                -1: "Unusual"
            }),
            hover_data=[
                "Category",
                "Description"
            ],
            title="Expense Anomaly Detection"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.success(
            "✅ No unusual expenses detected."
        )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "SmartSpend AI | SIGNAL SHIFT 2K26"
)

st.sidebar.caption(
    "Built with Python + Streamlit + Machine Learning"
)