import streamlit as st
import pandas as pd
import numpy as np
import gspread

from datetime import date
from google.oauth2.service_account import Credentials
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

SHEET_COLUMNS = [
    "Date",
    "Category",
    "Amount",
    "Description"
]


# ============================================================
# GOOGLE SHEETS CONNECTION
# ============================================================

@st.cache_resource
def connect_to_google_sheet():

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    credentials = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=scopes
    )

    client = gspread.authorize(credentials)

    sheet_id = st.secrets["google_sheet_id"]

    spreadsheet = client.open_by_key(sheet_id)

    worksheet = spreadsheet.sheet1

    return worksheet


# ============================================================
# LOAD DATA FROM GOOGLE SHEETS
# ============================================================

def load_data():

    try:

        worksheet = connect_to_google_sheet()

        records = worksheet.get_all_records()

        if not records:

            return pd.DataFrame(
                columns=SHEET_COLUMNS
            )

        df = pd.DataFrame(records)

        # Make sure required columns exist
        for column in SHEET_COLUMNS:

            if column not in df.columns:

                df[column] = ""

        df = df[SHEET_COLUMNS]

        # Convert Date
        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        # Convert Amount
        df["Amount"] = pd.to_numeric(
            df["Amount"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["Date", "Amount"]
        )

        return df

    except Exception as e:

        st.error(
            f"❌ Could not connect to Google Sheets: {e}"
        )

        return pd.DataFrame(
            columns=SHEET_COLUMNS
        )


# ============================================================
# SAVE DATA TO GOOGLE SHEETS
# ============================================================

def save_data(df):

    worksheet = connect_to_google_sheet()

    # Make a copy so we don't modify the original dataframe
    save_df = df.copy()

    # Convert dates to strings
    save_df["Date"] = pd.to_datetime(
        save_df["Date"],
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    # Make sure correct column order
    save_df = save_df[
        SHEET_COLUMNS
    ]

    # Convert NaN to empty values
    save_df = save_df.fillna("")

    # Convert everything to string
    values = save_df.astype(str).values.tolist()

    # Clear existing worksheet
    worksheet.clear()

    # Header row
    worksheet.update(
        range_name="A1:D1",
        values=[SHEET_COLUMNS]
    )

    # Write data
    if values:

        worksheet.update(
            range_name=f"A2:D{len(values) + 1}",
            values=values
        )


# ============================================================
# ADD ROW TO GOOGLE SHEETS
# ============================================================

def add_expense_to_sheet(
    expense_date,
    category,
    amount,
    description
):

    worksheet = connect_to_google_sheet()

    worksheet.append_row(
        [
            expense_date.strftime("%Y-%m-%d"),
            category,
            float(amount),
            description
        ],
        value_input_option="USER_ENTERED"
    )


# ============================================================
# MONTHLY EXPENSE CALCULATION
# ============================================================

def monthly_expenses(df):

    if df.empty:

        return pd.DataFrame(
            columns=[
                "Date",
                "Amount"
            ]
        )

    monthly = (
        df.groupby(
            df["Date"].dt.to_period("M")
        )["Amount"]
        .sum()
        .reset_index()
    )

    monthly["Date"] = (
        monthly["Date"]
        .dt.to_timestamp()
    )

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

    if result.empty:

        result["Anomaly"] = []

        return result

    if len(result) < 10:

        result["Anomaly"] = 1

        return result

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

    result["Anomaly"] = (
        model.fit_predict(features)
    )

    return result


# ============================================================
# RECOMMENDATIONS
# ============================================================

def generate_recommendations(df):

    recommendations = []

    if df.empty:

        return [
            "📊 Add some expenses to receive personalized recommendations."
        ]

    total = df["Amount"].sum()

    if total <= 0:

        return [
            "📊 Add valid expenses to receive recommendations."
        ]

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

    # Transport recommendation
    if "Transport" in category_spending:

        transport = category_spending["Transport"]

        if transport > total * 0.15:

            recommendations.append(
                "🚗 Transport expenses are relatively high. "
                "Consider public transport or combining trips."
            )

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

    if len(current_month_data) == 0 and not df.empty:

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

    else:

        highest_category = "N/A"


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

    if not monthly.empty:

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

    else:

        st.info(
            "No expense data available yet."
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

        if not category_df.empty:

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

        else:

            st.info(
                "No category data available."
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

            try:

                add_expense_to_sheet(
                    expense_date,
                    category,
                    amount,
                    description
                )

                st.success(
                    "✅ Expense added successfully to Google Sheets!"
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"❌ Could not add expense: {e}"
                )


# ============================================================
# IMPORT CSV
# ============================================================

elif page == "Import CSV":

    st.title(
        "📂 Import Expenses"
    )

    st.write(
        "Upload a CSV file and add the transactions to Google Sheets."
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
                    uploaded_df["Date"],
                    errors="coerce"
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

                uploaded_df = uploaded_df[
                    SHEET_COLUMNS
                ]

                # Add imported rows to existing data
                combined_df = pd.concat(
                    [
                        df,
                        uploaded_df
                    ],
                    ignore_index=True
                )

                save_data(
                    combined_df
                )

                st.success(
                    f"✅ Imported {len(uploaded_df)} transactions into Google Sheets."
                )

                st.dataframe(
                    uploaded_df,
                    use_container_width=True
                )

                # Refresh data
                df = load_data()

        except Exception as e:

            st.error(
                f"❌ Error importing CSV: {e}"
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

            if last_month > 0:

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

    if not df.empty:

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

    else:

        st.info(
            "No expense data available."
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
    "Built with Python + Streamlit + Google Sheets + Machine Learning"
)
