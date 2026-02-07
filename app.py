import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# ----- Page Config -----
st.set_page_config(
    page_title="Budget Risk Predictor",
    page_icon="💰",
    layout="centered"
)

# ----- Title -----
st.markdown("<h1 style='text-align: center; color: #1a535c;'>💸 Budget Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #4ECDC4;'>Track your expenses and see your budget risk instantly!</p>", unsafe_allow_html=True)
st.write("---")

# ----- Dataset -----
data = {
    "income": [2000, 5000, 10000, 20000, 30000, 40000, 45000, 50000, 25000, 35000],
    "expense": [1800, 4800, 9000, 18000, 27000, 36000, 44000, 49000, 26000, 37000],
    "risk":   [0,0,0,0,0,0,0,0,1,1]
}
df = pd.DataFrame(data)

X = df[["income","expense"]]
y = df["risk"]

model = DecisionTreeClassifier()
model.fit(X,y)

# ----- Inputs -----
st.header("📥 Enter Your Monthly Income & Expenses")
income = st.number_input("💰 Monthly Income:", min_value=0, value=0)
rent = st.number_input("🏠 Rent:", min_value=0, value=0)
food = st.number_input("🍔 Food:", min_value=0, value=0)
transport = st.number_input("🚗 Transport:", min_value=0, value=0)
shopping = st.number_input("🛍️ Shopping:", min_value=0, value=0)
other = st.number_input("📝 Other Expenses:", min_value=0, value=0)

total_expense = rent + food + transport + shopping + other

# ----- ML Prediction -----
new_data = pd.DataFrame([[income, total_expense]], columns=["income","expense"])
ml_prediction = model.predict(new_data)
ml_probability = model.predict_proba(new_data)

# ----- Rule-based Risk & Advice -----
if total_expense > income:
    risk = "HIGH Risk - Overspending"
    risk_color = "red"
    advice = "⚠️ You are spending more than your income! Reduce expenses immediately!"
elif total_expense >= 0.9*income:
    risk = "MEDIUM Risk - Warning Zone"
    risk_color = "orange"
    advice = "⚠️ Close to overspending! Consider saving more this month."
else:
    risk = "LOW Risk - Safe Budget" if ml_prediction[0]==0 else "MEDIUM Risk - Check Spending"
    risk_color = "green" if ml_prediction[0]==0 else "orange"
    advice = "✅ Great! You are within your budget." if ml_prediction[0]==0 else "⚠️ Monitor your spending."

# ----- Results Section -----
st.markdown(f"""
<div style='background-color:#f0f8ff; padding:15px; border-radius:10px'>
<h3 style='color:#1a535c'>📊 Budget Analysis</h3>
<p><b>Total Income:</b> {income}</p>
<p><b>Total Expense:</b> {total_expense}</p>
<p><b>Budget Risk Level:</b> <span style='color:{risk_color};'>{risk}</span></p>
<p><b>Confidence Level:</b> {round(max(ml_probability[0])*100,2)}%</p>
<p><b>Advice:</b> {advice}</p>
</div>
""", unsafe_allow_html=True)

# ----- Graphs Section -----
st.header("📈 Expense Breakdown")
categories = ["Rent","Food","Transport","Shopping","Other"]
values = [rent, food, transport, shopping, other]

# Safety for NaN
values = [0 if pd.isna(x) else x for x in values]

# ----- Bar Chart -----
st.subheader("Bar Chart")
st.bar_chart(pd.DataFrame({"Amount": values}, index=categories))
# ----- Pie Chart -----
st.subheader("Pie Chart")

if sum(values) == 0:
    st.warning("⚠️ No expenses entered yet! Pie chart cannot be displayed.")
else:
    colors = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#1A535C", "#FF9F1C"]
    fig1, ax1 = plt.subplots()
    ax1.pie(values, labels=categories, autopct="%1.1f%%", startangle=90, colors=colors)
    ax1.axis("equal")
    st.pyplot(fig1)

