
import streamlit as st
import pandas as pd
from datetime import date

# Format numbers
def shorten_currency(amount):
    return f"₹{amount/1_00_000:.1f}L" if amount < 1_00_00_000 else f"₹{amount/1_00_00_000:.1f}Cr"

st.set_page_config("TMF Accounting Dashboard", layout="wide")

# Load data
inventory = pd.read_csv("data/inventory.csv")
sales = pd.read_csv("data/sales.csv")

# Ensure Buyer Name column exists
if 'Buyer Name' not in sales.columns:
    sales['Buyer Name'] = ""

# Recalculate stock left from past sales
inventory['Quantity Left'] = inventory['Quantity Purchased']
for i, sale in sales.iterrows():
    match = (inventory.Item == sale.Item)
    inventory.loc[match, 'Quantity Left'] -= sale['Quantity Sold']

# Title
st.title("📊 TMF Accounting Dashboard")

# Financial calculations
investment_total = 10000000  # 1 Cr
inventory['Total Cost'] = inventory['Quantity Purchased'] * inventory['Unit Cost']
stock_value = (inventory['Quantity Left'] * inventory['Unit Cost']).sum()
stock_items = inventory['Quantity Left'].sum()
stock_purchased_qty = inventory['Quantity Purchased'].sum()
total_sales_revenue = (sales['Quantity Sold'] * sales['Selling Price']).sum()

# Calculate cost of goods sold (COGS)
cogs = 0
for i, row in sales.iterrows():
    match = inventory[inventory.Item == row.Item]
    unit_cost = match['Unit Cost'].iloc[0] if not match.empty else 0
    cogs += unit_cost * row['Quantity Sold']

profit = total_sales_revenue - cogs
investment_left = investment_total - inventory['Total Cost'].sum() + profit

# Break-even period estimation
avg_daily_profit = profit / max(1, len(sales['Date'].unique()))
break_even_days = (investment_total - profit) / avg_daily_profit if avg_daily_profit > 0 else float('inf')
break_even_estimate = f"{int(break_even_days)} days" if break_even_days != float('inf') else "Not yet profitable"

# Metric cards
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
col1.metric("Total Investment", shorten_currency(investment_total))
col2.metric("Investment Utilised", shorten_currency(inventory['Total Cost'].sum()))
col3.metric("Investment Left", shorten_currency(investment_left))
col4.metric("Total Profit", shorten_currency(profit))
col5.metric("Est.Break-Even Period", break_even_estimate)
col6.metric("Stock Purchased (Items)", f"{stock_purchased_qty}")
col7.metric("Stock Left (Items)", f"{stock_items}")

# Inventory Table
st.subheader("📦 Stock Purchased")
st.dataframe(inventory[['Item', 'Quantity Purchased', 'Unit Cost', 'Quantity Left', 'Total Cost']])

# Sales & Profit Table
st.subheader("🧾 Sales & Profit")
def get_unit_cost(row):
    match = inventory[inventory.Item == row.Item]
    return match['Unit Cost'].iloc[0] if not match.empty else 0

sales['Unit Cost'] = sales.apply(get_unit_cost, axis=1)
sales['Total Revenue'] = sales['Quantity Sold'] * sales['Selling Price']
sales['Profit'] = (sales['Selling Price'] - sales['Unit Cost']) * sales['Quantity Sold']
st.dataframe(sales[['Date', 'Buyer Name', 'Item', 'Quantity Sold', 'Unit Cost', 'Selling Price', 'Total Revenue', 'Profit']])

# Profit Sharing
st.subheader("🤝 Profit Sharing")
man_a_net = profit * 0.42
man_b_total = profit * 0.40
man_c_share = profit * 0.18

profit_df = pd.read_csv("data/profit.csv")
latest_period = pd.to_datetime(date.today()).strftime("%Y-%m")
if latest_period in profit_df["Period"].values:
    idx = profit_df[profit_df["Period"] == latest_period].index[0]
    profit_df.loc[idx, "Total Profit"] = profit
    profit_df.loc[idx, "Man A (Net)"] = man_a_net
    profit_df.loc[idx, "Man B"] = man_b_total
    profit_df.loc[idx, "Man C (From A)"] = man_c_share
else:
    new_row = {
        "Period": latest_period,
        "Total Profit": profit,
        "Man A (Net)": man_a_net,
        "Man B": man_b_total,
        "Man C (From A)": man_c_share,
    }
    profit_df = pd.concat([profit_df, pd.DataFrame([new_row])], ignore_index=True)
profit_df.to_csv("data/profit.csv", index=False)
st.dataframe(profit_df)

# Add Sale
st.subheader("➕ Add a Sale")
with st.form("Add Sale"):
    item = st.selectbox("Item", inventory["Item"].unique())
    qty = st.number_input("Quantity Sold", min_value=1)
    price = st.number_input("Selling Price", min_value=1)
    buyer = st.text_input("Buyer Name")
    submit = st.form_submit_button("Add Sale")

    if submit:
        new_sale = {
            "Date": str(date.today()),
            "Buyer Name": buyer,
            "Item": item,
            "Quantity Sold": qty,
            "Selling Price": price
        }
        sales = pd.concat([sales, pd.DataFrame([new_sale])], ignore_index=True)
        sales.to_csv("data/sales.csv", index=False)
        inventory.loc[inventory.Item == item, "Quantity Left"] -= qty
        inventory.to_csv("data/inventory.csv", index=False)
        st.success("Sale recorded!")

# Manage Sales Entries
st.subheader("🗑️ Manage Sales Entries")
if not sales.empty:
    for i in range(len(sales)):
        row = sales.iloc[i]
        col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 1, 2, 2, 1])
        col1.write(row["Date"])
        col2.write(f"{row['Item']} ({row['Buyer Name']})")
        col3.write(f"{row['Quantity Sold']} pcs")
        col4.write(f"₹{row['Selling Price']}")
        col5.write(f"₹{row['Profit']:.0f}")
        if col6.button("Delete", key=f"delete_{i}"):
            inventory.loc[inventory.Item == row["Item"], "Quantity Left"] += row["Quantity Sold"]
            sales = sales.drop(index=i).reset_index(drop=True)
            sales.to_csv("data/sales.csv", index=False)
            inventory.to_csv("data/inventory.csv", index=False)
            st.success("Sale deleted! Please refresh the page manually to see changes.")
            st.stop()
else:
    st.info("No sales entries to manage.")
