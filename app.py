
import streamlit as st
import pandas as pd
from datetime import date
from pathlib import Path

# Set page config
st.set_page_config("TMF Accounting Dashboard", layout="wide")

# Format numbers
def shorten_currency(amount):
    return f"₹{amount/1_00_000:.1f}L" if amount < 1_00_00_000 else f"₹{amount/1_00_00_000:.1f}Cr"

# Load data
inventory = pd.read_csv("data/inventory.csv")
sales_path = "data/sales.csv"
profit_path = "data/profit.csv"
sales = pd.read_csv(sales_path) if Path(sales_path).exists() else pd.DataFrame(columns=["Date", "Brand", "Model", "Quantity Sold", "Selling Price"])

# Recalculate stock left
inventory['Quantity Left'] = inventory['Quantity Purchased']
for i, sale in sales.iterrows():
    match = (inventory.Item == sale.Brand) & (inventory.Model == sale.Model)
    inventory.loc[match, 'Quantity Left'] -= sale['Quantity Sold']

# Dashboard Title
st.title("🧮 TMF Accounting Dashboard")

# Financials
investment_total = 10000000
inventory['Total Cost'] = inventory['Quantity Purchased'] * inventory['Unit Cost']
stock_value = (inventory['Quantity Left'] * inventory['Unit Cost']).sum()
stock_items = inventory['Quantity Left'].sum()
stock_purchased_qty = inventory['Quantity Purchased'].sum()
total_sales_revenue = (sales['Quantity Sold'] * sales['Selling Price']).sum()

# Cost of Goods Sold
cogs = 0
for i, row in sales.iterrows():
    match = inventory[(inventory.Item == row.Brand) & (inventory.Model == row.Model)]
    unit_cost = match['Unit Cost'].iloc[0] if not match.empty else 0
    cogs += unit_cost * row['Quantity Sold']

profit = total_sales_revenue - cogs
investment_left = investment_total - inventory['Total Cost'].sum() + profit

# Break-even period
avg_daily_profit = profit / max(1, len(sales['Date'].unique()))
break_even_days = (investment_total - profit) / avg_daily_profit if avg_daily_profit > 0 else float('inf')
break_even_estimate = f"{int(break_even_days)} days" if break_even_days != float('inf') else "Not yet profitable"

# Metric cards
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
col1.metric("💼 Total Investment", shorten_currency(investment_total))
col2.metric("🏗️ Investment Utilised", shorten_currency(inventory['Total Cost'].sum()))
col3.metric("🏦 Investment Left", shorten_currency(investment_left))
col4.metric("📊 Total Profit", shorten_currency(profit))
col5.metric("📅 Est. Break-Even", break_even_estimate)
col6.metric("🧾 Stock Purchased", f"{stock_purchased_qty}")
col7.metric("📦 Stock Left", f"{stock_items}")

# Inventory Table
st.subheader("📦 Stock Purchased")
inventory['Total Cost'] = inventory['Quantity Purchased'] * inventory['Unit Cost']
st.dataframe(inventory[['Item', 'Quantity Purchased', 'Unit Cost', 'Total Cost', 'Quantity Left']])

# Sales Table
st.subheader("🧾 Sales & Profit")

def get_unit_cost(row):
    match = inventory[(inventory.Item == row.Brand) & (inventory.Model == row.Model)]
    return match['Unit Cost'].iloc[0] if not match.empty else 0

if not sales.empty:
    sales['Unit Cost'] = sales.apply(get_unit_cost, axis=1)
    sales['Total Revenue'] = sales['Quantity Sold'] * sales['Selling Price']
    sales['Profit'] = (sales['Selling Price'] - sales['Unit Cost']) * sales['Quantity Sold']
    st.dataframe(sales[['Date', 'Item', 'Quantity Sold', 'Unit Cost', 'Selling Price', 'Total Revenue', 'Profit']])
else:
    st.info("No sales data available yet.")

# Profit Sharing
st.subheader("🤝 Profit Sharing")
man_a_net = profit * 0.42
man_b_total = profit * 0.40
man_c_share = profit * 0.18

profit_df = pd.read_csv(profit_path) if Path(profit_path).exists() else pd.DataFrame(columns=["Period", "Total Profit", "Man A (Net)", "Man B", "Man C (From A)"])
latest_period = pd.to_datetime(date.today()).strftime("%Y-%m")
if latest_period in profit_df["Period"].values:
    idx = profit_df[profit_df["Period"] == latest_period].index[0]
    profit_df.loc[idx] = [latest_period, profit, man_a_net, man_b_total, man_c_share]
else:
    new_row = {"Period": latest_period, "Total Profit": profit, "Man A (Net)": man_a_net, "Man B": man_b_total, "Man C (From A)": man_c_share}
    profit_df = pd.concat([profit_df, pd.DataFrame([new_row])], ignore_index=True)
profit_df.to_csv(profit_path, index=False)
st.dataframe(profit_df)

# Add Sale
st.subheader("➕ Add a Sale")
with st.form("Add Sale"):
    item = st.selectbox("Item", inventory["Item"].unique())
    qty = st.number_input("Quantity Sold", min_value=1)
    price = st.number_input("Selling Price", min_value=1)
    confirm = st.checkbox("Confirm sale entry")
    submit = st.form_submit_button("Add Sale")

    if submit and confirm:
        new_sale = {"Date": str(date.today()), "Item": item, "Quantity Sold": qty, "Selling Price": price}
        sales = pd.concat([sales, pd.DataFrame([new_sale])], ignore_index=True)
        sales.to_csv(sales_path, index=False)
        inventory.loc[(inventory.Item == brand) & (inventory.Model == model), "Quantity Left"] -= qty
        inventory.to_csv("data/inventory.csv", index=False)
        st.success("Sale recorded successfully!")

# Manage Sales
st.subheader("🗑️ Manage Sales Entries")
if not sales.empty:
    for i in range(len(sales)):
        row = sales.iloc[i]
        col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 1, 2, 2, 1])
        col1.write(row["Date"])
        col2.write(row['Item'])
        col3.write(f"{row['Quantity Sold']} pcs")
        col4.write(f"₹{row['Selling Price']}")
        col5.write(f"₹{row.get('Profit', '-')}")
        if col6.button("Delete", key=f"delete_{i}"):
            if st.checkbox(f"Confirm delete sale {i}"):
                inventory.loc[(inventory.Item == row["Brand"]) & (inventory.Model == row["Model"]), "Quantity Left"] += row["Quantity Sold"]
                sales = sales.drop(index=i).reset_index(drop=True)
                sales.to_csv(sales_path, index=False)
                inventory.to_csv("data/inventory.csv", index=False)
                st.success("Sale deleted. Please refresh to see the change.")
                st.stop()
else:
    st.info("No sales entries to manage.")