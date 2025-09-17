
import streamlit as st
import pandas as pd
from datetime import date

st.set_page_config("TMF Accounting Dashboard", layout="wide")

# Format numbers
def shorten_currency(amount):
    return f"₹{amount/1_00_000:.1f}L" if amount < 1_00_00_000 else f"₹{amount/1_00_00_000:.1f}Cr"

# Load data
inventory = pd.read_csv("data/inventory.csv")
sales = pd.read_csv("data/sales.csv")
profit_df = pd.read_csv("data/profit.csv")

# Recalculate Quantity Left
inventory['Quantity Left'] = inventory['Quantity Purchased']
for _, sale in sales.iterrows():
    match = (inventory.Item == sale.Item)
    inventory.loc[match, 'Quantity Left'] -= sale['Quantity Sold']

# Title
st.title("📊 TMF Accounting Dashboard")

# Financials
investment_total = 10000000
inventory['Total Cost'] = inventory['Quantity Purchased'] * inventory['Unit Cost']
stock_value = (inventory['Quantity Left'] * inventory['Unit Cost']).sum()
stock_items = inventory['Quantity Left'].sum()
stock_purchased_qty = inventory['Quantity Purchased'].sum()
total_sales_revenue = (sales['Quantity Sold'] * sales['Selling Price']).sum()

# COGS
cogs = 0
for _, row in sales.iterrows():
    match = inventory[inventory.Item == row.Item]
    unit_cost = match['Unit Cost'].iloc[0] if not match.empty else 0
    cogs += unit_cost * row['Quantity Sold']

profit = total_sales_revenue - cogs
investment_left = investment_total - inventory['Total Cost'].sum() + profit
avg_daily_profit = profit / max(1, len(sales['Date'].unique()))
break_even_days = (investment_total - profit) / avg_daily_profit if avg_daily_profit > 0 else float('inf')
break_even_estimate = f"{int(break_even_days)} days" if break_even_days != float('inf') else "Not yet profitable"

# Metrics
cols = st.columns(7)
cols[0].metric("Total Investment", shorten_currency(investment_total))
cols[1].metric("Investment Utilised", shorten_currency(inventory['Total Cost'].sum()))
cols[2].metric("Investment Left", shorten_currency(investment_left))
cols[3].metric("Total Profit", shorten_currency(profit))
cols[4].metric("Est.Break-Even Period", break_even_estimate)
cols[5].metric("Stock Purchased (Items)", f"{stock_purchased_qty}")
cols[6].metric("Stock Left (Items)", f"{stock_items}")

# Stock Table with totals
st.subheader("📦 Stock Purchased")
inventory_totals = pd.DataFrame([{
    "Item": "Total",
    "Quantity Purchased": inventory["Quantity Purchased"].sum(),
    "Quantity Left": inventory["Quantity Left"].sum(),
    "Unit Cost": "",
    "Total Cost": inventory["Total Cost"].sum()
}])
st.dataframe(pd.concat([inventory, inventory_totals], ignore_index=True)[["Item", "Quantity Purchased", "Quantity Left", "Unit Cost", "Total Cost"]])

# Sales & Profit
st.subheader("🧾 Sales & Profit")
def get_unit_cost(row):
    match = inventory[inventory.Item == row.Item]
    return match['Unit Cost'].iloc[0] if not match.empty else 0

sales['Unit Cost'] = sales.apply(get_unit_cost, axis=1)
sales['Total Revenue'] = sales['Quantity Sold'] * sales['Selling Price']
sales['Profit'] = (sales['Selling Price'] - sales['Unit Cost']) * sales['Quantity Sold']

sales_totals = pd.DataFrame([{
    "Date": "Total",
    "Buyer Name": "",
    "Item": "",
    "Quantity Sold": sales["Quantity Sold"].sum(),
    "Unit Cost": "",
    "Selling Price": "",
    "Total Revenue": sales["Total Revenue"].sum(),
    "Profit": sales["Profit"].sum()
}])
st.dataframe(pd.concat([sales, sales_totals], ignore_index=True)[["Date", "Buyer Name", "Item", "Quantity Sold", "Unit Cost", "Selling Price", "Total Revenue", "Profit"]])

# Add Sale
st.subheader("➕ Add a Sale")
with st.form("Add Sale"):
    item = st.selectbox("Item", inventory["Item"].unique())
    buyer = st.text_input("Buyer Name")
    qty = st.number_input("Quantity Sold", min_value=1)
    price = st.number_input("Selling Price", min_value=0.01, format="%.2f")
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

# Manage Sales
st.subheader("🗑️ Manage Sales Entries")
password = st.session_state.get("delete_password", None)

for i, row in sales.iterrows():
    col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 1, 2, 2, 2])
    col1.write(row["Date"])
    col2.write(row["Item"])
    col3.write(f"{row['Quantity Sold']} pcs")
    col4.write(f"₹{row['Selling Price']}")
    col5.write(f"₹{row['Profit']}")

    delete_trigger = col6.button("Delete", key=f"delete_btn_{i}")

    if delete_trigger:
        st.session_state[f"show_pw_{i}"] = True

    if st.session_state.get(f"show_pw_{i}", False):
        pw = col6.text_input("Password", type="password", key=f"pw_input_{i}")
        if pw == "admin123":  # secure this in production!
            inventory.loc[inventory.Item == row["Item"], "Quantity Left"] += row["Quantity Sold"]
            sales = sales.drop(index=i).reset_index(drop=True)
            sales.to_csv("data/sales.csv", index=False)
            inventory.to_csv("data/inventory.csv", index=False)
            st.success("Sale deleted! Please refresh the page manually.")
            st.stop()
        elif pw:
            st.error("Incorrect password.")


# Profit Sharing
st.subheader("🤝 Profit Sharing")

latest_period = pd.to_datetime(date.today()).strftime("%Y-%m")
man_a_net = profit * 0.42
man_b_total = profit * 0.40
man_c_share = profit * 0.18

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

# Format and display
display_profit_df = profit_df.copy()
display_profit_df["Total Profit"] = display_profit_df["Total Profit"].map(lambda x: f"₹{x:,.2f}")
display_profit_df["Man A (Net)"] = display_profit_df["Man A (Net)"].map(lambda x: f"₹{x:,.2f}")
display_profit_df["Man B"] = display_profit_df["Man B"].map(lambda x: f"₹{x:,.2f}")
display_profit_df["Man C (From A)"] = display_profit_df["Man C (From A)"].map(lambda x: f"₹{x:,.2f}")

st.dataframe(
    display_profit_df[["Period", "Total Profit", "Man A (Net)", "Man B", "Man C (From A)"]],
    use_container_width=True,
    hide_index=True
)
