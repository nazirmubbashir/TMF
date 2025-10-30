# app.py
import streamlit as st
import pandas as pd
from datetime import date, datetime
from pathlib import Path

st.set_page_config("TMF Accounting Dashboard", layout="wide")

# ------------------------------------------------------------------
# Paths & constants
# ------------------------------------------------------------------
DATA_DIR = Path("data")
INV_PATH = DATA_DIR / "inventory.csv"     # one row = one purchase lot (exact unit cost kept)
SALES_PATH = DATA_DIR / "sales.csv"       # one row = one sale entry
PROFIT_PATH = DATA_DIR / "profit.csv"

ADMIN_PW = "admin123"
TOTAL_INVESTMENT = 10_000_000  # ₹1 Cr (shared across Crown + WD)

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def ensure_data_files():
    DATA_DIR.mkdir(exist_ok=True)
    if not INV_PATH.exists():
        pd.DataFrame(columns=[
            "Item","Category","Lot ID","Purchase Date","Quantity Purchased","Quantity Left","Unit Cost"
        ]).to_csv(INV_PATH, index=False)
    if not SALES_PATH.exists():
        pd.DataFrame(columns=["Date","Item","Category","Quantity Sold","Selling Price"]).to_csv(SALES_PATH, index=False)
    if not PROFIT_PATH.exists():
        pd.DataFrame(columns=["Period","Total Profit","Man A (Net)","Man B","Man C (From A)"]).to_csv(PROFIT_PATH, index=False)

def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

def to_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return int(default)

def money(x):
    try:
        return f"₹{float(x):,.2f}"
    except:
        return ""

def shorten_currency(amount):
    try:
        amount = float(amount)
    except:
        return "₹0"
    if amount < 1_00_00_000:
        return f"₹{amount/1_00_000:.1f}L"
    else:
        return f"₹{amount/1_00_00_000:.1f}Cr"

def next_lot_id_for(inv_df, item, category):
    subset = inv_df[(inv_df["Item"]==item) & (inv_df["Category"]==category)]
    if subset.empty or "Lot ID" not in subset.columns:
        return 1
    return int(subset["Lot ID"].max()) + 1

def parse_date_safe(s):
    if pd.isna(s) or s is None or str(s).strip()=="":
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(str(s).strip(), fmt).date()
        except Exception:
            continue
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None

# ----------------------- FIFO core ---------------------------------
def replay_fifo_and_inventory(inv_df: pd.DataFrame, sales_df: pd.DataFrame):
    if inv_df.empty:
        cogs_per_row = [0.0] * len(sales_df)
        inv_left = inv_df.copy()
        return cogs_per_row, inv_left

    lots = inv_df.copy()
    for col in ["Quantity Purchased","Quantity Left","Unit Cost","Lot ID"]:
        if col in lots.columns:
            if col == "Unit Cost":
                lots[col] = pd.to_numeric(lots[col], errors="coerce").fillna(0.0).astype(float)
            else:
                lots[col] = pd.to_numeric(lots[col], errors="coerce").fillna(0).astype(int)
    if "Purchase Date" in lots.columns:
        lots["_pdate"] = lots["Purchase Date"].apply(parse_date_safe)
    else:
        lots["_pdate"] = None
    lots = lots.sort_values(by=["Item","Category","_pdate","Lot ID"], kind="stable").reset_index(drop=True)
    lots["remaining"] = lots["Quantity Purchased"]  # recompute fresh

    if not sales_df.empty:
        s = sales_df.copy()
        s["_sdate"] = s["Date"].apply(parse_date_safe)
        s = s.sort_values(by=["_sdate"]).reset_index(drop=False)  # keep original position in "index"
    else:
        s = pd.DataFrame(columns=list(sales_df.columns)+["_sdate"])
        s["_sdate"] = None
        s = s.reset_index(drop=False)

    cogs_map = {}
    for _, srow in s.iterrows():
        item = srow.get("Item")
        cat = srow.get("Category")
        qty = to_int(srow.get("Quantity Sold"), 0)
        if qty <= 0:
            cogs_map[srow["index"]] = 0.0
            continue
        mask = (lots["Item"]==item) & (lots["Category"]==cat)
        use_lots = lots[mask]
        if use_lots.empty:
            cogs_map[srow["index"]] = 0.0
            continue
        cost_total = 0.0
        qty_left_to_consume = qty
        for li in use_lots.index:
            if qty_left_to_consume <= 0:
                break
            lot_remaining = int(lots.at[li, "remaining"])
            if lot_remaining <= 0:
                continue
            consume = min(lot_remaining, qty_left_to_consume)
            cost_total += consume * float(lots.at[li, "Unit Cost"])
            lots.at[li, "remaining"] = lot_remaining - consume
            qty_left_to_consume -= consume
        cogs_map[srow["index"]] = float(cost_total)

    lots_out = lots.copy()
    lots_out["Quantity Left"] = lots_out["remaining"]
    lots_out = lots_out.drop(columns=["remaining","_pdate"], errors="ignore")

    cogs_per_row = []
    for i in sales_df.index:
        cogs_per_row.append(cogs_map.get(i, 0.0))

    return cogs_per_row, lots_out

def recompute_and_persist_inventory_left(inv_df: pd.DataFrame, sales_df: pd.DataFrame):
    _, lots_out = replay_fifo_and_inventory(inv_df, sales_df)
    lots_out.to_csv(INV_PATH, index=False)
    return lots_out

# ------------------------------------------------------------------
# Load & migrate data
# ------------------------------------------------------------------
ensure_data_files()
inventory = pd.read_csv(INV_PATH)
sales = pd.read_csv(SALES_PATH)
profit_df = pd.read_csv(PROFIT_PATH)

if "Category" not in inventory.columns:
    inventory["Category"] = "Crown"
if "Lot ID" not in inventory.columns:
    inventory["Lot ID"] = 0
    for (it, cat), idxs in inventory.groupby(["Item","Category"]).groups.items():
        for j, idx in enumerate(sorted(idxs), start=1):
            inventory.loc[idx, "Lot ID"] = j
if "Purchase Date" not in inventory.columns:
    inventory["Purchase Date"] = str(date.today())
for col in ["Quantity Purchased","Quantity Left"]:
    inventory[col] = pd.to_numeric(inventory.get(col, 0), errors="coerce").fillna(0).astype(int)
inventory["Unit Cost"] = pd.to_numeric(inventory.get("Unit Cost", 0.0), errors="coerce").fillna(0.0).astype(float)

if "Category" not in sales.columns:
    item_to_cat = dict(zip(inventory["Item"], inventory["Category"]))
    sales["Category"] = sales["Item"].map(item_to_cat).fillna("Crown")
if "Quantity Sold" not in sales.columns:
    sales["Quantity Sold"] = 0
if "Selling Price" not in sales.columns:
    sales["Selling Price"] = 0.0
for col in ["Quantity Sold","Selling Price"]:
    sales[col] = pd.to_numeric(sales[col], errors="coerce").fillna(0.0)
sales["Quantity Sold"] = sales["Quantity Sold"].astype(int)

inventory = recompute_and_persist_inventory_left(inventory, sales)

# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
def compute_metrics(inv_df, sales_df, investment_total=None):
    inv_df = inv_df.copy()
    inv_df["Total Cost"] = inv_df["Quantity Purchased"] * inv_df["Unit Cost"]
    # FIFO COGS for all sales
    if not sales_df.empty:
        cogs_list, _ = replay_fifo_and_inventory(inv_df, sales_df)
        total_cogs = sum(cogs_list)
        total_sales_revenue = float((sales_df["Quantity Sold"] * sales_df["Selling Price"]).sum())
    else:
        total_cogs = 0.0
        total_sales_revenue = 0.0
    profit = total_sales_revenue - total_cogs
    stock_value = (inv_df["Quantity Left"] * inv_df["Unit Cost"]).sum()
    stock_items = int(inv_df["Quantity Left"].sum())
    out = {"profit": profit, "stock_left_qty": stock_items, "stock_value": stock_value}
    if investment_total is not None:
        # Revamped capital logic: Utilised = current stock value; Left = total - stock value
        investment_used = stock_value
        investment_left = investment_total - investment_used
        # Break-even (rough): based on realised profit per distinct sales day
        if not sales_df.empty and "Date" in sales_df.columns:
            try:
                days = pd.to_datetime(sales_df["Date"]).dt.date.nunique()
            except Exception:
                days = sales_df["Date"].nunique()
        else:
            days = 0
        avg_daily_profit = profit / max(1, days)
        break_even_days = (investment_total - profit) / avg_daily_profit if avg_daily_profit > 0 else float("inf")
        break_even_estimate = f"{int(break_even_days)} days" if break_even_days != float("inf") else "Not yet profitable"
        out.update({
            "investment_total": investment_total,
            "investment_utilised": investment_used,
            "investment_left": investment_left,
            "break_even": break_even_estimate,
        })
    return out

def metrics_block(m, show_investment=True):
    with st.expander("📈 Metrics", expanded=True):
        if show_investment:
            cols = st.columns(6)
            cols[0].metric("Total Investment", shorten_currency(m["investment_total"]))
            cols[1].metric("Investment Utilised (Stock Value)", shorten_currency(m["investment_utilised"]))
            cols[2].metric("Investment Left (Capital Free)", shorten_currency(m["investment_left"]))
            cols[3].metric("Total Profit", shorten_currency(m["profit"]))
            cols[4].metric("Est. Break-Even Period", m["break_even"])
            cols[5].metric("Stock Left (Items)", f"{m['stock_left_qty']}")
        else:
            cols = st.columns(2)
            cols[0].metric("Total Profit", shorten_currency(m["profit"]))
            cols[1].metric("Stock Left (Items)", f"{m['stock_left_qty']}")
            # Stock Value (cost) intentionally hidden in category view per request


def stock_table(inv_df):
    inv_df = inv_df.copy()
    # Compute Total Cost at lot level
    inv_df["Total Cost"] = inv_df["Quantity Purchased"] * inv_df["Unit Cost"]

    # Prepare formatted table
    inv_fmt = inv_df.copy()
    inv_fmt["Unit Cost"] = inv_fmt["Unit Cost"].map(money)
    inv_fmt["Total Cost"] = inv_fmt["Total Cost"].map(money)

    # Compute totals (cost-only, no profit)
    total_purchased = int(inv_df["Quantity Purchased"].sum()) if not inv_df.empty else 0
    total_left = int(inv_df["Quantity Left"].sum()) if not inv_df.empty else 0
    total_cost_sum = float(inv_df["Total Cost"].sum()) if not inv_df.empty else 0.0

    totals_row = {
        "Item": "Total",
        "Category": "",
        "Lot ID": "",
        "Purchase Date": "",
        "Quantity Purchased": total_purchased,
        "Quantity Left": total_left,
        "Unit Cost": "",
        "Total Cost": money(total_cost_sum),
    }

    inv_fmt_with_total = (
        pd.concat([inv_fmt, pd.DataFrame([totals_row])], ignore_index=True)
        if not inv_fmt.empty else pd.DataFrame([totals_row])
    )

    st.dataframe(
        inv_fmt_with_total[
            ["Item","Category","Lot ID","Purchase Date","Quantity Purchased","Quantity Left","Unit Cost","Total Cost"]
        ],
        use_container_width=True,
        hide_index=True
    )


def sales_profit_table(inv_df, sales_df):
    if sales_df.empty:
        st.info("No sales yet.")
        return
    cogs_list, _ = replay_fifo_and_inventory(inv_df, sales_df)
    s = sales_df.copy().reset_index(drop=True)
    s["COGS (FIFO)"] = pd.Series(cogs_list).astype(float)
    unit_cost_per_sale = []
    for i, r in s.iterrows():
        qty = max(1, int(r["Quantity Sold"]))
        unit_cost_per_sale.append(float(s.at[i,"COGS (FIFO)"]) / qty)
    s["Unit Cost (Exact)"] = unit_cost_per_sale
    s["Total Revenue"] = s["Quantity Sold"] * s["Selling Price"]
    s["Profit"] = s["Total Revenue"] - s["COGS (FIFO)"]

    s_fmt = s.copy()
    for c in ["Selling Price","Unit Cost (Exact)","Total Revenue","COGS (FIFO)","Profit"]:
        s_fmt[c] = s_fmt[c].map(money)

    totals = {
        "Date": "Total",
        "Item": "",
        "Category": "",
        "Quantity Sold": int(s["Quantity Sold"].sum() if not s.empty else 0),
        "Selling Price": "",
        "Unit Cost (Exact)": "",
        "COGS (FIFO)": money(s["COGS (FIFO)"].sum() if not s.empty else 0.0),
        "Total Revenue": money(s["Total Revenue"].sum() if not s.empty else 0.0),
        "Profit": money(s["Profit"].sum() if not s.empty else 0.0),
    }
    s_fmt_tot = pd.concat([s_fmt, pd.DataFrame([totals])], ignore_index=True)

    st.dataframe(
        s_fmt_tot[["Date","Item","Category","Quantity Sold","Selling Price","Unit Cost (Exact)","COGS (FIFO)","Total Revenue","Profit"]],
        use_container_width=True,
        hide_index=True
    )

# ------------------------------------------------------------------
# Manage sections (inline Edit + Delete) — SESSION-STATE FIX
# ------------------------------------------------------------------
def manage_purchases_section(inv_cat, category_name):
    exp_key = f"exp_pur_{category_name}"
    with st.expander("🛠️ Manage Purchases (per-lot)", expanded=st.session_state.get(exp_key, False)):
        if inv_cat.empty:
            st.info("No purchases found for this category.")
            return

        inv_rows = inv_cat.reset_index(drop=True)

        for i, row in inv_rows.iterrows():
            rkey = f"{category_name}_pur_{int(row['Lot ID'])}_{i}"
            edit_key = f"edit_{rkey}"
            del_key  = f"del_{rkey}"
            info_key = f"info_{rkey}"

            if edit_key not in st.session_state:
                st.session_state[edit_key] = False
            if del_key not in st.session_state:
                st.session_state[del_key] = False

            c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1.8, 1.6, 1.2, 1.2, 1.2, 0.9, 0.9, 0.9])
            c1.write(f"**{row['Item']}**  • Lot #{int(row['Lot ID'])}")
            c2.write(f"{row['Category']}")
            c3.write(f"Purchased: {int(row['Quantity Purchased'])}")
            c4.write(f"Left: {int(row['Quantity Left'])}")
            c5.write(f"Cost/pc: {money(row['Unit Cost'])}")

            if c6.button("Edit", key=f"edit_btn_{rkey}_{i}"):
                st.session_state[edit_key] = not st.session_state[edit_key]
                st.session_state[del_key] = False
                st.session_state[exp_key] = True
                st.rerun()

            if c7.button("Delete", key=f"del_btn_{rkey}_{i}"):
                st.session_state[del_key] = not st.session_state[del_key]
                st.session_state[edit_key] = False
                st.session_state[exp_key] = True
                st.rerun()

            if c8.button("Info", key=f"info_btn_{rkey}_{i}"):
                st.info(f"Purchase Date: {row.get('Purchase Date','')}")

            # --- Edit form (persistent) ---
            if st.session_state[edit_key]:
                with st.form(f"edit_pur_form_{rkey}_{i}", clear_on_submit=True):
                    new_name = st.text_input("Model Name", value=str(row["Item"]), key=f"name_{rkey}_{i}")
                    purchased_qty = int(row["Quantity Purchased"])
                    left_qty = int(row["Quantity Left"])
                    consumed = purchased_qty - left_qty  # cannot reduce below this
                    new_qty = st.number_input(
                        "Quantity Purchased (total for this lot)",
                        min_value=int(consumed),
                        value=purchased_qty,
                        step=1,
                        help=f"Minimum {consumed} (already sold from this lot).",
                        key=f"qty_{rkey}_{i}"
                    )
                    new_cost = st.number_input("Unit Cost (₹)", min_value=0.0, value=float(row["Unit Cost"]), key=f"cost_{rkey}_{i}")
                    new_date = st.text_input("Purchase Date (YYYY-MM-DD)", value=str(row.get("Purchase Date","")), key=f"date_{rkey}_{i}")
                    submitted = st.form_submit_button("Save changes")
                    if submitted:
                        mask_row = (
                            (inventory["Item"]==row["Item"]) &
                            (inventory["Category"]==row["Category"]) &
                            (inventory["Lot ID"]==row["Lot ID"])
                        )
                        if not mask_row.any():
                            st.error("Purchase lot not found.")
                            st.stop()
                        new_qty_int = int(new_qty)
                        new_left = max(0, new_qty_int - consumed)
                        inventory.loc[mask_row, "Item"] = str(new_name).strip()
                        inventory.loc[mask_row, "Unit Cost"] = float(new_cost)
                        inventory.loc[mask_row, "Purchase Date"] = new_date.strip() if str(new_date).strip() else str(date.today())
                        inventory.loc[mask_row, "Quantity Purchased"] = new_qty_int
                        inventory.loc[mask_row, "Quantity Left"] = new_left
                        inventory.to_csv(INV_PATH, index=False)
                        st.success("Purchase lot updated.")
                        st.session_state[edit_key] = False
                        st.session_state[exp_key] = True
                        st.rerun()

            # --- Delete form (persistent) ---
            if st.session_state[del_key]:
                with st.form(f"del_pur_form_{rkey}_{i}", clear_on_submit=True):
                    pw = st.text_input("Enter password to delete this lot", type="password", key=f"pw_{rkey}_{i}")
                    confirm = st.checkbox("I understand this will remove this purchase lot permanently.", key=f"conf_{rkey}_{i}")
                    do_del = st.form_submit_button("Confirm Delete")
                    if do_del:
                        if (pw or "").strip() != ADMIN_PW:
                            st.error("Incorrect password.")
                        elif not confirm:
                            st.error("Please confirm deletion.")
                        else:
                            if int(row["Quantity Left"]) < int(row["Quantity Purchased"]):
                                st.error("Cannot delete: some quantity from this lot has been sold.")
                            else:
                                mask_row = (
                                    (inventory["Item"]==row["Item"]) &
                                    (inventory["Category"]==row["Category"]) &
                                    (inventory["Lot ID"]==row["Lot ID"])
                                )
                                if not mask_row.any():
                                    st.error("Lot not found (already removed).")
                                else:
                                    inventory.drop(index=inventory[mask_row].index, inplace=True)
                                    inventory.to_csv(INV_PATH, index=False)
                                    st.success("Purchase lot deleted.")
                                    st.session_state[del_key] = False
                                    st.session_state[exp_key] = True
                                    st.rerun()

def manage_sales_section(inv_cat, category_name):
    exp_key_sales = f"exp_sales_{category_name}"
    with st.expander("🗑️ Manage Sales Entries", expanded=st.session_state.get(exp_key_sales, False)):
        sales_cat = sales[sales["Category"] == category_name].copy()
        if sales_cat.empty:
            st.info("No sales recorded yet for this category.")
            return

        cogs_list, _ = replay_fifo_and_inventory(inv_cat, sales_cat)
        sales_cat = sales_cat.reset_index(drop=False)  # keep original index for locating in full df

        for i, row in sales_cat.iterrows():
            rkey = f"{category_name}_sale_{int(row['index'])}_{i}"
            edit_key = f"edit_{rkey}"
            del_key  = f"del_{rkey}"

            if edit_key not in st.session_state:
                st.session_state[edit_key] = False
            if del_key not in st.session_state:
                st.session_state[del_key] = False

            c1, c2, c3, c4, c5, c6, c7 = st.columns([1.6, 2.4, 1.2, 1.3, 1.4, 0.9, 0.9])
            c1.write(row["Date"])
            c2.write(f"{row['Item']} ({row['Category']})")
            c3.write(f"{int(row['Quantity Sold'])} pcs")
            c4.write(money(row["Selling Price"]))
            row_profit = float(row["Quantity Sold"]) * float(row["Selling Price"]) - float(cogs_list[i])
            c5.write(money(row_profit))

            if c6.button("Edit", key=f"edit_btn_{rkey}_{i}"):
                st.session_state[edit_key] = not st.session_state[edit_key]
                st.session_state[del_key] = False
                st.session_state[exp_key_sales] = True
                st.rerun()

            if c7.button("Delete", key=f"del_btn_{rkey}_{i}"):
                st.session_state[del_key] = not st.session_state[del_key]
                st.session_state[edit_key] = False
                st.session_state[exp_key_sales] = True
                st.rerun()

            # --- Edit sale (persistent) ---
            if st.session_state[edit_key]:
                with st.form(f"edit_sale_form_{rkey}_{i}", clear_on_submit=True):
                    new_qty = st.number_input("Quantity Sold", min_value=1, step=1, value=int(row["Quantity Sold"]), key=f"qty_sale_{rkey}_{i}")
                    new_sp  = st.number_input("Selling Price (₹/unit)", min_value=0.01, format="%.2f", value=float(row["Selling Price"]), key=f"sp_sale_{rkey}_{i}")
                    do_save = st.form_submit_button("Save changes")
                    if do_save:
                        master_idx = row["index"]
                        if master_idx in sales.index:
                            sales.loc[master_idx, "Quantity Sold"] = int(new_qty)
                            sales.loc[master_idx, "Selling Price"] = float(new_sp)
                            sales.to_csv(SALES_PATH, index=False)
                            recompute_and_persist_inventory_left(inventory, sales)
                            st.success("Sale updated.")
                            st.session_state[edit_key] = False
                            st.session_state[exp_key_sales] = True
                            st.rerun()
                        else:
                            st.error("Could not locate the sale entry to edit.")
                            st.rerun()

            # --- Delete sale (persistent) ---
            if st.session_state[del_key]:
                with st.form(f"del_sale_form_{rkey}_{i}", clear_on_submit=True):
                    pw = st.text_input("Enter password to delete sale", type="password", key=f"pw_sale_{rkey}_{i}")
                    do_del = st.form_submit_button("Confirm Delete")
                    if do_del:
                        if (pw or "").strip() != ADMIN_PW:
                            st.error("Incorrect password.")
                        else:
                            master_idx = row["index"]
                            if master_idx in sales.index:
                                sales.drop(index=master_idx, inplace=True)
                                sales.to_csv(SALES_PATH, index=False)
                                recompute_and_persist_inventory_left(inventory, sales)
                                st.success("Sale deleted and stock returned via FIFO recompute.")
                                st.session_state[del_key] = False
                                st.session_state[exp_key_sales] = True
                                st.rerun()
                            else:
                                st.error("Could not locate the sale entry.")
                                st.rerun()

# ------------------------------------------------------------------
# Category section
# ------------------------------------------------------------------
def category_section(cat_name):
    st.header(cat_name)
    inv_cat = inventory[inventory["Category"] == cat_name].copy()
    sales_cat = sales[sales["Category"] == cat_name].copy()
    m = compute_metrics(inv_cat, sales_cat, investment_total=None)
    m["stock_value"] = (inv_cat["Quantity Left"] * inv_cat["Unit Cost"]).sum()
    metrics_block(m, show_investment=False)

    with st.expander("📦 Stock Purchased (per-lot view)", expanded=False):
        stock_table(inv_cat)

    with st.expander("➕ Add Purchase (New Model or Initial Stock)", expanded=False):
        with st.form(f"add_purchase_{cat_name}"):
            item_new = st.text_input("Model Name (new)")
            qty_new = st.number_input("Quantity Purchased", min_value=1, step=1)
            unit_cost_new = st.number_input("Unit Cost (₹ per unit)", min_value=0.0, format="%.2f")
            pdate = st.text_input("Purchase Date (YYYY-MM-DD)", value=str(date.today()))
            submit_new = st.form_submit_button("Add Purchase")
            if submit_new:
                if item_new.strip() == "":
                    st.error("Please enter a model name.")
                else:
                    lot_id = next_lot_id_for(inventory, item_new.strip(), cat_name)
                    new_row = {
                        "Item": item_new.strip(),
                        "Category": cat_name,
                        "Lot ID": int(lot_id),
                        "Purchase Date": pdate.strip() if pdate.strip() else str(date.today()),
                        "Quantity Purchased": int(qty_new),
                        "Quantity Left": int(qty_new),
                        "Unit Cost": float(unit_cost_new),
                    }
                    inv_updated = pd.concat([inventory, pd.DataFrame([new_row])], ignore_index=True)
                    inv_updated.to_csv(INV_PATH, index=False)
                    st.success(f"Added lot #{lot_id} for '{item_new}' in {cat_name}.")
                    st.rerun()

    with st.expander("🔁 Restock Existing Model (creates new lot)", expanded=False):
        existing_items = inv_cat["Item"].unique().tolist()
        if len(existing_items) == 0:
            st.info("No items yet in this category. Add a purchase above first.")
        else:
            with st.form(f"restock_{cat_name}"):
                item_sel = st.selectbox("Select Model", existing_items, key=f"restock_select_{cat_name}")
                add_qty = st.number_input("Additional Quantity", min_value=1, step=1, key=f"restock_qty_{cat_name}")
                add_cost = st.number_input("Unit Cost for New Lot (₹/unit)", min_value=0.0, format="%.2f", key=f"restock_cost_{cat_name}")
                pdate = st.text_input("Purchase Date (YYYY-MM-DD)", value=str(date.today()))
                do_restock = st.form_submit_button("Add Restock Lot")
                if do_restock:
                    lot_id = next_lot_id_for(inventory, item_sel, cat_name)
                    new_row = {
                        "Item": item_sel,
                        "Category": cat_name,
                        "Lot ID": int(lot_id),
                        "Purchase Date": pdate.strip() if pdate.strip() else str(date.today()),
                        "Quantity Purchased": int(add_qty),
                        "Quantity Left": int(add_qty),
                        "Unit Cost": float(add_cost),
                    }
                    inv_updated = pd.concat([inventory, pd.DataFrame([new_row])], ignore_index=True)
                    inv_updated.to_csv(INV_PATH, index=False)
                    st.success(f"Restocked '{item_sel}' with new lot #{lot_id} (+{int(add_qty)} pcs) at {money(add_cost)} each.")
                    st.rerun()

    manage_purchases_section(inv_cat, cat_name)

    with st.expander("🧾 Sales & Profit (FIFO exact costing)", expanded=False):
        sales_profit_table(inv_cat, sales_cat)

    with st.expander("🛒 Add a Sale", expanded=False):
        items_for_sale = inv_cat.groupby("Item")["Quantity Left"].sum()
        items_for_sale = items_for_sale[items_for_sale > 0].index.tolist()
        if len(items_for_sale) == 0:
            st.info("No stock available to sell in this category.")
        else:
            with st.form(f"add_sale_{cat_name}"):
                item_sale = st.selectbox("Item", items_for_sale, key=f"sale_select_{cat_name}")
                qty_sale = st.number_input("Quantity Sold", min_value=1, step=1, key=f"sale_qty_{cat_name}")
                price_sale = st.number_input("Selling Price (₹/unit)", min_value=0.01, format="%.2f", key=f"sale_price_{cat_name}")
                submit_sale = st.form_submit_button("Record Sale")
                if submit_sale:
                    current_left_total = int(inv_cat[inv_cat["Item"]==item_sale]["Quantity Left"].sum())
                    if qty_sale > current_left_total:
                        st.error(f"Not enough stock. Only {current_left_total} left across lots.")
                    else:
                        new_sale = {
                            "Date": str(date.today()),
                            "Item": item_sale,
                            "Category": cat_name,
                            "Quantity Sold": int(qty_sale),
                            "Selling Price": float(price_sale)
                        }
                        sales_updated = pd.concat([sales, pd.DataFrame([new_sale])], ignore_index=True)
                        sales_updated.to_csv(SALES_PATH, index=False)
                        recompute_and_persist_inventory_left(inventory, sales_updated)
                        sales[:] = sales_updated
                        st.success("Sale recorded via FIFO costing!")
                        st.rerun()

    manage_sales_section(inv_cat, cat_name)

def build_fifo_ledger(inv_df: pd.DataFrame, sales_df: pd.DataFrame) -> pd.DataFrame:
    if inv_df.empty or sales_df.empty:
        return pd.DataFrame(columns=[
            "s_index","Date","Item","Category",
            "Lot ID","Purchase Date","qty","sale_price_unit","lot_cost_unit"
        ])
    lots = inv_df.copy()
    for col in ["Quantity Purchased","Quantity Left","Unit Cost","Lot ID"]:
        lots[col] = pd.to_numeric(lots.get(col, 0), errors="coerce").fillna(0)
    lots["Quantity Purchased"] = lots["Quantity Purchased"].astype(int)
    lots["Quantity Left"] = lots["Quantity Left"].astype(int)
    lots["Unit Cost"] = lots["Unit Cost"].astype(float)
    lots["Lot ID"] = lots["Lot ID"].astype(int)
    lots["_pdate"] = lots["Purchase Date"].apply(parse_date_safe)
    lots = lots.sort_values(by=["Item","Category","_pdate","Lot ID"], kind="stable").reset_index(drop=True)
    lots["remaining"] = lots["Quantity Purchased"]

    s = sales_df.copy()
    s["_sdate"] = s["Date"].apply(parse_date_safe)
    s = s.sort_values(by=["_sdate"]).reset_index(drop=False)

    rows = []
    for _, srow in s.iterrows():
        item = srow.get("Item")
        cat  = srow.get("Category")
        qty  = int(srow.get("Quantity Sold", 0))
        spu  = float(srow.get("Selling Price", 0.0))
        if qty <= 0: continue

        mask = (lots["Item"] == item) & (lots["Category"] == cat)
        use_lots = lots[mask]
        if use_lots.empty: continue

        qty_left = qty
        for li in use_lots.index:
            if qty_left <= 0: break
            lot_remaining = int(lots.at[li, "remaining"])
            if lot_remaining <= 0: continue
            take = min(lot_remaining, qty_left)
            rows.append({
                "s_index": srow["index"],
                "Date": srow["Date"],
                "Item": item,
                "Category": cat,
                "Lot ID": int(lots.at[li, "Lot ID"]),
                "Purchase Date": lots.at[li, "Purchase Date"],
                "qty": int(take),
                "sale_price_unit": float(spu),
                "lot_cost_unit": float(lots.at[li, "Unit Cost"]),
            })
            lots.at[li, "remaining"] = lot_remaining - take
            qty_left -= take

    return pd.DataFrame(rows)

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("📊 TMF Accounting Dashboard")
st.caption("Two categories: **Crown** and **WD** — separate purchases/sales with **per-lot exact costing (FIFO)**. ₹1 Cr investment is **shared** and shown on the **All** tab.")

tab_all, tab_crown, tab_wd = st.tabs(["All (Combined)", "Crown", "WD"])

# ------------------------------------------------------------------
# All (Combined) tab (Detailed-only)
# ------------------------------------------------------------------
with tab_all:
    # --- Detailed Financials (All) ---
    with st.expander("💼 Detailed Financials (All)", expanded=False):
        # Capital Deployed = sum of all historical purchase spend
        capital_deployed = float((inventory["Quantity Purchased"] * inventory["Unit Cost"]).sum()) if not inventory.empty else 0.0
        # Capital Recovered = all sales revenue
        total_revenue_all = float((sales["Quantity Sold"] * sales["Selling Price"]).sum()) if not sales.empty else 0.0
        # Compute profit using FIFO (same approach as earlier)
        if not sales.empty:
            cogs_list_all, _ = replay_fifo_and_inventory(inventory, sales)
            total_cogs_all = float(sum(cogs_list_all))
        else:
            total_cogs_all = 0.0
        total_profit_all = total_revenue_all - total_cogs_all
        # Current Capital Invested = Deployed - Recovered (revenue-based; change to stock value or COGS if desired)
        current_capital_invested = max(0.0, capital_deployed - total_revenue_all)
        # Cash-on-Cash Return = Profit / Deployed
        coc = (total_profit_all / capital_deployed * 100.0) if capital_deployed > 0 else 0.0

        cols1 = st.columns(4)
        cols1[0].metric("Total Investment Committed", shorten_currency(TOTAL_INVESTMENT))
        cols1[1].metric("Capital Deployed", shorten_currency(capital_deployed))
        cols1[2].metric("Capital Recovered", shorten_currency(total_revenue_all))
        cols1[3].metric("Current Capital Invested", shorten_currency(current_capital_invested))

        cols2 = st.columns(4)
        cols2[0].metric("Total Profit", shorten_currency(total_profit_all))
        # For consistency we can reuse break-even computed from combined metrics
        m_all_tmp = compute_metrics(inventory, sales, investment_total=TOTAL_INVESTMENT)
        cols2[1].metric("Est. Break-Even Period", m_all_tmp["break_even"])
        cols2[2].metric("Cash-on-Cash Return", f"{coc:.2f}%")
        cols2[3].metric("Stock Left (Items)", f"{m_all_tmp['stock_left_qty']}")

    # Keep the other All-level tables
    with st.expander("📦 Stock Purchased — All (per-lot)", expanded=False):
        stock_table(inventory)
    with st.expander("🧾 Sales & Profit — All (FIFO costing)", expanded=False):
        sales_profit_table(inventory, sales)
    # --- Profit Sharing — Monthly (no duplicates) ---
    ledger_all = build_fifo_ledger(inventory, sales)
    with st.expander("🤝 Profit Sharing — Monthly (no duplicates)", expanded=False):
        if ledger_all.empty:
            st.info("No sales yet.")
        else:
            df = ledger_all.copy()
            df["revenue"] = df["qty"] * df["sale_price_unit"]
            df["cogs"]    = df["qty"] * df["lot_cost_unit"]
            df["profit"]  = df["revenue"] - df["cogs"]
            df["Period"]  = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m")

            monthly = df.groupby("Period", as_index=False).agg(Total_Profit=("profit","sum"))
            monthly["Man A (Net)"]    = monthly["Total_Profit"] * 0.42
            monthly["Man B"]          = monthly["Total_Profit"] * 0.40
            monthly["Man C (From A)"] = monthly["Total_Profit"] * 0.18

            disp = monthly.copy()
            for c in ["Total_Profit","Man A (Net)","Man B","Man C (From A)"]:
                disp[c] = disp[c].map(money)

            st.dataframe(
                disp.rename(columns={"Total_Profit":"Total Profit"})[
                    ["Period","Total Profit","Man A (Net)","Man B","Man C (From A)"]
                ],
                use_container_width=True, hide_index=True
            )

    # --- Profit Sharing — By Lot (FIFO exact) ---
    with st.expander("📦 Profit Sharing — By Lot (FIFO exact)", expanded=False):
        if ledger_all.empty:
            st.info("No sales yet.")
        else:
            lotdf = ledger_all.copy()
            lotdf["revenue"] = lotdf["qty"] * lotdf["sale_price_unit"]
            lotdf["cogs"]    = lotdf["qty"] * lotdf["lot_cost_unit"]
            lotdf["profit"]  = lotdf["revenue"] - lotdf["cogs"]

            bylot = lotdf.groupby(["Item","Category","Lot ID","Purchase Date"], as_index=False).agg(
                Qty_Sold=("qty","sum"),
                Revenue=("revenue","sum"),
                COGS=("cogs","sum"),
                Profit=("profit","sum"),
            )

            inv_min = inventory[["Item","Category","Lot ID","Quantity Purchased","Quantity Left","Unit Cost","Purchase Date"]]
            bylot = bylot.merge(inv_min, on=["Item","Category","Lot ID","Purchase Date"], how="left")

            bylot["Man A (Net)"]    = bylot["Profit"] * 0.42
            bylot["Man B"]          = bylot["Profit"] * 0.40
            bylot["Man C (From A)"] = bylot["Profit"] * 0.18

            totals = {
                "Item": "Total", "Category": "", "Lot ID": "", "Purchase Date": "",
                "Qty_Sold": int(bylot["Qty_Sold"].sum() if not bylot.empty else 0),
                "Revenue": bylot["Revenue"].sum() if not bylot.empty else 0.0,
                "COGS": bylot["COGS"].sum() if not bylot.empty else 0.0,
                "Profit": bylot["Profit"].sum() if not bylot.empty else 0.0,
                "Quantity Purchased": int(bylot["Quantity Purchased"].fillna(0).sum() if not bylot.empty else 0),
                "Quantity Left": int(bylot["Quantity Left"].fillna(0).sum() if not bylot.empty else 0),
                "Unit Cost": "",
                "Man A (Net)": bylot["Man A (Net)"].sum() if not bylot.empty else 0.0,
                "Man B": bylot["Man B"].sum() if not bylot.empty else 0.0,
                "Man C (From A)": bylot["Man C (From A)"].sum() if not bylot.empty else 0.0,
            }
            bylot_tot = pd.concat([bylot, pd.DataFrame([totals])], ignore_index=True)

            df_show = bylot_tot.copy()
            for c in ["Revenue","COGS","Profit","Man A (Net)","Man B","Man C (From A)"]:
                df_show[c] = df_show[c].map(money)

            st.dataframe(
                df_show[[
                    "Item","Category","Lot ID","Purchase Date",
                    "Quantity Purchased","Quantity Left","Qty_Sold",
                    "Revenue","COGS","Profit",
                    "Man A (Net)","Man B","Man C (From A)"
                ]],
                use_container_width=True, hide_index=True
            )


# ------------------------------------------------------------------
# Category tabs
# ------------------------------------------------------------------
with tab_crown:
    category_section("Crown")
with tab_wd:
    category_section("WD")

