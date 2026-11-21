from datetime import datetime
import io
import json
import re
from google import genai
from google.oauth2.service_account import Credentials
import gspread
import pandas as pd
import pdfplumber
import streamlit as st

st.set_page_config(
    page_title="True Net Worth & Retirement Planner",
    page_icon="📈",
    layout="wide",
)

# ==========================================
# 1. GOOGLE SSO AUTHENTICATION GATE
# ==========================================
if not st.user.is_logged_in:
    st.title("🛡️ True Net Worth & Retirement Planner")
    st.markdown(
        "Sign in with your Google account to consolidate your CAS & brokerage statements, "
        "estimate portfolio returns with Gemini AI, and save your progress privately to Google Sheets."
    )
    if st.button("🔐 Log in with Google", type="primary"):
        st.login("google")
    st.stop()

# Sidebar User Header & Logout
st.sidebar.markdown(f"👤 **{st.user.name}** (`{st.user.email}`)")
if st.sidebar.button("Log out"):
    st.logout()
    st.stop()
st.sidebar.markdown("---")


# ==========================================
# 2. HELPER FUNCTIONS (PARSING, GEMINI, SHEETS)
# ==========================================
def clean_num(val_str):
    if (
        not val_str
        or str(val_str).strip() == ""
        or str(val_str).lower() == "nan"
    ):
        return 0.0
    cleaned = re.sub(r"[^\d.-]", "", str(val_str))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def parse_cas_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    full_text = re.sub(r"(\d),\s+(\d)", r"\1,\2", full_text)
    full_text = re.sub(r"(\d)\.\s+(\d)", r"\1.\2", full_text)
    blocks = re.split(r"(?=Folio No:\s*[\d/]+)", full_text)
    holdings = []

    for block in blocks:
        if "Closing Unit Balance:" not in block or "ISIN:" not in block:
            continue
        units_m = re.search(r"Closing Unit Balance:\s*([\d,]+\.\d+)", block)
        mkt_m = re.search(
            r"Market Value on\s*[\d]{2}-[A-Za-z]{3}-\d{4}:\s*INR\s*([\d,]+\.\d+)",
            block,
        )
        units = clean_num(units_m.group(1)) if units_m else 0.0
        market_val = clean_num(mkt_m.group(1)) if mkt_m else 0.0
        if units <= 0 and market_val <= 0:
            continue

        scheme_m = re.search(
            r"^[A-Z0-9]+-\s*(.*?)\s*\((?:Non-Demat|Demat)\)\s*-\s*ISIN:\s*([A-Z0-9]+)\s*\(\s*Advisor:\s*(.*?)\s*\)",
            block,
            re.MULTILINE | re.DOTALL,
        )
        nav_m = re.search(
            r"NAV on\s*[\d]{2}-[A-Za-z]{3}-\d{4}:\s*INR\s*([\d,]+\.\d+)", block
        )
        cost_m = re.search(r"Total Cost Value:\s*([\d,]+\.\d+)", block)

        scrip = (
            re.sub(r"\s+", " ", scheme_m.group(1)).strip()
            if scheme_m
            else "Unknown MF"
        )
        nav = clean_num(nav_m.group(1)) if nav_m else 0.0
        cost_val = clean_num(cost_m.group(1)) if cost_m else 0.0
        asset_class = (
            "Debt"
            if any(
                k in scrip.lower()
                for k in ["liquid", "debt", "duration", "arbitrage", "overnight"]
            )
            else "Equity"
        )

        holdings.append({
            "Source": "Mutual Fund (CAS)",
            "Scrip": scrip,
            "Currency": "INR",
            "Units": units,
            "Price (INR)": nav,
            "Cost Value (INR)": cost_val,
            "Market Value (INR)": market_val,
            "Asset Class": asset_class,
        })

    return pd.DataFrame(holdings)


def parse_generic_broker_file(uploaded_file, currency="INR", fx_rate=1.0):
    try:
        content = uploaded_file.getvalue()
        if uploaded_file.name.endswith(".csv"):
            text_content = content.decode("utf-8", errors="ignore")
            lines = text_content.splitlines()
            header_idx = 0
            for idx, line in enumerate(lines):
                line_upper = line.upper()
                if (
                    "SECURITY" in line_upper
                    or "SYMBOL" in line_upper
                    or (
                        "QUANTITY" in line_upper
                        and "MARKET VALUE" in line_upper
                    )
                ):
                    header_idx = idx
                    break
            df = pd.read_csv(
                io.StringIO("\n".join(lines[header_idx:])),
                engine="python",
                on_bad_lines="skip",
            )
        else:
            df = pd.read_excel(io.BytesIO(content))

        df.columns = [str(c).strip() for c in df.columns]
        parsed = pd.DataFrame()
        parsed["Source"] = f"Broker ({uploaded_file.name})"

        scrip_col = next(
            (
                c
                for c in [
                    "SYMBOL",
                    "Scrip",
                    "Symbol",
                    "Security",
                    "Description",
                ]
                if c in df.columns
            ),
            None,
        )
        parsed["Scrip"] = (
            df[scrip_col].astype(str) if scrip_col else "Unknown Asset"
        )
        parsed["Currency"] = df.get("Currency", currency)

        def get_col_values(possible_names, default=0.0):
            col = next((c for c in possible_names if c in df.columns), None)
            if col is not None:
                return df[col].apply(lambda x: clean_num(str(x)))
            return default

        units = get_col_values(["QTY", "Quantity", "Long Term Qty", "Units"], 1.0)
        price = get_col_values(["LTP", "Price", "Market Price", "NAV"], 0.0)
        cost = get_col_values(
            [
                "INVESTMENT VALUE",
                "Total Cost",
                "Cost Value",
                "Cost",
                "Unit Cost",
            ],
            0.0,
        )

        parsed["Units"] = units
        parsed["Price (INR)"] = price * fx_rate
        parsed["Cost Value (INR)"] = cost * fx_rate

        mkt_col = next(
            (c for c in ["CURRENT VALUE", "Market Value"] if c in df.columns),
            None,
        )
        if mkt_col is not None:
            parsed["Market Value (INR)"] = (
                df[mkt_col].apply(lambda x: clean_num(str(x))) * fx_rate
            )
        else:
            parsed["Market Value (INR)"] = parsed["Units"] * parsed["Price (INR)"]

        ignored_labels = [
            "Equity",
            "Shares",
            "Shares - Total",
            "Equity - Total",
            "Cash and Equivalent",
            "Cash / Bank",
            "Cash / Bank - Total",
            "Other Assets",
            "Security",
        ]
        parsed = parsed[
            parsed["Scrip"].notna()
            & (parsed["Scrip"].str.strip() != "")
            & (~parsed["Scrip"].str.strip().isin(ignored_labels))
            & (~parsed["Scrip"].str.contains("Total", case=False, na=False))
            & (parsed["Market Value (INR)"] > 0)
        ]

        parsed["Asset Class"] = parsed["Scrip"].apply(
            lambda x: (
                "Debt"
                if any(
                    k in str(x).lower()
                    for k in ["bond", "debt", "fd", "cash", "liquid"]
                )
                else "Equity"
            )
        )
        return parsed
    except Exception as e:
        st.error(f"Error parsing {uploaded_file.name}: {e}")
        return pd.DataFrame()


def estimate_portfolio_return_with_gemini(df_holdings):
    """Uses Gemini AI to inspect holdings and suggest a conservative annual CAGR."""
    if "GEMINI_API_KEY" not in st.secrets:
        return (
            9.5,
            "Default fallback used (Add GEMINI_API_KEY to secrets.toml for live AI analysis).",
        )

    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    summary_records = df_holdings[
        ["Scrip", "Currency", "Asset Class", "Market Value (INR)"]
    ].to_dict(orient="records")

    prompt = f"""
    Analyze this investment portfolio and estimate a realistic, CONSERVATIVE expected annualized rate of return (CAGR %) over the next 7-10 years.
    For example, a US growth fund like VPMAX may have ~14% historical peaks, so a conservative forward estimate would discount that to ~10-11%. Indian equities typically warrant ~10-12% conservative estimates, and Indian debt/liquid funds ~6-7%.
    
    Portfolio Holdings:
    {json.dumps(summary_records, indent=2)}
    
    Return ONLY valid JSON with two keys:
    - "conservative_cagr_pct": float (e.g., 10.2)
    - "rationale": string (2 concise sentences explaining the weighted return expectation)
    """
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        data = json.loads(response.text)
        return float(data.get("conservative_cagr_pct", 9.5)), data.get(
            "rationale", ""
        )
    except Exception as e:
        return 9.5, f"AI estimation fallback ({e})"


def save_to_google_sheet(user_email, df_holdings, profile_dict):
  """Saves consolidated holdings and retirement assumptions to a Google Sheet in the user's Drive."""
  creds = get_user_drive_credentials()
  if not creds:
    return None

  gc = gspread.authorize(creds)

  sheet_title = f"TrueNetWorth_{user_email}"
  try:
    sh = gc.open(sheet_title)
  except gspread.SpreadsheetNotFound:
    sh = gc.create(sheet_title)

  # Tab 1: Holdings
  ws_holdings = sh.sheet1
  ws_holdings.update_title("Holdings")
  ws_holdings.clear()
  ws_holdings.update(
      [df_holdings.columns.values.tolist()] + df_holdings.values.tolist()
  )

  # Tab 2: Assumptions & Summary
  try:
    ws_summary = sh.worksheet("Plan_Summary")
  except gspread.WorksheetNotFound:
    ws_summary = sh.add_worksheet(title="Plan_Summary", rows=30, cols=5)
  ws_summary.clear()
  summary_rows = [["Metric", "Value"]] + [
      [k, str(v)] for k, v in profile_dict.items()
  ]
  ws_summary.update(summary_rows)

  return sh.url


def load_from_google_sheet(user_email):
  """Loads previously saved holdings from the user's Google Sheet."""
  creds = get_user_drive_credentials()
  if not creds:
    return pd.DataFrame()

  gc = gspread.authorize(creds)
  try:
    sh = gc.open(f"TrueNetWorth_{user_email}")
    ws = sh.worksheet("Holdings")
    records = ws.get_all_records()
    return pd.DataFrame(records)
  except Exception:
    return pd.DataFrame()

import os
from google_auth_oauthlib.flow import Flow

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]

def get_user_drive_credentials(redirect_uri=None):
  """Returns authorized Google Credentials for the current user, or renders a connect link."""
  if redirect_uri is None:
    auth_cfg = st.secrets.get("auth", {})
    configured_uri = auth_cfg.get("redirect_uri")
    if configured_uri:
      redirect_uri = configured_uri.replace("/oauth2callback", "")
    else:
      redirect_uri = "https://wealthmanager.streamlit.app"

  if "gdrive_creds" in st.session_state:
    creds = st.session_state["gdrive_creds"]
    if getattr(creds, "expired", False) and getattr(
        creds, "refresh_token", None
    ):
      try:
        from google.auth.transport.requests import Request

        creds.refresh(Request())
        st.session_state["gdrive_creds"] = creds
      except Exception:
        del st.session_state["gdrive_creds"]
    else:
      return creds

  auth_section = st.secrets["auth"]
  google_cfg = auth_section.get("google", auth_section)

  client_config = {
      "web": {
          "client_id": google_cfg["client_id"],
          "client_secret": google_cfg["client_secret"],
          "auth_uri": "https://accounts.google.com/o/oauth2/auth",
          "token_uri": "https://oauth2.googleapis.com/token",
      }
  }

  flow = Flow.from_client_config(
      client_config, scopes=SCOPES, redirect_uri=redirect_uri
  )

  auth_code = st.query_params.get("code")
  if auth_code:
    os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
    try:
      flow.fetch_token(code=auth_code)
      st.session_state["gdrive_creds"] = flow.credentials
      st.query_params.clear()
      st.rerun()
    except Exception as e:
      st.query_params.clear()
      st.error(f"Could not complete Google Drive authorization: {e}")
      return None

  auth_url, _ = flow.authorization_url(
      access_type="offline", include_granted_scopes="true", prompt="consent"
  )
  st.link_button(
      "🔗 Connect Google Drive & Sheets (Your Account)", auth_url, type="primary"
  )
  return None


def deduplicate_holdings(df: pd.DataFrame) -> pd.DataFrame:
  if df.empty:
    return df
  df = df.copy()
  df["_scrip_norm"] = df["Scrip"].astype(str).str.strip().str.lower()
  df["_curr_norm"] = df["Currency"].astype(str).str.strip().str.upper()
  df["_units_norm"] = (
      pd.to_numeric(df["Units"], errors="coerce").fillna(0).round(4)
  )
  return (
      df.drop_duplicates(
          subset=["_scrip_norm", "_curr_norm", "Source"],
          keep="last",
      )
      .drop(columns=["_scrip_norm", "_curr_norm", "_units_norm"])
      .reset_index(drop=True)
  )

# ==========================================
# 3. SIDEBAR: DATA INGESTION & DEMOGRAPHICS
# ==========================================
st.title("🛡️ True Net Worth & Retirement Trajectory Simulator")

if st.sidebar.button("📥 Load Last Saved Portfolio from Google Sheets"):
    saved_df = load_from_google_sheet(st.user.email)
    if not saved_df.empty:
        st.session_state["loaded_sheet_df"] = saved_df
        st.sidebar.success("Loaded saved portfolio from Google Sheets!")
    else:
        st.sidebar.info("No saved Google Sheet found yet for your account.")

st.sidebar.header("1. Upload CAS PDF")
cas_file = st.sidebar.file_uploader(
    "Upload CAMS/KFintech CAS PDF", type=["pdf"]
)

st.sidebar.header("2. Brokerage Files (Multiple Allowed)")
broker_files = st.sidebar.file_uploader(
    "Upload CSV or Excel Reports",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
)
broker_curr = st.sidebar.selectbox(
    "Default File Currency", ["INR", "USD", "EUR", "GBP"]
)
broker_fx = st.sidebar.number_input(
    "FX Rate to INR", value=86.5 if broker_curr != "INR" else 1.0, step=0.1
)

st.sidebar.header("3. Manual Asset Entry (Optional)")
manual_name = st.sidebar.text_input("Asset Name (e.g., VPMAX / Real Estate)", "")
manual_val = st.sidebar.number_input("Market Value (INR)", value=0.0, step=10000.0)
manual_cost = st.sidebar.number_input("Cost Value (INR)", value=0.0, step=10000.0)
manual_class = st.sidebar.selectbox("Asset Class", ["Equity", "Debt"])

st.sidebar.header("4. Personal & Income Profile")
dob = st.sidebar.date_input("Date of Birth", value=datetime(1969, 1, 1))
today = datetime.today().date()
age = (
    today.year
    - dob.year
    - ((today.month, today.day) < (dob.month, dob.day))
)

employment_status = st.sidebar.selectbox(
    "Employment Status", ["Salaried", "Unemployed", "Retired"]
)

monthly_salary = 0.0
annual_savings_addition = 0.0
target_retire_age = age

if employment_status == "Salaried":
    monthly_salary = st.sidebar.number_input(
        "Monthly Salary (INR)", value=150000.0, step=10000.0
    )
    annual_savings_addition = st.sidebar.number_input(
        "Annual New Investment Contribution (INR)",
        value=600000.0,
        step=50000.0,
        help="How much new money you invest per year until retirement.",
    )
    target_retire_age = st.sidebar.number_input(
        "Target Retirement Age (Income Start Age)",
        min_value=age,
        max_value=85,
        value=max(60, age),
        step=1,
    )
elif employment_status == "Unemployed":
    target_retire_age = st.sidebar.number_input(
        "Age to Start Drawing Income",
        min_value=age,
        max_value=85,
        value=age,
        step=1,
    )
else:
    target_retire_age = age

# ==========================================
# 4. DATA CONSOLIDATION
# ==========================================
all_dfs = []

if "loaded_sheet_df" in st.session_state:
  sheet_df = st.session_state["loaded_sheet_df"].copy()
  for col in ["Units", "Price (INR)", "Cost Value (INR)", "Market Value (INR)"]:
    if col in sheet_df.columns:
      sheet_df[col] = sheet_df[col].apply(lambda x: clean_num(str(x)))
  all_dfs.append(sheet_df)

if cas_file:
  df_cas = parse_cas_pdf(cas_file)
  if not df_cas.empty:
    all_dfs.append(df_cas)

if broker_files:
  for b_file in broker_files:
    df_b = parse_generic_broker_file(
        b_file, currency=broker_curr, fx_rate=broker_fx
    )
    if not df_b.empty:
      all_dfs.append(df_b)

if manual_val > 0:
  all_dfs.append(
      pd.DataFrame([{
          "Source": "Manual Entry",
          "Scrip": manual_name if manual_name else "Manual Asset",
          "Currency": "INR",
          "Units": 1.0,
          "Price (INR)": manual_val,
          "Cost Value (INR)": manual_cost if manual_cost > 0 else manual_val,
          "Market Value (INR)": manual_val,
          "Asset Class": manual_class,
      }])
  )

if all_dfs:
  raw_df = pd.concat(all_dfs, ignore_index=True)
  master_df = deduplicate_holdings(raw_df)
  dupes = len(raw_df) - len(master_df)
  if dupes > 0:
    st.caption(f"ℹ️ Skipped {dupes} duplicate holding entry/entries.")
else:
  master_df = pd.DataFrame()

if master_df.empty:
    st.info(
        "👈 Upload your CAS PDF, brokerage files, or load your saved Google Sheet from the sidebar to begin."
    )
    st.stop()

# ==========================================
# 5. PORTFOLIO DISPLAY & METRICS
# ==========================================
st.subheader("📋 Consolidated Portfolio Holdings")
display_df = master_df.copy()
display_df["Gain/Loss (INR)"] = (
    display_df["Market Value (INR)"] - display_df["Cost Value (INR)"]
)

st.dataframe(
    display_df.style.format({
        "Units": "{:,.2f}",
        "Price (INR)": "{:,.2f}",
        "Cost Value (INR)": "{:,.2f}",
        "Market Value (INR)": "{:,.2f}",
        "Gain/Loss (INR)": "{:,.2f}",
    }),
    width="stretch",
)

total_market_value = master_df["Market Value (INR)"].sum()
total_cost_value = master_df["Cost Value (INR)"].sum()
total_gain = total_market_value - total_cost_value

equity_value = master_df[master_df["Asset Class"] == "Equity"][
    "Market Value (INR)"
].sum()
debt_value = master_df[master_df["Asset Class"] == "Debt"][
    "Market Value (INR)"
].sum()
equity_pct = (
    (equity_value / total_market_value * 100) if total_market_value > 0 else 0.0
)
debt_pct = (
    (debt_value / total_market_value * 100) if total_market_value > 0 else 0.0
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Net Worth", f"₹ {total_market_value:,.0f}")
col2.metric("Total Invested Cost", f"₹ {total_cost_value:,.0f}")
col3.metric(
    "Unrealized Gain",
    f"₹ {total_gain:,.0f}",
    delta=(
        f"{(total_gain / total_cost_value * 100):.1f}%"
        if total_cost_value > 0
        else "0%"
    ),
)
col4.metric("Asset Split", f"Eq: {equity_pct:.1f}% | Debt: {debt_pct:.1f}%")

st.markdown("---")

# ==========================================
# 6. GEMINI RETURN PREDICTION & EDITABLE SIMULATION CONTROLS
# ==========================================
st.subheader("🎛️ Interactive Return & Withdrawal Simulator")

if "ai_cagr" not in st.session_state:
    st.session_state["ai_cagr"] = 10.0
    st.session_state["ai_rationale"] = (
        "Click 'Estimate Conservative Return with Gemini' to analyze your specific holdings."
    )

ai_col1, ai_col2 = st.columns([1, 3])
with ai_col1:
    if st.button("✨ Estimate Return with Gemini AI"):
        with st.spinner("Analyzing holdings..."):
            est_cagr, rationale = estimate_portfolio_return_with_gemini(
                master_df
            )
            st.session_state["ai_cagr"] = est_cagr
            st.session_state["ai_rationale"] = rationale
with ai_col2:
    st.info(f"**Gemini Portfolio Assessment:** {st.session_state['ai_rationale']}")

# Editable Simulation Parameters (Changing any of these automatically updates the chart below)
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)

with ctrl1:
    expected_return_pct = st.number_input(
        "Expected Annual Return (%)",
        min_value=0.0,
        max_value=30.0,
        value=float(st.session_state["ai_cagr"]),
        step=0.5,
        help="Pre-filled by Gemini AI; edit freely to test different market scenarios.",
    )

with ctrl2:
    default_swr_income = round(total_market_value * 0.04, -3)
    desired_annual_income = st.number_input(
        "Annual Retirement Income / Withdrawal (INR)",
        min_value=0.0,
        value=float(default_swr_income),
        step=50000.0,
        help="Defaults to 4% Safe Withdrawal Rate of current corpus. Increase or decrease to see corpus longevity.",
    )

with ctrl3:
    inflation_pct = st.number_input(
        "Annual Inflation Adjustment (%)",
        min_value=0.0,
        max_value=15.0,
        value=5.0,
        step=0.5,
        help="Increases annual withdrawals each year to preserve purchasing power.",
    )

with ctrl4:
    projection_end_age = st.number_input(
        "Project Until Age",
        min_value=max(age + 5, int(target_retire_age) + 5),
        max_value=105,
        value=90,
        step=1,
    )

# ==========================================
# 7. CORPUS PROJECTION ENGINE & INTERACTIVE CHART
# ==========================================
r_rate = expected_return_pct / 100.0
inf_rate = inflation_pct / 100.0

projection_rows = []
corpus = total_market_value
current_year = today.year
depletion_age = None
depletion_year = None

for sim_age in range(age, int(projection_end_age) + 1):
    yr_offset = sim_age - age
    sim_year = current_year + yr_offset

    if sim_age < target_retire_age:
        phase = "Accumulation (Working)"
        contribution = annual_savings_addition
        withdrawal = 0.0
    else:
        phase = "Retirement (Withdrawal)"
        contribution = 0.0
        # Adjust desired annual income for inflation from today
        withdrawal = desired_annual_income * ((1 + inf_rate) ** yr_offset)

    # Calculate end-of-year corpus
    investment_growth = corpus * r_rate
    end_corpus = corpus + investment_growth + contribution - withdrawal

    if end_corpus <= 0 and depletion_age is None:
        depletion_age = sim_age
        depletion_year = sim_year
        end_corpus = 0.0

    projection_rows.append({
        "Age": sim_age,
        "Year": sim_year,
        "Phase": phase,
        "Starting Corpus (INR)": round(corpus, 2),
        "Annual Withdrawal (INR)": round(withdrawal, 2),
        "Ending Corpus (INR)": round(max(0.0, end_corpus), 2),
    })

    corpus = max(0.0, end_corpus)

proj_df = pd.DataFrame(projection_rows)

# Key Longevity & Retirement Readiness Callouts
st.markdown("### 📊 Corpus Trajectory & Longevity Outlook")

m1, m2, m3 = st.columns(3)
corpus_at_retirement = proj_df.loc[
    proj_df["Age"] == target_retire_age, "Starting Corpus (INR)"
].values[0]
final_corpus = proj_df["Ending Corpus (INR)"].iloc[-1]

m1.metric(
    f"Projected Corpus at Age {int(target_retire_age)} ({current_year + (int(target_retire_age) - age)})",
    f"₹ {corpus_at_retirement:,.0f}",
)
m2.metric(
    f"Projected Corpus at Age {int(projection_end_age)}",
    f"₹ {final_corpus:,.0f}",
)

if depletion_age is not None:
    years_lasted = depletion_age - target_retire_age
    m3.error(
        f"⚠️ **Corpus Depletes at Age {depletion_age} ({depletion_year})** — lasts {max(0, years_lasted)} years into retirement."
    )
else:
    m3.success(
        f"✅ **Corpus Lasts Beyond Age {int(projection_end_age)}!** Sustainable at ₹ {desired_annual_income:,.0f}/yr."
    )

# Interactive Line/Area Chart (Updates automatically whenever any input above changes)
chart_data = proj_df.set_index("Year")[
    ["Ending Corpus (INR)", "Annual Withdrawal (INR)"]
]
st.line_chart(chart_data, height=400)

with st.expander("View Year-by-Year Corpus Projection Table"):
    st.dataframe(
        proj_df.style.format({
            "Starting Corpus (INR)": "{:,.0f}",
            "Annual Withdrawal (INR)": "{:,.0f}",
            "Ending Corpus (INR)": "{:,.0f}",
        }),
        width="stretch",
    )

# ==========================================
# 8. SAVE TO GOOGLE SHEETS
# ==========================================
st.markdown("---")
if st.button("💾 Save Consolidated Portfolio & Plan to Google Sheets", type="primary"):
    profile_summary = {
        "User Email": st.user.email,
        "Last Updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Age": age,
        "Employment Status": employment_status,
        "Target Retirement Age": target_retire_age,
        "Total Net Worth (INR)": round(total_market_value, 2),
        "Equity Allocation (%)": round(equity_pct, 1),
        "Debt Allocation (%)": round(debt_pct, 1),
        "Expected Annual Return (%)": expected_return_pct,
        "Annual Retirement Income (INR)": desired_annual_income,
        "Projected Corpus at Retirement (INR)": corpus_at_retirement,
        "Depletion Age": depletion_age if depletion_age else f">{projection_end_age}",
    }
    with st.spinner("Syncing to Google Sheets..."):
        sheet_url = save_to_google_sheet(
            st.user.email, master_df, profile_summary
        )
        if sheet_url:
            st.success(
                f"Saved successfully! [Open your Google Sheet]({sheet_url})"
            )

