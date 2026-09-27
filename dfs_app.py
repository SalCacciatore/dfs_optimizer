import pandas as pd
import streamlit as st
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, PULP_CBC_CMD

st.set_page_config(page_title="DFS Lineup Optimizer", page_icon="🏈", layout="wide")

REQUIRED_COLS = ['Player', 'DK Pos', 'DK Salary', 'DK Proj', 'DK Ceiling',
                 'DK Floor', 'Small Field', 'Large Field']


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------
def perc_remover(num):
    # Already numeric (e.g. Excel parsed it) -> pass through
    if isinstance(num, (int, float)):
        return float(num)
    number = str(num).split('%')[0].strip()
    return float(number) if number else 0.0


def dollar_remover(num):
    if isinstance(num, (int, float)):
        return float(num)
    s = str(num)
    number = s.split('$')[1] if '$' in s else s
    number = number.replace(',', '').strip()
    return float(number)


def clean_df(df):
    df = df.copy()
    df['Large Field'] = df['Large Field'].apply(lambda x: float(perc_remover(x)))
    df['Small Field'] = df['Small Field'].apply(lambda x: float(perc_remover(x)))
    df['DK Salary'] = df['DK Salary'].apply(lambda x: float(dollar_remover(x)))
    for col in ['DK Proj', 'DK Ceiling', 'DK Floor']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    df['DK Pos'] = df['DK Pos'].astype(str).str.strip().str.upper()
    df['Player'] = df['Player'].astype(str).str.strip()
    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
def optimize_dfs_lineup(df, num_lineups=1, optimize_for='ceiling',
                        locked_players=None, excluded_players=None,
                        min_unique=1, te_in_flex=True):
    prob = LpProblem("DFS_Lineup_Optimizer", LpMaximize)
    player_vars = {idx: LpVariable(f"player_{idx}", cat='Binary') for idx in df.index}

    obj_col = 'DK Proj' if optimize_for == 'projection' else 'DK Ceiling'
    prob += lpSum([df.loc[idx, obj_col] * player_vars[idx] for idx in df.index])

    for name in (locked_players or []):
        idx = df[df['Player'] == name].index[0]
        prob += player_vars[idx] == 1
    for name in (excluded_players or []):
        idx = df[df['Player'] == name].index[0]
        prob += player_vars[idx] == 0

    prob += lpSum([df.loc[idx, 'DK Salary'] * player_vars[idx] for idx in df.index]) <= 50000
    prob += lpSum([player_vars[idx] for idx in df.index]) == 9

    qb = df[df['DK Pos'] == 'QB'].index
    rb = df[df['DK Pos'] == 'RB'].index
    wr = df[df['DK Pos'] == 'WR'].index
    te = df[df['DK Pos'] == 'TE'].index
    dst = df[df['DK Pos'] == 'DST'].index

    prob += lpSum([player_vars[i] for i in qb]) == 1
    prob += lpSum([player_vars[i] for i in rb]) >= 2
    prob += lpSum([player_vars[i] for i in rb]) <= 3
    prob += lpSum([player_vars[i] for i in wr]) >= 3
    prob += lpSum([player_vars[i] for i in wr]) <= 4
    prob += lpSum([player_vars[i] for i in te]) >= 1
    prob += lpSum([player_vars[i] for i in te]) <= (2 if te_in_flex else 1)
    prob += lpSum([player_vars[i] for i in dst]) == 1
    prob += lpSum([player_vars[i] for i in rb.union(wr).union(te)]) == 7

    lineups = []
    status_msg = None
    for n in range(1, num_lineups + 1):
        prob.solve(PULP_CBC_CMD(msg=0))
        if LpStatus[prob.status] != 'Optimal':
            status_msg = (f"Only {len(lineups)} lineup(s) possible — no feasible lineup #{n} "
                          f"(status: {LpStatus[prob.status]}). Loosen locks/exclusions or Min unique.")
            break
        selected = [i for i in df.index if player_vars[i].varValue > 0.5]
        lu = df.loc[selected].copy()
        lu['Lineup'] = n
        lineups.append(lu)
        prob += (lpSum([player_vars[i] for i in selected]) <= 9 - min_unique,
                 f"unique_vs_lineup_{n}")
    return lineups, status_msg


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🏈 DFS Lineup Optimizer")
st.caption("DraftKings NFL classic: QB, 2 RB, 3 WR, TE, FLEX, DST — $50,000 cap")

uploaded = st.file_uploader("Upload projection set (CSV)", type=['csv'])

if uploaded is None:
    st.info(f"Required columns: {', '.join(REQUIRED_COLS)}")
    st.stop()

try:
    raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Couldn't read file: {e}")
    st.stop()

missing = [c for c in REQUIRED_COLS if c not in raw.columns]
if missing:
    st.error(f"Missing columns: {', '.join(missing)}")
    st.stop()

try:
    df = clean_df(raw)
except Exception as e:
    st.error(f"Cleaning failed (check % and $ formatting): {e}")
    st.stop()

with st.expander(f"Player pool ({len(df)} players)", expanded=False):
    st.dataframe(df[REQUIRED_COLS], use_container_width=True, hide_index=True)

# Sidebar settings
st.sidebar.header("Settings")
num_lineups = st.sidebar.number_input("Number of lineups", min_value=1, max_value=150, value=20, step=1)
optimize_for = st.sidebar.radio("Optimize for", ['ceiling', 'projection'],
                                format_func=str.capitalize)
min_unique = st.sidebar.number_input("Min unique players vs. every other lineup",
                                     min_value=1, max_value=9, value=1, step=1)
te_in_flex = st.sidebar.toggle("Allow TE in FLEX", value=True)

players = sorted(df['Player'].unique())
locked = st.sidebar.multiselect("Lock players (in every lineup)", players)
excluded = st.sidebar.multiselect("Exclude players", [p for p in players if p not in locked])

if st.sidebar.button("Build lineups", type="primary", use_container_width=True):
    with st.spinner("Optimizing..."):
        lineups, status_msg = optimize_dfs_lineup(
            df, num_lineups=int(num_lineups), optimize_for=optimize_for,
            locked_players=locked, excluded_players=excluded,
            min_unique=int(min_unique), te_in_flex=te_in_flex,
        )
    st.session_state['lineups'] = lineups
    st.session_state['status_msg'] = status_msg

lineups = st.session_state.get('lineups')
if not lineups:
    st.stop()

if st.session_state.get('status_msg'):
    st.warning(st.session_state['status_msg'])

all_lu = pd.concat(lineups)

# Summary table
summary = all_lu.groupby('Lineup').agg(
    Salary=('DK Salary', 'sum'),
    Proj=('DK Proj', 'sum'),
    Ceiling=('DK Ceiling', 'sum'),
    Floor=('DK Floor', 'sum'),
    Own_Small=('Small Field', 'sum'),
    Own_Large=('Large Field', 'sum'),
).round(2).reset_index()
summary.columns = ['Lineup', 'Salary', 'Proj', 'Ceiling', 'Floor',
                   'Total Own (Small %)', 'Total Own (Large %)']

st.subheader(f"{len(lineups)} lineups")
st.dataframe(summary, use_container_width=True, hide_index=True)

# Exposure
exposure = (all_lu.groupby(['Player', 'DK Pos']).size()
            .reset_index(name='Lineups'))
exposure['Exposure %'] = (exposure['Lineups'] / len(lineups) * 100).round(1)
exposure = exposure.sort_values('Lineups', ascending=False)

col1, col2 = st.columns([2, 1])
with col2:
    st.subheader("Exposure")
    st.dataframe(exposure, use_container_width=True, hide_index=True, height=500)

with col1:
    st.subheader("Lineup detail")
    pos_order = {'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3, 'DST': 4}
    for lu in lineups:
        n = lu['Lineup'].iloc[0]
        lu = lu.assign(_o=lu['DK Pos'].map(pos_order)).sort_values(['_o', 'DK Salary'],
                                                                  ascending=[True, False])
        label = (f"Lineup {n} — ${lu['DK Salary'].sum():,.0f} | "
                 f"Proj {lu['DK Proj'].sum():.2f} | Ceil {lu['DK Ceiling'].sum():.2f}")
        with st.expander(label, expanded=(n == 1)):
            st.dataframe(lu[['Player', 'DK Pos', 'DK Salary', 'DK Proj', 'DK Ceiling',
                             'DK Floor', 'Small Field', 'Large Field']],
                         use_container_width=True, hide_index=True)

st.download_button(
    "Download all lineups (CSV)",
    all_lu[['Lineup', 'Player', 'DK Pos', 'DK Salary', 'DK Proj', 'DK Ceiling',
            'DK Floor', 'Small Field', 'Large Field']].to_csv(index=False),
    file_name="lineups.csv",
    mime="text/csv",
)
