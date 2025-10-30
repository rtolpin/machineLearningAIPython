import pandas as pd
import numpy as np
import re
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

# Scalers
from sklearn.preprocessing import RobustScaler, MinMaxScaler

# Deep learning (Keras)
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout

# Optional: if you want to suppress TensorFlow info messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

RE_RESTAURANT = re.compile(r"\b(restaurant|ristorante|trattoria|bistro|brasserie|steakhouse|sushi|izakaya|tapas|taqueria|pizzeria|diner|cafe|caf[eé]|bar|lounge|wine|brew|pub)\b", re.I)
RE_DELIVERY = re.compile(r"\b(doordash|uber\s*eats|ubereats|grubhub|seamless|postmates|deliveroo|caviar|slice)\b", re.I)
RE_COFFEE = re.compile(r"\b(starbucks|blue bottle|gregory'?s|stumptown|joe coffee|pret|blank street|coffee|latte|cappuccino|espresso)\b", re.I)

social_categories = [
    'Category_TRAVEL',
    'Category_PERSONAL',
    'Category_GIFTS_AND_DONATIONS',
    'Category_ENTERTAINMENT'
]

def safe_str(x): 
    return x if isinstance(x, str) else ""

def score_food_social(row):
    if row.get('Category_FOOD_AND_DRINK', 0) != 1:
        return 0.0

    desc = safe_str(row.get('Description', ''))
    d = row.get('Transaction Date')

    score = 0.0
        
    if RE_RESTAURANT.search(desc):
        score += 0.6

    if RE_DELIVERY.search(desc):
        score -= 0.5

    amt = row.get('Amount', 0) or 0
    if amt >= 0 and amt < 25:
        score -= 0.05

    score = np.clip(score, 0.0, 1.0)

    return score

def compute_scores_social(df):
    tmp = df.copy()
    date_col = 'Transaction Date'
    
    tmp['Initial_SocialScore'] = tmp.apply(score_food_social, axis=1)

    tmp['FOOD_AND_DRINK_Social_Score'] = np.where(
        tmp['Initial_SocialScore'] >= 0.5,
        tmp['Amount'],
        0.0
    )

    tmp['FOOD_AND_DRINK_Solo_Score'] = np.where(
        tmp['Initial_SocialScore'] < 0.5,
        tmp['Amount'],
        0.0
    )
    
    tmp = tmp.groupby(tmp[date_col].dt.date).agg(
        Total=('Amount', 'sum'),
        FOOD_AND_DRINK_Social_Score=('FOOD_AND_DRINK_Social_Score', 'sum'),
        FOOD_AND_DRINK_Solo_Score=('FOOD_AND_DRINK_Solo_Score', 'sum'),
    ).reset_index().rename(columns={date_col: 'Date'})
    tmp['Date'] = pd.to_datetime(tmp['Date'])

    eps = 1e-6
    tmp['FOOD_AND_DRINK_Social_Share'] = tmp['FOOD_AND_DRINK_Social_Score'] / (tmp['Total'] + eps)
    tmp['Share_FOOD_AND_DRINK_Solo_Share']   = tmp['FOOD_AND_DRINK_Solo_Score'] / (tmp['Total'] + eps)

    return tmp
    
def add_next_day_calendar(daily: pd.DataFrame) -> pd.DataFrame:
    nxt = daily[['DayOfWeek', 'Month', 'IsWeekend', 'DOW_sin', 'DOW_cos', 'MOY_sin', 'MOY_cos']].shift(-1)
    nxt.columns = [f'Next_{c}' for c in nxt.columns]
    out = daily.join(nxt)
    return out
   
def daily_aggregate(df, social_score_df):
    date_col = 'Transaction Date'
    cat_cols = [c for c in df.columns if c.startswith('Category_')]
    per_cat_amounts = {c.replace('Category_', 'Amt_'): df['Amount'] * df[c].astype(int) for c in cat_cols}
    category_amounts_df = pd.DataFrame(per_cat_amounts)

    tmp_df = pd.concat([df[[date_col, 'Amount']], category_amounts_df], axis=1)

    daily = (
        tmp_df.groupby(tmp_df[date_col].dt.date)
           .sum(numeric_only=True)
           .rename_axis('Date')
           .reset_index()
    )
    
    daily['Date'] = pd.to_datetime(daily['Date'])

    daily = pd.merge(daily, social_score_df, on='Date', how='left')
    
    full = pd.DataFrame({'Date': pd.date_range(daily['Date'].min(), daily['Date'].max(), freq='D')})
    daily = full.merge(daily, on='Date', how='left').fillna(0.0)
    
    amt_cols = [c for c in daily.columns if c.startswith('Amt_')]
    if amt_cols:
        daily['Total'] = daily[amt_cols].sum(axis=1)
    elif 'Amount' in daily.columns:
        daily['Total'] = daily['Amount']
    else:
        raise ValueError("No amount columns found after aggregation.")

    daily['DayOfWeek'] = daily['Date'].dt.dayofweek  # 0=Mon
    daily['IsWeekend'] = (daily['DayOfWeek'] >= 5).astype(int)
    daily['Month'] = daily['Date'].dt.month

    # cyclic encodings
    daily['DOW_sin'] = np.sin(2 * np.pi * daily['DayOfWeek'] / 7)
    daily['DOW_cos'] = np.cos(2 * np.pi * daily['DayOfWeek'] / 7)
    daily['MOY_sin'] = np.sin(2 * np.pi * (daily['Month'] - 1) / 12)
    daily['MOY_cos'] = np.cos(2 * np.pi * (daily['Month'] - 1) / 12)

    week_length = 7
    biweekly_length = 14
    monthly_length = 30

    for w in (week_length, biweekly_length, monthly_length):
        daily[f'Roll{w}_Total_Mean'] = daily['Total'].rolling(w, min_periods=1).mean()
        daily[f'Roll{w}_Total_Std']  = daily['Total'].rolling(w, min_periods=1).std(ddof=0).fillna(0.0)

    eps = 1e-6
    for amt in amt_cols:
        share_col = amt.replace('Amt_', 'Share_')
        daily[share_col] = daily[amt] / (daily['Total'] + eps)

    def safe_col(name):
        return name if name in daily.columns else None

    C2A = lambda cat: f'Amt_{cat}'
    social_comps = list(filter(None, map(safe_col, map(C2A, [
        'FOOD_AND_DRINK', 'TRAVEL', 'PERSONAL', 'GIFTS_AND_DONATIONS'
    ]))))
    
    wellness_comps = list(filter(None, map(safe_col, map(C2A, [
        'HEALTH_AND_WELLNESS', 'PERSONAL'
    ]))))
    
    essentials_comps = list(filter(None, map(safe_col, map(C2A, [
        'GROCERIES', 'GAS', 'HOME', 'BILLS_AND_UTILITIES', 'FOOD_AND_DRINK', 'TRAVEL'
    ]))))
    
    entertainment_comps = list(filter(None, map(safe_col, map(C2A, [
        'ENTERTAINMENT', 'SHOPPING', 'FOOD_AND_DRINK'
    ]))))

    travel_comps = list(filter(None, map(safe_col, map(C2A, [
        'TRAVEL'
    ]))))

    shopping_comps = list(filter(None, map(safe_col, map(C2A, [
        'SHOPPING'
    ]))))

    daily['Social_Spend']   = daily[social_comps].sum(axis=1) if social_comps else 0.0
    daily['Wellness_Spend'] = daily[wellness_comps].sum(axis=1) if wellness_comps else 0.0
    daily['Essentials_Spend'] = daily[essentials_comps].sum(axis=1) if essentials_comps else 0.0
    daily['Entertainment_Spend'] = daily[entertainment_comps].sum(axis=1) if entertainment_comps else 0.0
    daily['Travel_Spend'] = daily[travel_comps].sum(axis=1) if travel_comps else 0.0
    daily['Shopping_Spend'] = daily[shopping_comps].sum(axis=1) if shopping_comps else 0.0

    daily['Social_Share']   = daily['Social_Spend'] / (daily['Total'] + eps)
    daily['Wellness_Share'] = daily['Wellness_Spend'] / (daily['Total'] + eps)
    daily['Essentials_Share'] = daily['Essentials_Spend'] / (daily['Total'] + eps)
    daily['Entertainment_Share'] = daily['Entertainment_Spend'] / (daily['Total'] + eps)
    daily['Travel_Share'] = daily['Travel_Spend'] / (daily['Total'] + eps)
    daily['Shopping_Share'] = daily['Shopping_Spend'] / (daily['Total'] + eps)

    ratio = daily['Total'] / (daily['Roll30_Total_Mean'] + eps)
    ratio_clipped = np.clip(ratio, 0, 2)

    social_amt_no_fnd = daily[[c.replace('Category_', 'Amt_') for c in social_categories]].sum(axis=1)
    daily['Social_Share_no_FnD'] = social_amt_no_fnd / (daily['Total'] + eps)

    daily['Social_Share'] = np.clip(
        daily['Social_Share_no_FnD'] + daily['FOOD_AND_DRINK_Social_Share'], 0, 1
    )
    
    daily['Social_Spend'] = (
        daily['Social_Spend'] - daily['FOOD_AND_DRINK_Solo_Score']
    ).clip(lower=0)
    
    daily['Social_Share'] = (
        daily['Social_Spend'] / (daily['Total'] + eps)
    ).clip(0, 1)

    daily['stayed_home'] = (
        (daily['Total'] < 20) &                  
        (daily.get('Social_Share', 0) < 0.05) &  
        (daily.get('Wellness_Share', 0) < 0.05)  
    ).astype(int)

    ratio = daily['Total'] / (daily['Roll30_Total_Mean'] + eps)
    ratio_clipped = np.clip(ratio, 0, 2)

    stay_home_flag = daily.get('stayed_home', 0)
    no_spend_boost = (daily['Total'] <= 0 + eps).astype(int) * 0.15
    frugality_base = 1 - (ratio_clipped / 2.0)

    social_flag = daily.get('Social_Share', 0).clip(0, 1)
    wellness_flag = daily.get('Wellness_Share', 0).clip(0, 1)
    
    # Boost for meaningful activity even if low spend; penalty for overspending
    activity_boost = 0.3 * social_flag + 0.3 * wellness_flag
    overspend_penalty = np.where(ratio > 1.5, 0.2, 0.0)
    
    daily['Activity_Index'] = np.clip(
        (ratio_clipped / 2.0) + activity_boost - overspend_penalty,
        0, 1
    )

    home_bonus = 0.1 * stay_home_flag                      # reward rest days
    social_drag = 0.2 * social_flag                        # socializing costs money
    shopping_pen = 0.2 * daily.get('Shopping_Share', 0).clip(0, 1)  # impulse buys

    daily['Frugality_Index'] = np.clip(
        frugality_base + no_spend_boost + home_bonus - social_drag - shopping_pen,
        0, 1
    )

    mood = (
        0.6 * daily['Social_Share'] +
        0.3 * daily['Wellness_Share'] -
        0.2 * daily['stayed_home']
    )
    
    daily['Mood_Index'] = np.clip(mood, 0, 1)

    daily['HasTransaction'] = (daily['Total'] > 0).astype(int)

    return daily

def build_features_and_targets(daily: pd.DataFrame, lookback_days=30):
    # ---- 1) Feature groups ----
    amt_cols   = [c for c in daily.columns if c.startswith('Amt_')]
    share_cols = [c for c in daily.columns if c.startswith('Share_')]
    roll_cols  = [c for c in daily.columns if c.startswith('Roll')]
    cal_cols   = ['DOW_sin', 'DOW_cos', 'MOY_sin', 'MOY_cos', 'IsWeekend']
    misc_cols  = [
        'Total','Social_Spend','Wellness_Spend','Bills_Spend',
        'Essentials_Spend','Entertainment_Spend','Travel_Spend','Shopping_Spend',
        'Social_Share','Wellness_Share','Bills_Share','Essentials_Share',
        'Entertainment_Share','Travel_Share','Shopping_Share',
        'FOOD_AND_DRINK_Social_Score','FOOD_AND_DRINK_Solo_Score','FOOD_AND_DRINK_Social_Share',
        'DayOfWeek','Month','Social_Share_no_FnD','stayed_home',
        'Activity_Index','Mood_Index','Frugality_Index'
    ]
    next_cal_cols = [c for c in [
        'Next_DayOfWeek','Next_Month','Next_IsWeekend',
        'Next_DOW_sin','Next_DOW_cos','Next_MOY_sin','Next_MOY_cos'
    ] if c in daily.columns]

    feat_cols = []
    for group in (amt_cols, share_cols, roll_cols, cal_cols, misc_cols, next_cal_cols):
        feat_cols.extend([c for c in group if c in daily.columns])
    # dedupe but preserve order
    feat_cols = list(dict.fromkeys(feat_cols))

    # ---- 2) Targets (predict tomorrow) ----
    target_cols = [
        'Social_Spend','Wellness_Spend','Essentials_Spend',
        'Shopping_Spend','Entertainment_Spend','Activity_Index','Mood_Index','Frugality_Index'
    ]
    target_cols = [c for c in target_cols if c in daily.columns]

    if len(target_cols) == 0:
        raise ValueError("No target columns available in 'daily'.")

    # shift -1 to align each row's features with tomorrow's targets
    y = daily[target_cols].shift(-1)

    # rows usable for training (no NaNs in targets; and if Next_* exist, they must be non-NaN)
    # Keep days with activity (tune threshold as you like)
    active_mask = daily['Total'] > 0  # or (daily['Total'] > 5)
    y = daily[target_cols].shift(-1)
    usable = active_mask & ~(y.isna().any(axis=1))
    if next_cal_cols:
        usable &= ~(daily[next_cal_cols].isna().any(axis=1))
    
    X = daily.loc[usable, feat_cols]
    y = y.loc[usable, :]

    if len(X) < lookback_days + 2:
        raise ValueError(
            f"Not enough rows after filtering to build sequences. "
            f"Have {len(X)}, need at least {lookback_days + 2}."
        )

    # ---- 3) Time-aware split (last 20% is validation) ----
    split_idx = max(lookback_days + 1, int(len(X) * 0.8))  # ensure train is long enough for sequences
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    # Guard again post-split
    if len(X_train) < lookback_days + 1 or len(X_val) < lookback_days + 1:
        # If val too small, fall back to all-train (user can do CV outside)
        X_train, y_train = X, y
        X_val,   y_val   = X.iloc[0:0], y.iloc[0:0]

    # ---- 4) Scalers and constant-target handling ----
    X_scaler = RobustScaler()
    y_scaler = MinMaxScaler(clip=True)

    X_train_s = X_scaler.fit_transform(X_train)
    X_val_s   = X_scaler.transform(X_val) if len(X_val) else np.empty((0, X_train_s.shape[1]))

    # constant targets mask (computed on TRAIN ONLY)
    const_mask_s = (y_train.nunique(dropna=False) <= 1)
    const_cols = y_train.columns[const_mask_s]
    vary_cols  = y_train.columns[~const_mask_s]

    # store constant values (if any)
    const_values = {}
    if len(const_cols) > 0:
        const_values = y_train[const_cols].iloc[0].astype(float).to_dict()

    # Scale only varying target columns
    y_train_s_df = y_train.copy()
    y_val_s_df   = y_val.copy()

    if len(vary_cols) > 0:
        y_train_s_df[vary_cols] = y_scaler.fit_transform(y_train[vary_cols])
        y_val_s_df[vary_cols]   = y_scaler.transform(y_val[vary_cols]) if len(y_val) else y_val[vary_cols]

    # Fill constant columns with zeros (any fixed value OK) in scaled space
    for c in const_cols:
        y_train_s_df[c] = 0.0
        if len(y_val_s_df):
            y_val_s_df[c] = 0.0

    y_train_s = y_train_s_df.values
    y_val_s   = y_val_s_df.values if len(y_val_s_df) else np.empty((0, len(target_cols)))

    # ---- 5) Sequence-ify ----
    def to_sequences(arr_X, arr_y, lookback):
        X_seq, y_seq = [], []
        for i in range(lookback, len(arr_X)):
            X_seq.append(arr_X[i - lookback:i, :])
            y_seq.append(arr_y[i, :])
        return np.array(X_seq), np.array(y_seq)

    X_train_seq, y_train_seq = to_sequences(X_train_s, y_train_s, lookback_days)
    X_val_seq,   y_val_seq   = to_sequences(X_val_s,   y_val_s,   lookback_days) if len(X_val_s) else (np.empty((0, lookback_days, X_train_s.shape[1])), np.empty((0, len(target_cols))))

    # ---- 6) Return everything needed downstream ----
    # const_mask aligned to target_cols order as a boolean np.array
    const_mask = const_mask_s.reindex(target_cols).fillna(False).to_numpy(bool)

    return (X_train_seq, y_train_seq, X_val_seq, y_val_seq,
            feat_cols, target_cols, X_scaler, y_scaler,
            const_mask, const_values)


def build_model(input_timesteps, input_features, output_dim, cell='lstm'):
    model = Sequential()
    model.add(Input(shape=(input_timesteps, input_features)))  # <-- add this first

    if cell.lower() == 'gru':
        model.add(GRU(64))
    else:
        model.add(LSTM(64))

    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_dim, activation='linear'))  # regression outputs

    model.compile(optimizer='adam', loss='mse')
    return model


def is_bad_description(desc):
    bad_desc = {"Transaction Date", "Posted Date", "Description", "Amount"}
    if pd.isna(desc):
        return True
    desc = str(desc).strip()
    if desc in bad_desc:
        return True
    if '@' in desc:
        return True
    # Looks numeric or negative numeric
    if re.fullmatch(r'-?\d+(?:[.,]\d+)?', desc):
        return True
    return False

def read_file_preprocess_data():
    column_names = [
        "Transaction Date",
        "Posted Date",
        "Description",
        "Amount",
        "Category"
    ]
    
    df = pd.read_csv('sampleData.csv', usecols=column_names)

    mask = df['Description'].apply(is_bad_description)
    df = df[~mask].copy()
    
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
    df['Posted Date'] = pd.to_datetime(df['Posted Date'])
    df['Amount'] = (df['Amount'].replace('[\$,]', '', regex=True).astype(float))
    df = pd.get_dummies(df, columns=['Category'])
    df = df.sort_values('Transaction Date')

    # print(df.columns)

    start_date = pd.Timestamp(f"{pd.Timestamp.today().year}-01-01")
    end_date = pd.Timestamp.today().normalize()  # current date without time
    full_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Check which dates are missing
    actual_dates = pd.Series(df['Transaction Date'].unique())
    missing_dates = full_range.difference(actual_dates)
    
    # print(f"✅ Start Date in data: {df['Transaction Date'].min().date()}")
    # print(f"✅ End Date in data:   {df['Transaction Date'].max().date()}")
    # print(f"📅 Total expected days: {len(full_range)}")
    # print(f"📊 Total days in data:  {len(actual_dates)}")
    # print(f"⚠️ Missing days: {len(missing_dates)}")
    # print(missing_dates)

    return df
    

def plot_history_with_prediction_clean(daily_df, pred, days=60, smooth_window=7):
    import matplotlib.pyplot as plt
    from matplotlib.dates import AutoDateLocator, AutoDateFormatter
    import pandas as pd
    import numpy as np

    numeric_cols = ['Total','Activity_Index','Mood_Index','Frugality_Index']

    df = daily_df[['Date'] + numeric_cols].copy()
    df['Date'] = pd.to_datetime(df['Date'], utc=False)
    df = df.sort_values('Date')

    # predicted tomorrow row
    last_date = df['Date'].max()
    tomorrow  = last_date + pd.Timedelta(days=1)
    pred_row = {
        'Date': tomorrow,
        'Total': float(pred.get('Total', 0.0)),
        'Activity_Index': float(np.clip(pred.get('Activity_Index', 0.0), 0, 1)),
        'Mood_Index': float(np.clip(pred.get('Mood_Index', 0.0), 0, 1)),
        'Frugality_Index': float(np.clip(pred.get('Frugality_Index', 0.0), 0, 1)),
    }

    recent = pd.concat([df.tail(days), pd.DataFrame([pred_row])], ignore_index=True)
    recent = recent.sort_values('Date').reset_index(drop=True)

    # ensure numeric dtypes
    recent[numeric_cols] = recent[numeric_cols].apply(pd.to_numeric, errors='coerce')

    numeric_cols = ['Total','Activity_Index','Mood_Index','Frugality_Index']

    # ensure numeric just in case
    recent[numeric_cols] = recent[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    smoothed_num = recent[numeric_cols].rolling(window=smooth_window, min_periods=1).mean()
    smoothed = pd.concat([recent[['Date']], smoothed_num], axis=1)

    for c in numeric_cols:
        smoothed[c] = recent[c].rolling(window=smooth_window, min_periods=1).mean()

    # plot (clean)
    fig, ax = plt.subplots(figsize=(10, 5)); ax2 = ax.twinx()
    ax.plot(smoothed['Date'], smoothed['Total'], color='#1f77b4', linewidth=2.5, label='Total Spend')
    ax2.plot(smoothed['Date'], smoothed['Activity_Index'], color='#2ca02c', linewidth=2, label='Activity Index')
    ax2.plot(smoothed['Date'], smoothed['Mood_Index'], color='#ff7f0e', linewidth=2, label='Mood Index')
    ax2.plot(smoothed['Date'], smoothed['Frugality_Index'], color='#9467bd', linewidth=2, label='Frugality Index')
    ax2.set_ylim(0, 1)

    ax.axvline(tomorrow, color='gray', linestyle='--', linewidth=1)
    ax.text(tomorrow, ax.get_ylim()[1]*0.95, 'Predicted', color='gray', rotation=90, va='top', ha='left', fontsize=9)

    locator = AutoDateLocator(); formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator); ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    ax.set_xlabel('Date'); ax.set_ylabel('Total Spend ($)'); ax2.set_ylabel('Indices (0–1)')
    ax.set_title(f'Last {min(days, len(df))} Days + Predicted Tomorrow (rolling={smooth_window})')

    h1,l1 = ax.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc='upper left', frameon=False, fontsize=9)

    for spine in ['top','right']: ax.spines[spine].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.grid(alpha=0.2); plt.tight_layout(); plt.show()



def predict_next_day(daily: pd.DataFrame,
                     model,
                     feat_cols,
                     target_cols,
                     X_scaler,
                     y_scaler,
                     lookback_days: int,
                     n_steps,
                     n_feats,
                     const_mask=None,
                     const_values=None):

    if const_mask is None:
        const_mask = np.zeros(len(target_cols), dtype=bool)
    else:
        const_mask = np.asarray(const_mask, dtype=bool)
    if const_values is None:
        const_values = {}

    X_all = daily[feat_cols].copy()

    # --- NEW: fill final Next_* features with tomorrow’s calendar so they’re not NaN ---
    next_cols = [c for c in X_all.columns if c.startswith("Next_")]
    if next_cols:
        # Compute tomorrow from the max Date in daily
        last_date = pd.to_datetime(daily["Date"].max())
        tomorrow  = last_date + pd.Timedelta(days=1)
        dow = tomorrow.dayofweek
        month = tomorrow.month
        is_weekend = int(dow >= 5)
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)
        moy_sin = np.sin(2 * np.pi * (month - 1) / 12)
        moy_cos = np.cos(2 * np.pi * (month - 1) / 12)

        fill_map = {
            'Next_DayOfWeek': dow,
            'Next_Month': month,
            'Next_IsWeekend': is_weekend,
            'Next_DOW_sin': dow_sin,
            'Next_DOW_cos': dow_cos,
            'Next_MOY_sin': moy_sin,
            'Next_MOY_cos': moy_cos
        }
        for col in next_cols:
            if col in fill_map:
                X_all.loc[X_all.index[-1], col] = fill_map[col]
            else:
                # any unexpected Next_* -> conservative 0
                X_all.loc[X_all.index[-1], col] = 0.0

        # If any remaining NaNs in Next_* (rare), forward-fill then back-fill
        X_all[next_cols] = X_all[next_cols].ffill().bfill()

    if len(X_all) < lookback_days:
        raise ValueError(f"Need at least {lookback_days} rows to predict next day.")

    # Scale & window
    X_all_s = X_scaler.transform(X_all)
    window = X_all_s[-lookback_days:, :][None, ...]

    # After you form `window`
    window = window.astype(np.float32, copy=False)           # dtype stable
    # Make sure lookback_days and number of features don't change across calls
    assert window.shape[1:] == (n_steps, n_feats)

    y_pred_scaled = model.predict(window, verbose=0)[0]

    # Inverse only varying targets
    vary_mask = ~const_mask
    y_pred = np.empty_like(y_pred_scaled, dtype=float)
    if vary_mask.any():
        inv = y_scaler.inverse_transform(y_pred_scaled[vary_mask].reshape(1, -1))[0]
        y_pred[vary_mask] = inv

    # Fill constants
    for i, col in enumerate(target_cols):
        if const_mask[i]:
            y_pred[i] = float(const_values.get(col, 0.0))

    # Post-process
    pred = dict(zip(target_cols, y_pred))
    for k in ['Activity_Index','Mood_Index','Frugality_Index',
              'Social_Share','Wellness_Share','Essentials_Share',
              'Entertainment_Share','Travel_Share','Shopping_Share']:
        if k in pred:
            pred[k] = float(np.clip(pred[k], 0.0, 1.0))
    for k in ['Total','Social_Spend','Wellness_Spend','Essentials_Spend',
              'Shopping_Spend','Entertainment_Spend']:
        if k in pred:
            pred[k] = float(max(0.0, pred[k]))

    components = ['Social_Spend','Wellness_Spend','Essentials_Spend',
                  'Shopping_Spend','Entertainment_Spend','Travel_Spend']
    sum_components = float(sum(pred.get(k, 0.0) for k in components))
    
    # Use the sum as the authoritative Total
    pred['Total'] = sum_components

    return pred


def main():
    df = read_file_preprocess_data()
    tmp_df = compute_scores_social(df)
    daily_df = daily_aggregate(df, tmp_df)
    daily_df = add_next_day_calendar(daily_df)

    lookback_days = 30
    (X_train_seq, y_train_seq, X_val_seq, y_val_seq,
     feat_cols, target_cols, X_scaler, y_scaler,
     const_mask, const_values) = build_features_and_targets(daily_df, lookback_days=lookback_days)

    tf.keras.backend.clear_session()
    
    model = build_model(
        input_timesteps=X_train_seq.shape[1],
        input_features=X_train_seq.shape[2],
        output_dim=len(target_cols),
        cell='lstm'  # or 'gru'
    )

    n_steps  = X_train_seq.shape[1] 
    n_feats  = X_train_seq.shape[2]
    
    # Build the model graph with a fixed input signature
    model(tf.keras.Input(shape=(n_steps, n_feats)))

    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=40,
        batch_size=32,
        verbose=1,
    )

    next_day_pred = predict_next_day(
        daily=daily_df,
        model=model,
        feat_cols=feat_cols,
        target_cols=target_cols,
        X_scaler=X_scaler,
        y_scaler=y_scaler,
        lookback_days=lookback_days,
        const_mask=const_mask,
        const_values=const_values,
        n_steps=n_steps,
        n_feats=n_feats,
    )
    print("📈 Predicted tomorrow:", next_day_pred)
    plot_history_with_prediction_clean(daily_df, next_day_pred)

main()
