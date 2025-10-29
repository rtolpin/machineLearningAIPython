import pandas as pd
import numpy as np
import re

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

    return daily

    

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
    
    df = pd.read_csv('spendingSummaryTableYTD_Final.csv', usecols=column_names)

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

df = read_file_preprocess_data()
tmp_df = compute_scores_social(df)
daily_df = daily_aggregate(df, tmp_df)
print(daily_df)