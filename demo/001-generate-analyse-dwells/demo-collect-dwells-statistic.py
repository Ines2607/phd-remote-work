"""
Demo: Collect Dwells Statistics from Parquet Files

This script replicates the outlier detection analysis from `collect-dwells-statistics.py` 
but processes monthly parquet files from a local folder instead of querying BigQuery.

Overview:
- Load monthly parquet files from a folder
- Calculate statistics for each month (same as BigQuery query)
- Detect outliers using the same logic
- Generate HTML reports with charts
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from datetime import datetime
import base64
from io import BytesIO
import pygeohash

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def safe_float(value, default=0):
    """Safely convert value to float, handling None and NaN."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0):
    """Safely convert value to int, handling None and NaN."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 encoded string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str


def calculate_derived_columns(df):
    """
    Calculate derived columns that are in BigQuery but not in parquet files.
    
    Args:
        df (pd.DataFrame): DataFrame with raw parquet data
        
    Returns:
        pd.DataFrame: DataFrame with added derived columns
    """
    df = df.copy()
    
    # Calculate duration_hours from duration_seconds
    if 'duration_seconds' in df.columns and 'duration_hours' not in df.columns:
        df['duration_hours'] = df['duration_seconds'] / 3600
    
    # Calculate geohash from latitude/longitude if not present
    if 'geohash' not in df.columns:
        if 'centroid_latitude' in df.columns and 'centroid_longitude' in df.columns:
            df['geohash'] = df.apply(
                lambda row: pygeohash.encode(row['centroid_latitude'], row['centroid_longitude'], precision=7)
                if pd.notna(row['centroid_latitude']) and pd.notna(row['centroid_longitude'])
                else None,
                axis=1
            )
        else:
            print("Warning: No geohash or lat/lon columns found. Geohash will be missing.")
            df['geohash'] = None
    
    # Calculate flag_night from timestamp/hour
    if 'flag_night' not in df.columns:
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['flag_night'] = df['hour'].isin([20, 21, 22, 23, 0, 1, 2, 3])
        elif 'hour' in df.columns:
            df['flag_night'] = df['hour'].isin([20, 21, 22, 23, 0, 1, 2, 3])
        else:
            print("Warning: No timestamp or hour column found. flag_night will be False.")
            df['flag_night'] = False
    
    # Calculate flag_work_hours from timestamp/hour
    if 'flag_work_hours' not in df.columns:
        if 'hour' in df.columns:
            df['flag_work_hours'] = (df['hour'] >= 8) & (df['hour'] <= 17)
        else:
            print("Warning: No hour column found. flag_work_hours will be False.")
            df['flag_work_hours'] = False
    
    return df


def calculate_monthly_statistics_from_parquet(df, month):
    """
    Calculate monthly statistics from parquet data, replicating BigQuery query logic.
    
    Args:
        df (pd.DataFrame): Raw dwells data for one month
        month (str): Month identifier (YYYYMM format)
        
    Returns:
        pd.DataFrame: Single row DataFrame with all statistics for the month
    """
    # Ensure date and timestamp are datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract hour and day of week
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
    if 'date' in df.columns:
        df['day_of_week'] = df['date'].dt.dayofweek + 1  # 1=Sunday, 7=Saturday
    
    # Calculate time period
    df['time_period'] = df['hour'].apply(
        lambda h: 'morning' if 4 <= h <= 7
        else 'afternoon' if 8 <= h <= 15
        else 'evening' if 16 <= h <= 19
        else 'night'
    )
    
    # User-level statistics (group by identifier)
    # Note: Can't count 'identifier' since it's in the groupby, so we count 'date' instead
    user_stats = df.groupby(['identifier', 'identifier_type']).agg({
        'date': ['count', 'nunique'],  # count for dwells_count, nunique for days_count
        'geohash': 'nunique',  # geohashes_count
        'duration_hours': 'sum',  # total_hours
    })
    
    # Flatten multi-level columns
    user_stats.columns = ['dwells_count', 'days_count', 'geohashes_count', 'total_hours']
    user_stats = user_stats.reset_index()
    
    # Add weekday/weekend counts
    weekday_counts = df[df['day_of_week'].notna() & ~df['day_of_week'].isin([1, 7])].groupby('identifier')['date'].nunique()
    weekend_counts = df[df['day_of_week'].notna() & df['day_of_week'].isin([1, 7])].groupby('identifier')['date'].nunique()
    
    user_stats['weekdays_count'] = user_stats['identifier'].map(weekday_counts).fillna(0)
    user_stats['weekends_count'] = user_stats['identifier'].map(weekend_counts).fillna(0)
    
    # Aggregated statistics
    stats = {
        'month': month,
        # User stats
        'total_users': len(user_stats),
        'total_dwells': user_stats['dwells_count'].sum(),
        'users_gaid': len(user_stats[user_stats['identifier_type'] == 'GAID']),
        'users_idfa': len(user_stats[user_stats['identifier_type'] == 'IDFA']),
        
        # Frequency stats
        'avg_signals_per_user': user_stats['dwells_count'].mean(),
        'median_signals_per_user': user_stats['dwells_count'].median(),
        'p25_dwells': user_stats['dwells_count'].quantile(0.25),
        'p75_dwells': user_stats['dwells_count'].quantile(0.75),
        'p90_dwells': user_stats['dwells_count'].quantile(0.90),
        'p95_dwells': user_stats['dwells_count'].quantile(0.95),
        
        # User count by dwell ranges
        'users_1_dwell': len(user_stats[user_stats['dwells_count'] == 1]),
        'users_2_3_dwells': len(user_stats[(user_stats['dwells_count'] >= 2) & (user_stats['dwells_count'] <= 3)]),
        'users_4_9_dwells': len(user_stats[(user_stats['dwells_count'] >= 4) & (user_stats['dwells_count'] <= 9)]),
        'users_10_19_dwells': len(user_stats[(user_stats['dwells_count'] >= 10) & (user_stats['dwells_count'] <= 19)]),
        'users_20plus_dwells': len(user_stats[user_stats['dwells_count'] >= 20]),
        'users_4plus_dwells': len(user_stats[user_stats['dwells_count'] >= 4]),
        'users_10plus_dwells': len(user_stats[user_stats['dwells_count'] >= 10]),
        
        # Temporal stats
        'avg_days_per_user': user_stats['days_count'].mean(),
        'median_days_per_user': user_stats['days_count'].median(),
        'avg_weekdays_per_user': user_stats['weekdays_count'].mean(),
        'avg_weekends_per_user': user_stats['weekends_count'].mean(),
        
        # Spatial stats
        'avg_unique_geohashes_per_user': user_stats['geohashes_count'].mean(),
        'median_unique_geohashes_per_user': user_stats['geohashes_count'].median(),
        
        # Duration stats (user-level)
        'avg_total_hours_per_user': user_stats['total_hours'].mean(),
        'median_total_hours_per_user': user_stats['total_hours'].median(),
    }
    
    # Dwell-level statistics
    stats['avg_duration_hours'] = df['duration_hours'].mean()
    stats['median_duration_hours'] = df['duration_hours'].median()
    stats['dwells_over_2_hours'] = len(df[df['duration_hours'] > 2])
    stats['dwells_over_4_hours'] = len(df[df['duration_hours'] > 4])
    stats['dwells_over_180_seconds'] = len(df[df['duration_seconds'] > 180])
    
    stats['avg_bump_count'] = df['bump_count'].mean() if 'bump_count' in df.columns else 0
    stats['median_bump_count'] = df['bump_count'].median() if 'bump_count' in df.columns else 0
    
    # Time period counts
    stats['dwells_morning'] = len(df[df['time_period'] == 'morning'])
    stats['dwells_afternoon'] = len(df[df['time_period'] == 'afternoon'])
    stats['dwells_evening'] = len(df[df['time_period'] == 'evening'])
    stats['dwells_night'] = len(df[df['time_period'] == 'night'])
    
    # Work/night hours
    stats['dwells_work_hours'] = len(df[df['flag_work_hours'] == True]) if 'flag_work_hours' in df.columns else 0
    stats['dwells_night_hours'] = len(df[df['flag_night'] == True]) if 'flag_night' in df.columns else 0
    
    # Weekday/weekend
    stats['total_weekday_dwells'] = len(df[df['day_of_week'].notna() & ~df['day_of_week'].isin([1, 7])])
    stats['total_weekend_dwells'] = len(df[df['day_of_week'].notna() & df['day_of_week'].isin([1, 7])])
    
    # Total unique geohashes
    stats['total_unique_geohashes'] = df['geohash'].nunique() if 'geohash' in df.columns else 0
    
    # Calculate shares
    stats['share_gaid'] = safe_float(stats['users_gaid'] / stats['total_users']) if stats['total_users'] > 0 else 0
    stats['share_idfa'] = safe_float(stats['users_idfa'] / stats['total_users']) if stats['total_users'] > 0 else 0
    stats['share_frequent_4plus'] = safe_float(stats['users_4plus_dwells'] / stats['total_users']) if stats['total_users'] > 0 else 0
    stats['share_frequent_10plus'] = safe_float(stats['users_10plus_dwells'] / stats['total_users']) if stats['total_users'] > 0 else 0
    stats['share_dwells_over_2_hours'] = safe_float(stats['dwells_over_2_hours'] / stats['total_dwells']) if stats['total_dwells'] > 0 else 0
    stats['share_dwells_over_4_hours'] = safe_float(stats['dwells_over_4_hours'] / stats['total_dwells']) if stats['total_dwells'] > 0 else 0
    stats['share_dwells_over_180_seconds'] = safe_float(stats['dwells_over_180_seconds'] / stats['total_dwells']) if stats['total_dwells'] > 0 else 0
    stats['avg_dwells_per_day_per_user'] = safe_float(
        stats['avg_signals_per_user'] / stats['avg_days_per_user']
    ) if stats['avg_days_per_user'] > 0 else 0
    
    return pd.DataFrame([stats])


def load_monthly_parquet_files(parquet_folder):
    """
    Load all monthly parquet files and calculate statistics.
    
    Args:
        parquet_folder (str): Path to folder containing parquet files
        
    Returns:
        pd.DataFrame: DataFrame with one row per month containing all statistics
    """
    # Find all parquet files
    parquet_files = glob.glob(os.path.join(parquet_folder, "*.parquet"))
    
    if len(parquet_files) == 0:
        raise ValueError(f"No parquet files found in {parquet_folder}")
    
    print(f"Found {len(parquet_files)} parquet files")
    
    all_monthly_stats = []
    
    for file_path in sorted(parquet_files):
        # Extract month from filename (assume format YYYYMM.parquet)
        filename = os.path.basename(file_path)
        month = filename.replace('.parquet', '').replace('data_filtered_', '')
        
        # Validate month format (should be 6 digits: YYYYMM)
        if not month.isdigit() or len(month) != 6:
            print(f"Warning: Skipping {filename} - invalid month format")
            continue
        
        print(f"\nProcessing {month}...")
        
        try:
            # Load parquet file
            df = pd.read_parquet(file_path)
            print(f"  Loaded {len(df):,} rows")
            
            # Calculate derived columns
            df = calculate_derived_columns(df)
            
            # Calculate monthly statistics
            month_stats = calculate_monthly_statistics_from_parquet(df, month)
            all_monthly_stats.append(month_stats)
            
            print(f"  ✓ Calculated statistics for {month}")
            
        except Exception as e:
            print(f"  ✗ Error processing {month}: {e}")
            continue
    
    if len(all_monthly_stats) == 0:
        raise ValueError("No months were successfully processed")
    
    # Combine all months
    monthly_df = pd.concat(all_monthly_stats, ignore_index=True)
    print(f"\n✓ Processed {len(monthly_df)} months successfully")
    
    return monthly_df


def rename_columns_for_analysis(monthly_df):
    """
    Rename columns to match expected format with prefixes.
    
    Args:
        monthly_df (pd.DataFrame): DataFrame with statistics
        
    Returns:
        pd.DataFrame: DataFrame with renamed columns
    """
    df = monthly_df.copy()
    column_mapping = {}
    
    for col in df.columns:
        if col == 'month':
            continue
        elif col.startswith('total_users') or col.startswith('users_') or col.startswith('share_') or col.startswith('avg_signals') or col.startswith('median_signals'):
            column_mapping[col] = f"user_{col}"
        elif col.startswith('total_dwells') or col.startswith('avg_dwells') or col.startswith('median_dwells') or col.startswith('p') or col.startswith('users_'):
            column_mapping[col] = f"freq_{col}"
        elif col.startswith('avg_days') or col.startswith('median_days') or col.startswith('dwells_morning') or col.startswith('dwells_afternoon') or col.startswith('dwells_evening') or col.startswith('dwells_night') or col.startswith('dwells_work') or col.startswith('dwells_night_hours') or col.startswith('total_week'):
            column_mapping[col] = f"temp_{col}"
        elif col.startswith('avg_unique') or col.startswith('median_unique') or col.startswith('total_unique'):
            column_mapping[col] = f"spatial_{col}"
        elif col.startswith('avg_duration') or col.startswith('median_duration') or col.startswith('dwells_over') or col.startswith('share_dwells') or col.startswith('avg_bump') or col.startswith('median_bump') or col.startswith('avg_total_hours') or col.startswith('median_total_hours'):
            column_mapping[col] = f"dur_{col}"
        elif col.startswith('dwells_over_180') or col.startswith('share_dwells_over_180'):
            column_mapping[col] = f"qual_{col}"
    
    df = df.rename(columns=column_mapping)
    return df


def detect_outliers_by_month(monthly_df):
    """
    Detect outliers for each month across all numeric columns.
    Outliers are defined as:
    - Values less than 2 standard deviations below the median
    - Values equal to 0 (for number columns)
    - Values equal to 1 (for share columns)
    
    Args:
        monthly_df (pd.DataFrame): Monthly statistics DataFrame with 'month' column
        
    Returns:
        pd.DataFrame: DataFrame with columns: month, outlier_count, total_columns, 
                     outlier_percentage, flag
    """
    # Get all numeric columns (excluding 'month')
    numeric_cols = monthly_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'month' in numeric_cols:
        numeric_cols.remove('month')
    
    if len(numeric_cols) == 0:
        print("Warning: No numeric columns found for outlier detection")
        return pd.DataFrame(columns=['month', 'outlier_count', 'total_columns', 
                                    'outlier_percentage', 'flag'])
    
    # Initialize outlier tracking
    outlier_matrix = pd.DataFrame(index=monthly_df.index)
    outlier_matrix['month'] = monthly_df['month'].values
    
    # For each numeric column, detect outliers
    for col in numeric_cols:
        if col not in monthly_df.columns:
            continue
        
        # Check if this is a share column
        is_share_col = 'share' in col.lower()
        
        # Initialize outlier flags for this column
        is_outlier = pd.Series(False, index=monthly_df.index)
        
        # Outlier condition 1: Values less than 2 std dev below median
        median_val = monthly_df[col].median()
        std_val = monthly_df[col].std()
        
        # Only check std dev outlier if std is valid and not 0
        if not (pd.isna(std_val) or std_val == 0):
            lower_bound = median_val - 2 * std_val
            # Only flag values LESS than lower bound (not greater than upper bound)
            is_outlier = is_outlier | (monthly_df[col] < lower_bound)
        
        # Outlier condition 2: Values equal to 0 (for number columns)
        if not is_share_col:
            is_outlier = is_outlier | (monthly_df[col] == 0)
        
        # Outlier condition 3: Values equal to 1 (for share columns)
        if is_share_col:
            is_outlier = is_outlier | (monthly_df[col] == 1)
        
        outlier_matrix[col] = is_outlier.astype(int)
    
    # Aggregate by month: count how many columns each month is an outlier in
    outlier_cols = [col for col in outlier_matrix.columns if col != 'month']
    outlier_matrix['outlier_count'] = outlier_matrix[outlier_cols].sum(axis=1)
    
    # Calculate total columns analyzed (excluding month column)
    total_columns = len(outlier_cols)
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'month': outlier_matrix['month'],
        'outlier_count': outlier_matrix['outlier_count'],
        'total_columns': total_columns,
        'outlier_percentage': (outlier_matrix['outlier_count'] / total_columns * 100).round(2)
    })
    
    # Add flag for months with > 25% outlier rate
    result_df['flag'] = result_df['outlier_percentage'].apply(
        lambda x: 'REMOVE FROM ANALYSIS' if x > 25 else ''
    )
    
    # Add warning flag for months with 10-25% outlier rate (for yellow highlighting)
    result_df['warning'] = result_df['outlier_percentage'].apply(
        lambda x: True if 10 <= x <= 25 else False
    )
    
    # Sort by month
    result_df = result_df.sort_values('month').reset_index(drop=True)
    
    return result_df


def generate_outlier_report(outlier_df, output_dir):
    """Generate HTML report for outlier analysis."""
    table_rows = ""
    for _, row in outlier_df.iterrows():
        if row['flag']:
            row_class = ' class="remove-row"'
        elif row['warning']:
            row_class = ' class="warning-row"'
        else:
            row_class = ''
        table_rows += f"""
        <tr{row_class}>
            <td>{row['month']}</td>
            <td>{int(row['outlier_count'])}</td>
            <td>{int(row['total_columns'])}</td>
            <td>{row['outlier_percentage']:.2f}%</td>
            <td><strong>{row['flag']}</strong></td>
        </tr>
        """
    
    flagged_count = len(outlier_df[outlier_df['flag'] != ''])
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Outlier Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border: 1px solid #ddd; }}
            th {{ background-color: #e74c3c; color: white; font-weight: bold; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            tr.remove-row {{ background-color: #ffebee; font-weight: bold; }}
            tr.remove-row td {{ color: #c62828; }}
            tr.warning-row {{ background-color: #fff9c4; }}
            tr.warning-row td {{ color: #f57f17; }}
            .summary {{ background-color: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 20px 0; }}
            .summary strong {{ color: #856404; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Outlier Analysis Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Total Months Analyzed:</strong> {len(outlier_df)}</p>
            
            <div class="summary">
                <strong>Summary:</strong> Outliers are defined as:
                <ul style="margin: 10px 0; padding-left: 20px;">
                    <li>Values less than 2 standard deviations below the median</li>
                    <li>Values equal to 0 (for number columns)</li>
                    <li>Values equal to 1 (for share columns)</li>
                </ul>
                <ul style="margin: 10px 0; padding-left: 20px;">
                    <li>Months with <strong>over 25%</strong> outlier rate are flagged for removal (red highlight)</li>
                    <li>Months with <strong>10-25%</strong> outlier rate are highlighted in yellow (warning)</li>
                </ul>
                <br>
                <strong>Months Flagged for Removal:</strong> {flagged_count} out of {len(outlier_df)}
            </div>
            
            <h2>Outlier Analysis by Month</h2>
            <table>
                <thead>
                    <tr>
                        <th>Month</th>
                        <th>Number of Times Outlier</th>
                        <th>Total Columns Analyzed</th>
                        <th>Outlier Percentage</th>
                        <th>Flag</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    
    output_file = os.path.join(output_dir, 'outlier_analysis_report.html')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"  - outlier_analysis_report.html")


def create_trend_chart(df, x_col, y_col, title):
    """Create a trend chart and return as base64 encoded string."""
    fig, ax = plt.subplots(figsize=(12, 6))
    df_sorted = df.sort_values(x_col)
    ax.plot(df_sorted[x_col], df_sorted[y_col], marker='o', linewidth=2)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    img_str = fig_to_base64(fig)
    plt.close(fig)
    return img_str


def create_day_period_share_chart(df):
    """Create a stacked bar chart showing share of dwells by day period and return as base64."""
    df = df.copy()
    df['month_dt'] = pd.to_datetime(df['month'], format='%Y%m', errors='coerce')
    df = df.sort_values('month_dt').dropna(subset=['month_dt'])
    
    total_dwells_col = 'freq_total_dwells'
    if total_dwells_col not in df.columns:
        print(f"Warning: {total_dwells_col} not found in DataFrame columns")
        return None
    
    df['share_morning'] = (df['temp_dwells_morning'] / df[total_dwells_col] * 100).fillna(0)
    df['share_afternoon'] = (df['temp_dwells_afternoon'] / df[total_dwells_col] * 100).fillna(0)
    df['share_evening'] = (df['temp_dwells_evening'] / df[total_dwells_col] * 100).fillna(0)
    df['share_night'] = (df['temp_dwells_night'] / df[total_dwells_col] * 100).fillna(0)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x_pos = range(len(df))
    width = 0.6
    
    ax.bar(x_pos, df['share_morning'], width, label='Morning (4-7)', color='#f39c12', alpha=0.8)
    ax.bar(x_pos, df['share_afternoon'], width, bottom=df['share_morning'], 
           label='Afternoon (8-15)', color='#3498db', alpha=0.8)
    ax.bar(x_pos, df['share_evening'], width, 
           bottom=df['share_morning'] + df['share_afternoon'],
           label='Evening (16-19)', color='#9b59b6', alpha=0.8)
    ax.bar(x_pos, df['share_night'], width,
           bottom=df['share_morning'] + df['share_afternoon'] + df['share_evening'],
           label='Night (20-3)', color='#2c3e50', alpha=0.8)
    
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Share of Dwells (%)', fontsize=12)
    ax.set_title('Share of Dwells by Day Period Over Time', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.strftime('%Y-%m') for m in df['month_dt']], rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    plt.tight_layout()
    
    img_str = fig_to_base64(fig)
    plt.close(fig)
    return img_str


def create_weekday_weekend_share_chart(df):
    """Create a stacked bar chart showing share of weekday vs weekend dwells and return as base64."""
    df = df.copy()
    df['month_dt'] = pd.to_datetime(df['month'], format='%Y%m', errors='coerce')
    df = df.sort_values('month_dt').dropna(subset=['month_dt'])
    
    total_dwells_col = 'freq_total_dwells'
    if total_dwells_col not in df.columns:
        print(f"Warning: {total_dwells_col} not found in DataFrame columns")
        return None
    
    df['share_weekday'] = (df['temp_total_weekday_dwells'] / df[total_dwells_col] * 100).fillna(0)
    df['share_weekend'] = (df['temp_total_weekend_dwells'] / df[total_dwells_col] * 100).fillna(0)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x_pos = range(len(df))
    width = 0.6
    
    ax.bar(x_pos, df['share_weekday'], width, label='Weekday (Sunday-Thursday)', color='#3498db', alpha=0.8)
    ax.bar(x_pos, df['share_weekend'], width, bottom=df['share_weekday'],
           label='Weekend (Friday-Saturday)', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Share of Dwells (%)', fontsize=12)
    ax.set_title('Share of Dwells by Weekday vs Weekend Over Time', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.strftime('%Y-%m') for m in df['month_dt']], rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    plt.tight_layout()
    
    img_str = fig_to_base64(fig)
    plt.close(fig)
    return img_str


def generate_monthly_report(df, output_dir):
    """Generate HTML report for all months."""
    key_metrics = [
        ('user_total_users', 'User Total Users'),
        ('freq_total_dwells', 'Freq Total Dwells'),
        ('temp_avg_days_per_user', 'Temp Avg Days Per User'),
        ('user_share_frequent_10plus', 'User Share Frequent 10Plus'),
        ('spatial_total_unique_geohashes', 'Spatial Total Unique Geohashes'),
        ('spatial_avg_unique_geohashes_per_user', 'Spatial Avg Unique Geohashes Per User'),
        ('dur_median_duration_hours', 'Dur Median Duration Hours'),
        ('dur_median_bump_count', 'Dur Median Bump Count'),
        ('user_share_gaid', 'User Share Gaid'),
    ]
    
    chart_html = ""
    
    for metric, title in key_metrics:
        if metric in df.columns:
            try:
                img_str = create_trend_chart(df, 'month', metric, title)
                chart_html += f'<h3>{title}</h3><img src="data:image/png;base64,{img_str}" style="max-width: 100%; margin: 20px 0;"><br>'
            except Exception as e:
                print(f"Warning: Could not create chart for {metric}: {e}")
                continue
    
    if all(col in df.columns for col in ['temp_dwells_morning', 'temp_dwells_afternoon', 
                                         'temp_dwells_evening', 'temp_dwells_night', 'freq_total_dwells']):
        try:
            img_str = create_day_period_share_chart(df)
            if img_str:
                chart_html += f'<h3>Temp Share Dwells by Day Period</h3><img src="data:image/png;base64,{img_str}" style="max-width: 100%; margin: 20px 0;"><br>'
        except Exception as e:
            print(f"Warning: Could not create day period share chart: {e}")
    
    if all(col in df.columns for col in ['temp_total_weekday_dwells', 'temp_total_weekend_dwells', 'freq_total_dwells']):
        try:
            img_str = create_weekday_weekend_share_chart(df)
            if img_str:
                chart_html += f'<h3>Temp Share Dwells by Weekday vs Weekend</h3><img src="data:image/png;base64,{img_str}" style="max-width: 100%; margin: 20px 0;"><br>'
        except Exception as e:
            print(f"Warning: Could not create weekday/weekend share chart: {e}")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dwells Statistics - All Months</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            h3 {{ color: #7f8c8d; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Dwells Statistics - All Months Summary</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Total Months:</strong> {len(df)}</p>
            
            <h2>Trend Charts</h2>
            {chart_html}
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, 'all_months_report.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)


def generate_aggregated_reports(monthly_df, output_dir):
    """Generate HTML report and CSV for monthly statistics."""
    monthly_df.to_csv(os.path.join(output_dir, 'all_months_summary.csv'), index=False)
    generate_monthly_report(monthly_df, output_dir)
    
    print(f"\nReports saved to: {output_dir}")
    print("  - all_months_report.html")
    print("  - all_months_summary.csv")


def main(parquet_folder, output_dir=None):
    """
    Main execution function - processes parquet files and generates reports.
    
    Args:
        parquet_folder (str): Path to folder containing monthly parquet files
        output_dir (str): Output directory for reports (default: project_root/results)
    """
    print("="*60)
    print("Starting outlier detection analysis from parquet files")
    print("="*60)
    
    # Set default output directory (project root / results)
    if output_dir is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        output_dir = os.path.join(project_root, 'results')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Step 1: Load monthly parquet files and calculate statistics
    print("\nStep 1: Loading parquet files and calculating statistics...")
    monthly_df = load_monthly_parquet_files(parquet_folder)
    
    # Step 2: Rename columns to match expected format
    print("\nStep 2: Renaming columns...")
    monthly_df = rename_columns_for_analysis(monthly_df)
    
    # Step 3: Detect outliers
    print("\nStep 3: Detecting outliers...")
    outlier_df = detect_outliers_by_month(monthly_df)
    generate_outlier_report(outlier_df, output_dir)
    
    # Step 4: Generate aggregated reports
    print("\nStep 4: Generating reports...")
    generate_aggregated_reports(monthly_df, output_dir)
    
    print("\n" + "="*60)
    print("✓ Analysis complete!")
    print("="*60)
    print(f"\nOutput files saved to: {output_dir}")
    print("  - all_months_summary.csv")
    print("  - all_months_report.html")
    print("  - outlier_analysis_report.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Collect dwells statistics from parquet files and detect outliers'
    )
    parser.add_argument(
        '--parquet-folder', 
        type=str, 
        required=True,
        help='Path to folder containing monthly parquet files (one file per month, named YYYYMM.parquet)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=None,
        help='Output directory for reports (default: project_root/results)'
    )
    
    args = parser.parse_args()
    
    main(args.parquet_folder, args.output_dir)

