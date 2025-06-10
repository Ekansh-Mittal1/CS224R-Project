import pandas as pd
import glob
from datetime import datetime, timedelta
import os

def extrapolate_missing_data(df):
    """
    Extrapolates missing data points in the DataFrame by filling gaps with the last known close price.
    Also adds a column indicating whether each row was extrapolated or original.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock price data
        
    Returns:
        pd.DataFrame: DataFrame with extrapolated data points and is_extrapolated column
    """
    # Convert ts_event to datetime if it's not already
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    
    # Sort by timestamp
    df = df.sort_values('ts_event')
    
    # Create a complete time series with 1-second intervals
    start_time = df['ts_event'].min()
    end_time = df['ts_event'].max()
    all_times = pd.date_range(start=start_time, end=end_time, freq='1S')
    
    # Create a new DataFrame with all timestamps
    complete_df = pd.DataFrame({'ts_event': all_times})
    
    # Merge with original data
    merged_df = pd.merge(complete_df, df, on='ts_event', how='left')
    
    # Forward fill the close prices
    merged_df['close'] = merged_df['close'].ffill()
    
    # For extrapolated rows, set open to the previous close
    mask = merged_df['open'].isna()
    merged_df.loc[mask, 'open'] = merged_df.loc[mask, 'close']
    
    # Set high, low to the same value as close for extrapolated rows
    merged_df.loc[mask, 'high'] = merged_df.loc[mask, 'close']
    merged_df.loc[mask, 'low'] = merged_df.loc[mask, 'close']
    
    # Set volume to 0 for extrapolated rows
    merged_df.loc[mask, 'volume'] = 0
    
    # Add is_extrapolated column
    merged_df['is_extrapolated'] = mask
    
    # Forward fill other columns that should be preserved
    columns_to_ffill = ['rtype', 'publisher_id', 'instrument_id', 'symbol']
    for col in columns_to_ffill:
        merged_df[col] = merged_df[col].ffill()
    
    return merged_df

# Create output directory if it doesn't exist
output_dir = 'data/extrapolated'
os.makedirs(output_dir, exist_ok=True)

# Get list of all CSV files in data directory
data_files = glob.glob('data/og/*.csv')

# Iterate through each file
for file in data_files:
    # Read CSV into DataFrame
    df = pd.read_csv(file)
    
    # Extrapolate missing data
    extrapolated_df = extrapolate_missing_data(df)
    
    # Get just the filename without the path
    filename = os.path.basename(file)
    
    # Save the extrapolated data to the new file
    output_path = os.path.join(output_dir, f"extrapolated_{filename}")
    extrapolated_df.to_csv(output_path, index=False)
    
