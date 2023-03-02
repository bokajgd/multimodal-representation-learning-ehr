"""Script for expanding numerical features to binary features. """

import pandas as pd
import numpy as np

def expand_numeric_cols_to_binary_percentile_cols(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Expand numerical columns to binary columns based on percentile ranges.
    From each column with numerical data, it creates 7 new columns with binary values 
    representing whether the original value is among the top 1%, 1-5%, 5-25%, 25-75%, 
    75-95%, 95%-99%, 99-100% of the values in the column.
    
    Args:
        feature_df (pd.DataFrame): DataFrame with flattened features.
        
    Returns:
        pd.DataFrame: DataFrame with expanded features.
    """
    
    # select only columns with numeric data types and drop patient_id column
    numeric_cols = feature_df.select_dtypes(include=np.number).drop(columns=['patient_id'])
    
    # initialize an empty list to store the new DataFrames for each column
    expanded_data = []

    # iterate over each column
    for col in numeric_cols.columns:

        col_data = numeric_cols[col].values
        
        # calc the percentiles for the column data
        p1, p5, p25, p75, p95, p99 = np.percentile(col_data, [1, 5, 25, 75, 95, 99])
        
        # calc new binary columns for each percentile range
        col_p1 = np.where(col_data <= p1, 1, 0)
        col_p5 = np.where((col_data > p1) & (col_data <= p5), 1, 0)
        col_p25 = np.where((col_data > p5) & (col_data <= p25), 1, 0)
        col_p50 = np.where((col_data > p25) & (col_data <= p75), 1, 0)
        col_p75 = np.where((col_data > p75) & (col_data <= p95), 1, 0)
        col_p95 = np.where((col_data > p95) & (col_data <= p99), 1, 0)
        col_p99 = np.where(col_data > p99, 1, 0)
        
        # create a new df with the expanded columns for the current column
        new_data = pd.DataFrame({f'{col}_p1': col_p1, f'{col}_p5': col_p5, f'{col}_p25': col_p25, 
                                 f'{col}_p50': col_p50, f'{col}_p75': col_p75, f'{col}_p95': col_p95, 
                                 f'{col}_p99': col_p99})
        
        # append the new DataFrame to the list of expanded DataFrames
        expanded_data.append(new_data)
    
    # concatenated all the expanded DataFrames along the columns axis
    output_data = pd.concat([feature_df] + expanded_data, axis=1)
    
    # Return the output DataFrame
    return output_data