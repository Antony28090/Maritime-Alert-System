import pandas as pd
import os
import glob

# Configuration for bounding box (Tamil Nadu & Sri Lanka context)
LAT_MIN = 5.5
LAT_MAX = 14.0
LON_MIN = 76.0
LON_MAX = 83.0

RAW_DATA_DIR = os.path.join("data", "Raw")
PROCESSED_DATA_DIR = os.path.join("data", "Processed")

def process_files():
    # Ensure processed directory exists
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
        print(f"Created directory: {PROCESSED_DATA_DIR}")

    # Get list of all CSV files in Raw directory
    csv_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {RAW_DATA_DIR}")
        return

    print(f"Found {len(csv_files)} files to process.")

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")
        
        try:
            # Read CSV
            df = pd.read_csv(file_path)
            
            # Check if required columns exist
            required_cols = ['cell_ll_lat', 'cell_ll_lon']
            if not all(col in df.columns for col in required_cols):
                print(f"Skipping {filename}: Missing required columns {required_cols}")
                continue

            # Filter data based on bounding box
            filtered_df = df[
                (df['cell_ll_lat'] >= LAT_MIN) & (df['cell_ll_lat'] <= LAT_MAX) &
                (df['cell_ll_lon'] >= LON_MIN) & (df['cell_ll_lon'] <= LON_MAX)
            ]
            
            # Save if data remains after filtering
            if not filtered_df.empty:
                output_path = os.path.join(PROCESSED_DATA_DIR, filename)
                filtered_df.to_csv(output_path, index=False)
                print(f"Saved {len(filtered_df)} rows to {output_path}")
            else:
                print(f"No data in range for {filename}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    process_files()
