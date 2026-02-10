import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Ensure we can import model.py from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import FedModel

# --- Setup Paths ---
# Get the folder where this script lives (src)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root (FedSentinel)
project_root = os.path.dirname(current_dir)

DATA_DIR = os.path.join(project_root, "data", "raw_html")
RESULTS_DIR = os.path.join(project_root, "data", "results")

# Create results folder if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_analysis():
    # 1. Initialize the AI Model
    print("--- Initializing BERT Model (this may take a moment) ---")
    model = FedModel()
    
    # 2. Load the scraped text files
    print(f"--- Loading data from {DATA_DIR} ---")
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory not found at {DATA_DIR}")
        return

    df = model.load_data_from_dir(DATA_DIR)
    
    if len(df) < 2:
        print("Not enough data to analyze. Need at least 2 statements.")
        return
        
    print(f"Loaded {len(df)} statements successfully.")
    
    # 3. Calculate Semantic Drift
    # We compare the current statement (t) with the previous one (t-1)
    print("--- Calculating Semantic Drift ---")
    
    similarities = []
    dates = []
    
    # Start loop from index 1 (the second item) so we can compare to index 0
    for i in range(1, len(df)):
        current_text = df.iloc[i]['text']
        prev_text = df.iloc[i-1]['text']
        current_date = df.iloc[i]['date']
        
        # Calculate score (1.0 = Same, 0.0 = Different)
        similarity_score = model.compare_statements(current_text, prev_text)
        
        # Convert to "Drift" (0.0 = Same, 1.0 = Different)
        drift = 1 - similarity_score
        
        similarities.append(drift)
        dates.append(current_date)
        
        # Print progress every 10 items
        if i % 10 == 0:
            print(f"Analyzed {current_date.date()}: Drift Score = {drift:.4f}")

    # 4. Save Data to CSV
    results_df = pd.DataFrame({
        'date': dates,
        'semantic_drift': similarities
    })
    
    csv_path = os.path.join(RESULTS_DIR, 'semantic_drift.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Saved drift data to: {csv_path}")

    # 5. Generate and Save Plot
    print("--- Generating Visualization ---")
    plt.figure(figsize=(12, 6))
    
    # Plot the line
    plt.plot(results_df['date'], results_df['semantic_drift'], 
             color='#1f77b4', linewidth=2, label='Semantic Drift')
    
    # Add styling
    plt.title('Federal Reserve Semantic Drift (Policy Shifts)', fontsize=16)
    plt.ylabel('Magnitude of Change (0=No Change, 1=Total Change)', fontsize=12)
    plt.xlabel('Year', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save image
    plot_path = os.path.join(RESULTS_DIR, 'drift_plot.png')
    plt.savefig(plot_path)
    print(f"Graph saved to: {plot_path}")
    
    # Show plot window
    plt.show()

if __name__ == "__main__":
    run_analysis()