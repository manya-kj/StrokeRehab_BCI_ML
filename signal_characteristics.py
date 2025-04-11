import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_dataset_details(df):
    # Unique patient analysis
    print("Patient Analysis:")
    print(f"Total Patients: {df['patient'].nunique()}")
    print(f"Patient IDs: {sorted(df['patient'].unique())}")
    
    # Label distribution
    print("\nLabel Distribution:")
    label_dist = df['label'].value_counts(normalize=True)
    print(label_dist)
    
    # Epoch analysis
    print("\nEpoch Analysis:")
    print(f"Total Epochs: {df['epoch'].nunique()}")
    print(f"Epochs per Patient:")
    print(df.groupby('patient')['epoch'].nunique())
    
    # EEG Signal Statistics
    eeg_columns = [col for col in df.columns if col.startswith('EEG-')]
    print("\nEEG Signal Statistics:")
    signal_stats = df[eeg_columns].agg(['mean', 'std', 'min', 'max'])
    print(signal_stats)

# Visualization of Signal Variations
def visualize_signals(df, num_channels=4):
    eeg_columns = [col for col in df.columns if col.startswith('EEG-')]
    selected_channels = eeg_columns[:num_channels]
    
    plt.figure(figsize=(15, 10))
    for i, channel in enumerate(selected_channels, 1):
        plt.subplot(num_channels, 1, i)
        plt.title(f'Signal Variation - {channel}')
        plt.plot(df[channel])
    
    plt.tight_layout()
    plt.show()

# Usage
df = pd.read_csv('BCICIV_2a_all_patients.csv')
analyze_dataset_details(df)
visualize_signals(df)