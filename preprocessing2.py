import os
import numpy as np
import pandas as pd
import scipy.signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging

class EEGPreprocessing:
    def __init__(self, dataset_path):
        os.makedirs('results', exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            filename='results/eeg_preprocessing.log'
        )

        self.dataset_path = dataset_path
        self.raw_data = None
        self.preprocessed_data = None
        self.epochs = None

    # Make sure load_data is defined as part of the class
    def load_data(self):
        try:
            self.raw_data = pd.read_csv(self.dataset_path)

            # Force patient ID to integer type for safety
            if 'patient' in self.raw_data.columns:
                self.raw_data['patient'] = self.raw_data['patient'].astype(int)

            logging.info("EEG data loaded successfully")
            return self.raw_data
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise e

    def filter_data(self, data):
        try:
            filtered_data = data.copy()
            eeg_columns = [col for col in data.columns if col.startswith('EEG-')]
            for col in eeg_columns:
                b, a = scipy.signal.butter(4, [0.5, 40], btype='band', fs=250)
                filtered_data[col] = scipy.signal.lfilter(b, a, data[col])
            logging.info("Band-pass filtering applied")
            return filtered_data
        except Exception as e:
            logging.error(f"Filtering failed: {e}")
            raise e

    def normalize_data(self, data):
        try:
            # Define columns to exclude from normalization
            exclude_cols = ['patient', 'label', 'time', 'epoch']
            eeg_columns = [col for col in data.columns if col.startswith('EEG-')]

            scaler = StandardScaler()
            normalized_eeg = pd.DataFrame(
                scaler.fit_transform(data[eeg_columns]),
                columns=eeg_columns
            )

            # Combine normalized EEG with original metadata
            metadata = data[exclude_cols].reset_index(drop=True)
            normalized_data = pd.concat([metadata, normalized_eeg], axis=1)

            logging.info("Data normalization completed successfully")
            return normalized_data
        except Exception as e:
            logging.error(f"Normalization failed: {e}")
            raise e

    def segment_into_epochs(self, data, epoch_duration, overlap):
        try:
            fs = 250  # Sampling frequency
            samples_per_epoch = int(epoch_duration * fs)
            step_size = int(samples_per_epoch * (1 - overlap))

            epochs = []
            for start in range(0, len(data) - samples_per_epoch + 1, step_size):
                end = start + samples_per_epoch
                epochs.append(data.iloc[start:end])
            logging.info(f"Segmented into {len(epochs)} epochs")
            return epochs
        except Exception as e:
            logging.error(f"Epoch segmentation failed: {e}")
            raise e

    def visualize_preprocessing(self, original_data, cleaned_data, epochs):
        try:
            os.makedirs('results', exist_ok=True)
            plot_columns = [col for col in original_data.columns if col.startswith('EEG-')][:3]

            plt.figure(figsize=(20, 15))
            plt.suptitle('EEG Preprocessing Stages', fontsize=16)

            # Original
            plt.subplot(3, 1, 1)
            plt.title('Original Signal')
            for col in plot_columns:
                plt.plot(original_data[col].head(500), label=col)
            plt.legend()
            plt.ylabel('Amplitude')

            # Cleaned
            plt.subplot(3, 1, 2)
            plt.title('Filtered + Normalized Signal')
            for col in plot_columns:
                plt.plot(cleaned_data[col].head(500), label=col)
            plt.legend()
            plt.ylabel('Amplitude')

            # Epochs
            plt.subplot(3, 1, 3)
            plt.title('First Few Epochs')
            if epochs:
                for i in range(min(3, len(epochs))):
                    for col in plot_columns:
                        plt.plot(epochs[i][col], label=f'Epoch {i} - {col}')
            plt.legend()
            plt.xlabel('Samples')
            plt.ylabel('Normalized Amplitude')
            plt.tight_layout()
            plt.savefig('results/eeg_preprocessing_stages.png', dpi=300)
            plt.close()

            logging.info("Preprocessing visualization saved")
        except Exception as e:
            logging.error(f"Visualization error: {e}")
            raise e

    def run_preprocessing_pipeline(self, epoch_duration=1.0, overlap=0.5):
        try:
            data = self.load_data()  # Make sure this method is being called
            cleaned = self.filter_data(data)
            normalized = self.normalize_data(cleaned)
            epochs = self.segment_into_epochs(normalized, epoch_duration, overlap)
            self.visualize_preprocessing(data, normalized, epochs)
            
            # Split by patient-wise epochs
            epoch_labels = [epoch['label'].iloc[0] for epoch in epochs]
            patient_ids = [epoch['patient'].iloc[0] for epoch in epochs]

            # Initial train-test split (80/20)
            train_val_epochs, test_epochs = train_test_split(
                epochs, test_size=0.2, random_state=42, stratify=patient_ids
            )

            # From train_val split, extract validation set (20% of train_val → 16% of total)
            train_epochs, val_epochs = train_test_split(
                train_val_epochs, test_size=0.2, random_state=42, stratify=[e['patient'].iloc[0] for e in train_val_epochs]
            )

            return train_epochs, val_epochs, test_epochs

        except Exception as e:
            logging.error(f"Preprocessing pipeline failed: {e}")
            raise e
