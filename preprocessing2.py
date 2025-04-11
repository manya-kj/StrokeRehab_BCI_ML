import os
import numpy as np
import pandas as pd
import scipy.signal
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging

class EEGPreprocessing:
    def __init__(self, dataset_path):
        """
        Initialize EEG preprocessing pipeline
        """
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
    
    def load_data(self):
        """
        Load raw EEG data from CSV file.
        """
        try:
            self.raw_data = pd.read_csv(self.dataset_path)
            logging.info("EEG data loaded successfully")
            return self.raw_data
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise e
    
    def filter_data(self, data):
        """
        Apply band-pass filtering to remove noise.
        """
        try:
            filtered_data = data.apply(
                lambda x: scipy.signal.lfilter(
                    *scipy.signal.butter(4, [0.5, 40], btype='band', fs=250),
                    x
                ) if pd.api.types.is_numeric_dtype(x) else x
            )
            logging.info("Band-pass filtering applied")
            return filtered_data
        except Exception as e:
            logging.error(f"Filtering failed: {e}")
            raise e
    
    def normalize_data(self, data):
        """
            Normalize EEG signals while excluding non-numeric columns.
        """
        try:
        # Select only numeric columns (e.g., EEG signal data)
            numeric_data = data.select_dtypes(include=[np.number])
        
        # Normalize numeric EEG data
            scaler = StandardScaler()
            normalized_numeric_data = pd.DataFrame(
                scaler.fit_transform(numeric_data),
                columns=numeric_data.columns
            )
        
        # Merge normalized data with non-numeric columns (e.g., labels)
            non_numeric_data = data.select_dtypes(exclude=[np.number])
            normalized_data = pd.concat([normalized_numeric_data, non_numeric_data], axis=1)
        
            logging.info("Data normalization completed successfully")
            return normalized_data
        except Exception as e:
            logging.error(f"Normalization failed: {e}")
        raise e
    
    def segment_into_epochs(self, data, epoch_duration, overlap):
        """
        Segment data into epochs based on duration and overlap.
        """
        try:
            fs = 250  # Assuming a sampling frequency of 250 Hz
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
        """
        Visualize preprocessing stages with extensive error handling
        """
        try:
            os.makedirs('results', exist_ok=True)
            plot_columns = original_data.columns[:min(3, len(original_data.columns))]
            
            plt.figure(figsize=(20, 15))
            plt.suptitle('EEG Preprocessing Stages', fontsize=16)
            
            # Original Signal
            plt.subplot(3, 1, 1)
            plt.title('Original Signal')
            for col in plot_columns:
                plt.plot(original_data[col].head(500), label=col)
            plt.legend()
            plt.ylabel('Amplitude')
            
            # Cleaned Signal
            plt.subplot(3, 1, 2)
            plt.title('Noise-Cleaned Signal')
            for col in plot_columns:
                plt.plot(cleaned_data[col].head(500), label=col)
            plt.legend()
            plt.ylabel('Amplitude')
            
            # First Few Epochs
            plt.subplot(3, 1, 3)
            plt.title('Epoched Signal')
            if epochs and len(epochs) > 0:
                for i in range(min(5, len(epochs))):
                    epoch_data = epochs[i]
                    for col in plot_columns:
                        plt.plot(epoch_data[col], label=f'Epoch {i} - {col}')
            else:
                plt.text(0.5, 0.5, 'No epochs available', 
                         horizontalalignment='center', 
                         verticalalignment='center')
            
            plt.legend()
            plt.xlabel('Samples')
            plt.ylabel('Normalized Amplitude')
            plt.tight_layout()
            plt.savefig('results/eeg_preprocessing_stages.png', dpi=300)
            plt.close()
            
            logging.info("Preprocessing visualization saved successfully")
        except Exception as e:
            logging.error(f"Visualization error: {e}")
            raise e
    
    def run_preprocessing_pipeline(self, epoch_duration=1.0, overlap=0.5):
        """
        Full preprocessing pipeline for EEG data.
        """
        try:
            # Load raw data
            data = self.load_data()
            
            # Apply filtering
            cleaned_data = self.filter_data(data)
            
            # Normalize data
            normalized_data = self.normalize_data(cleaned_data)
            
            # Segment into epochs
            self.epochs = self.segment_into_epochs(normalized_data, epoch_duration, overlap)
            
            # Visualize preprocessing stages
            self.visualize_preprocessing(data, cleaned_data, self.epochs)
            
            logging.info("Preprocessing pipeline completed successfully")
            return self.epochs
        except Exception as e:
            logging.error(f"Preprocessing pipeline failed: {e}")
            raise e


def main():
    # Path to your EEG dataset
    dataset_path = 'BCICIV_2a_all_patients.csv'
    
    try:
        # Initialize preprocessor
        preprocessor = EEGPreprocessing(dataset_path)
        
        # Run preprocessing pipeline
        processed_epochs = preprocessor.run_preprocessing_pipeline(
            epoch_duration=1.0,  # 1-second epochs
            overlap=0.5          # 50% overlap between epochs
        )
        
        print(f"Preprocessing complete. Epochs shape: {len(processed_epochs)}")
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()