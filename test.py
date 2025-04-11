import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc
)

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class MotorImageryFeatureExtractor:
    """
    Extracts advanced features from EEG epochs
    """
    @staticmethod
    def extract_features(epochs):
        """
        Extract comprehensive features from EEG epochs
        
        Parameters:
        -----------
        epochs : list of pd.DataFrame
            List of EEG epochs
        
        Returns:
        --------
        tuple: Features and corresponding labels
        """
        features = []
        labels = []
        
        for epoch in epochs:
            # Extract EEG columns
            eeg_cols = [col for col in epoch.columns if col.startswith('EEG')]
            eeg_data = epoch[eeg_cols].values
            
            # Time-domain features
            time_features = [
                np.mean(eeg_data, axis=0),     # Mean
                np.std(eeg_data, axis=0),      # Standard Deviation
                np.max(eeg_data, axis=0),      # Max
                np.min(eeg_data, axis=0)       # Min
            ]
            
            # Frequency-domain features (Power Spectral Density)
            f, psd = signal.welch(eeg_data.T)
            freq_features = [
                np.mean(psd, axis=1),          # Mean PSD
                np.max(psd, axis=1)            # Max PSD
            ]
            
            # Combine features
            epoch_features = np.concatenate(time_features + freq_features)
            features.append(epoch_features)
            
            # Get label (assuming 'label' column exists)
            labels.append(epoch['label'].iloc[0])
        
        return np.array(features), np.array(labels)

class MotorImageryClassifier:
    """
    Performs classification using traditional and deep learning methods
    """
    def __init__(self, features, labels):
        """
        Initialize classifier
        
        Parameters:
        -----------
        features : np.ndarray
            Input features
        labels : np.ndarray
            Corresponding labels
        """
        self.X = features
        self.y = labels
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Returns:
        --------
        tuple: Training and testing data
        """
        # Ensure we have enough samples for splitting
        if len(self.X) < 5:
            raise ValueError("Not enough samples to split. Increase epoch generation or reduce test_size.")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=self.y
        )
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def traditional_ml_classifier(self):
        """
        Perform grid search for SVM classifier
        
        Returns:
        --------
        tuple: Best model and parameters
        """
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(probability=True))
        ])
        
        param_grid = {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__gamma': ['scale', 'auto']
        }
        
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        best_model = grid_search.best_estimator_
        
        # Performance evaluation
        y_pred = best_model.predict(self.X_test)
        print("Best Parameters:", grid_search.best_params_)
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        return best_model, grid_search.best_params_
    
    def deep_learning_model(self):
        """
        Create and train deep learning model
        
        Returns:
        --------
        tuple: Trained model and training history
        """
        # Reshape input for deep learning
        X_train_reshaped = self.X_train.reshape(
            self.X_train.shape[0], 
            self.X_train.shape[1], 
            1
        )
        X_test_reshaped = self.X_test.reshape(
            self.X_test.shape[0], 
            self.X_test.shape[1], 
            1
        )
        
        # One-hot encode labels
        y_train_encoded = to_categorical(self.y_train)
        y_test_encoded = to_categorical(self.y_test)
        
        # Model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(
                64, 3, 
                activation='relu', 
                input_shape=(X_train_reshaped.shape[1], 1)
            ),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(
                len(np.unique(self.y_train)), 
                activation='softmax'
            )
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=5
        )
        
        # Train model
        history = model.fit(
            X_train_reshaped, y_train_encoded,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr]
        )
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(
            X_test_reshaped, y_test_encoded
        )
        print(f"Test Accuracy: {test_accuracy*100:.2f}%")
        
        return model, history

class ResultVisualizer:
    """
    Visualizes classification results
    """
    @staticmethod
    def plot_results(model, X_test, y_test, history=None):
        """
        Create comprehensive visualization of results
        
        Parameters:
        -----------
        model : keras.Model or sklearn.Pipeline
            Trained classification model
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        history : keras.callbacks.History, optional
            Training history for deep learning models
        """
        plt.figure(figsize=(15, 10))
        
        # Confusion Matrix
        plt.subplot(2, 2, 1)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        
        # ROC Curve
        plt.subplot(2, 2, 2)
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        y_pred_proba = model.predict_proba(X_test)
        
        for i in range(len(np.unique(y_test))):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC Curve (class {i}, AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        
        # Training History (if deep learning)
        if history:
            plt.subplot(2, 2, 3)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.subplot(2, 2, 4)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('motor_imagery_analysis_results.png')
        plt.close()

def main():
    """
    Main execution pipeline
    """
    try:
        # Import your preprocessing class
        from preprocessing2 import EEGPreprocessing
        
        # Path to your dataset
        dataset_path = 'BCICIV_2a_all_patients.csv'
        
        # Initialize preprocessor
        preprocessor = EEGPreprocessing(dataset_path)
        
        # Run preprocessing pipeline
        processed_epochs = preprocessor.run_preprocessing_pipeline()
        
        # Feature Extraction
        features, labels = MotorImageryFeatureExtractor.extract_features(processed_epochs)
        
        # Classification
        classifier = MotorImageryClassifier(features, labels)
        X_train, X_test, y_train, y_test = classifier.prepare_data()
        
        # Traditional ML
        svm_model, best_params = classifier.traditional_ml_classifier()
        
        # Deep Learning
        dl_model, history = classifier.deep_learning_model()
        
        # Visualization
        ResultVisualizer.plot_results(dl_model, X_test, y_test, history)
        
        print("Motor Imagery Analysis Complete!")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()