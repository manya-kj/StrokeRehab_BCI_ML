# from preprocessing2 import EEGPreprocessing  # Import the EEGPreprocessing class directly
# from rehabilitation_analysis import run_rehabilitation_analysis

# def main():
#     # Step 1: Initialize and run the EEG preprocessing pipeline
#     print("Starting preprocessing of EEG data...")
#     dataset_path = 'BCICIV_2a_all_patients.csv'
#     preprocessor = EEGPreprocessing(dataset_path)
    
#     # Run preprocessing pipeline to get epochs
#     epochs = preprocessor.run_preprocessing_pipeline(
#         epoch_duration=1.0,  # 1-second epochs
#         overlap=0.5          # 50% overlap between epochs
#     )
    
#     # Step 2: Run the rehabilitation analysis pipeline
#     print("Starting rehabilitation analysis...")
#     results = run_rehabilitation_analysis(epochs)
    
#     print("Project execution complete.")
#     print("Check 'patient_reports' directory for detailed rehabilitation recommendations.")

# if __name__ == "__main__":
#     main()

from preprocessing2 import EEGPreprocessing
import rehabilitation_analysis as rehab
import multiprocessing as mp
import os
import numpy as np
from consolidated_reports import generate_consolidated_report

def main():
    # Set environment variables to control thread usage
    os.environ['OMP_NUM_THREADS'] = '4'  # Control OpenMP threads
    os.environ['MKL_NUM_THREADS'] = '4'  # Control Intel MKL threads
    np.set_printoptions(precision=3)     # Reduce numpy printing precision (saves memory)
    
    # Optional: Disable unnecessary matplotlib features
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    # Step 1: Initialize and run the EEG preprocessing pipeline
    print("Starting preprocessing of EEG data...")
    dataset_path = 'BCICIV_2a_all_patients.csv'
    preprocessor = EEGPreprocessing(dataset_path)
    
    # Run preprocessing pipeline to get epochs
    epochs = preprocessor.run_preprocessing_pipeline(
        epoch_duration=1.0,  # 1-second epochs
        overlap=0.5          # 50% overlap between epochs
    )
    
    # Optional: Process only a subset for quick testing
    # epochs = epochs[:500]  # Comment this line for full processing
    
    # Step 2: Run the rehabilitation analysis pipeline with multiprocessing
    print("Starting rehabilitation analysis...")
    # Set multiprocessing usage
    num_cpus = max(1, mp.cpu_count() - 1)  # Leave one CPU free for system
    print(f"Using {num_cpus} CPU cores for processing")
    
    # Extract features, analyze patterns and estimate impairment
    print("Extracting features and analyzing patterns...")
    features_df, patient_ids, labels = rehab.extract_rehabilitation_features(
        epochs, use_parallel=True, n_jobs=num_cpus
    )
    
    task_means = rehab.analyze_motor_imagery_patterns(features_df, labels)
    
    # Estimate impairment levels
    print("Estimating impairment levels...")
    impairment_df = rehab.estimate_impairment_level(features_df, labels, patient_ids)
    
    # Build therapy recommendations
    print("Building therapy recommendations...")
    patient_recommendations = rehab.build_therapy_recommendation_model(
        features_df, impairment_df
    )
    
    # Predict therapy effectiveness
    print("Predicting therapy effectiveness...")
    effectiveness_predictions = rehab.predict_therapy_effectiveness(
        features_df, patient_recommendations
    )
    
    # Generate consolidated report
    print("Generating consolidated report...")
    report_path = generate_consolidated_report(
        patient_recommendations, effectiveness_predictions, impairment_df
    )
    
    print("Analysis complete!")
    print(f"Consolidated report available at: {report_path}")

if __name__ == "__main__":
    main()