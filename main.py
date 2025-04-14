from preprocessing2 import EEGPreprocessing
import rehabilitation_analysis as rehab
import multiprocessing as mp
import os
import numpy as np
from consolidated_reports import generate_consolidated_report

def main():
    dataset_path = 'BCICIV_2a_all_patients.csv'
    print("Starting preprocessing...")
    try:
        preprocessor = EEGPreprocessing(dataset_path)  # Pass filename here
        train_epochs, val_epochs, test_epochs = preprocessor.run_preprocessing_pipeline(epoch_duration=1.0, overlap=0.5)
        print(f"Preprocessing complete. Train: {len(train_epochs)}, Validation: {len(val_epochs)}, Test: {len(test_epochs)} epochs.")

        # CPU config
        num_cpus = max(1, mp.cpu_count() - 1)

        # ----- TRAIN -----
        print("Processing TRAIN set...")
        train_features, train_pids, train_labels = rehab.extract_rehabilitation_features(train_epochs, use_parallel=True, n_jobs=num_cpus)
        train_impairments = rehab.estimate_impairment_level(train_features, train_labels, train_pids)
        train_recommendations = rehab.build_therapy_recommendation_model(train_features, train_impairments)
        train_effectiveness = rehab.predict_therapy_effectiveness(train_features, train_recommendations)

        # ----- VALIDATION (Optional for tuning) -----
        print("Processing VALIDATION set...")
        val_features, val_pids, val_labels = rehab.extract_rehabilitation_features(val_epochs, use_parallel=True, n_jobs=num_cpus)
        val_impairments = rehab.estimate_impairment_level(val_features, val_labels, val_pids)
        val_recommendations = rehab.build_therapy_recommendation_model(val_features, val_impairments)
        val_effectiveness = rehab.predict_therapy_effectiveness(val_features, val_recommendations)

        # ----- TEST -----
        print("Processing TEST set...")
        test_features, test_pids, test_labels = rehab.extract_rehabilitation_features(test_epochs, use_parallel=True, n_jobs=num_cpus)
        test_impairments = rehab.estimate_impairment_level(test_features, test_labels, test_pids)
        test_recommendations = rehab.build_therapy_recommendation_model(test_features, test_impairments)
        test_effectiveness = rehab.predict_therapy_effectiveness(test_features, test_recommendations)

        # Only generate final reports for TEST set (best practice)
        print("Generating consolidated report from TEST set...")
        report_path = generate_consolidated_report(
            test_recommendations, test_effectiveness, test_impairments
        )

        print("Done! Report available at:", report_path)

    except Exception as e:
        print(f"Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
