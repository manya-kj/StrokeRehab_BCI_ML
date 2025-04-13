from rehabilitation_analysis import run_rehabilitation_analysis
from preprocessing2 import EEGPreprocessing
import pandas as pd
import numpy as np
import os
import multiprocessing as mp
from collections import Counter

def generate_consolidated_report(patient_recommendations, effectiveness_predictions, impairment_df):
    """
    Generate a consolidated CSV report of all patients instead of individual reports.
    Does not generate any visualizations, only a detailed CSV file.
    
    Parameters:
    patient_recommendations: DataFrame with therapy recommendations
    effectiveness_predictions: DataFrame with predicted effectiveness
    impairment_df: DataFrame with estimated impairment levels
    
    Returns:
    str: Path to the generated CSV file
    """
    # Create output directory
    os.makedirs('consolidated_reports', exist_ok=True)
    
    # Merge all data
    print("Merging patient data...")
    report_data = pd.merge(
        patient_recommendations,
        effectiveness_predictions[['patient_id', 'predicted_effectiveness', 'key_indicators']],
        on='patient_id'
    )
    
    report_data = pd.merge(
        report_data,
        impairment_df[['patient_id', 'impairment_score', 'laterality_index']],
        on='patient_id'
    )
    
    def fix_patient_id(patient_id):
        # Check if it's in scientific notation (negative exponent)
        if isinstance(patient_id, float) and 'E-' in str(patient_id).upper():
            # Extract numeric value from ID string
            # Map range of values to integers 1-9
            # Simple approach: sort all unique values and assign sequential IDs
            unique_ids = sorted(report_data['patient_id'].unique())
            id_mapping = {old_id: i+1 for i, old_id in enumerate(unique_ids)}
            return id_mapping[patient_id]
        return patient_id
    
    report_data['original_patient_id'] = report_data['patient_id']
    report_data['patient_id'] = report_data['patient_id'].astype(int)


    # Total patients count
    total_patients = len(report_data)
    print(f"Processing data for {total_patients} patients...")
    
    # Categorize impairment and effectiveness
    impairment_categories = []
    effectiveness_categories = []
    
    for _, row in report_data.iterrows():
        # Impairment categorization
        if row['impairment_score'] < 0.3:
            impairment_categories.append("Mild")
        elif row['impairment_score'] < 0.7:
            impairment_categories.append("Moderate")
        else:
            impairment_categories.append("Severe")
        
        # Effectiveness categorization
        if row['predicted_effectiveness'] > 75:
            effectiveness_categories.append("Excellent")
        elif row['predicted_effectiveness'] > 50:
            effectiveness_categories.append("Good")
        elif row['predicted_effectiveness'] > 25:
            effectiveness_categories.append("Moderate")
        else:
            effectiveness_categories.append("Limited")
    
    # Add categories to dataframe
    report_data['impairment_category'] = impairment_categories
    report_data['effectiveness_category'] = effectiveness_categories
    
    # Clean up therapy names for better readability
    report_data['therapy_name'] = report_data['recommended_therapy'].apply(lambda x: x.replace('_', ' ').title())
    
    # Add therapy details
    therapy_details = {
        'constraint_induced_movement_therapy': {
            'description': 'Restricts the unaffected limb to force use of the affected limb',
            'best_for': 'Patients with mild impairment in hand/arm function',
            'frequency': '4-6 hours daily for 2-3 weeks',
            'expected_results': 'Improved arm and hand function through neural reorganization'
        },
        'robot_assisted_therapy': {
            'description': 'Uses robotic devices to assist with repetitive movement tasks',
            'best_for': 'Patients with moderate to severe impairment requiring movement assistance',
            'frequency': '30-45 minute sessions, 3-5 times weekly for 6-8 weeks',
            'expected_results': 'Improved range of motion and motor control'
        },
        'bilateral_training': {
            'description': 'Focuses on simultaneous movements of both arms to facilitate neural coupling',
            'best_for': 'Patients with moderate impairment in hand/arm function',
            'frequency': '45-60 minute sessions, 3 times weekly for 8-10 weeks',
            'expected_results': 'Improved coordination and function of affected limb'
        },
        'functional_electrical_stimulation': {
            'description': 'Uses electrical currents to stimulate nerves and muscles',
            'best_for': 'Patients with weak muscle activation or spasticity',
            'frequency': '30 minute sessions, daily for 4-6 weeks',
            'expected_results': 'Improved muscle strength and reduced spasticity'
        },
        'mental_practice_with_mirror_therapy': {
            'description': 'Combines visualization with visual feedback using mirrors',
            'best_for': 'Patients with severe impairment or those with good cognitive function',
            'frequency': '20-30 minute sessions, twice daily for 6-8 weeks',
            'expected_results': 'Improved motor planning and execution'
        }
    }
    
    # Add therapy details to the dataframe
    report_data['therapy_description'] = report_data['recommended_therapy'].apply(
        lambda x: therapy_details.get(x, {}).get('description', 'N/A')
    )
    report_data['therapy_best_for'] = report_data['recommended_therapy'].apply(
        lambda x: therapy_details.get(x, {}).get('best_for', 'N/A')
    )
    report_data['recommended_frequency'] = report_data['recommended_therapy'].apply(
        lambda x: therapy_details.get(x, {}).get('frequency', 'N/A')
    )
    report_data['expected_results'] = report_data['recommended_therapy'].apply(
        lambda x: therapy_details.get(x, {}).get('expected_results', 'N/A')
    )
    
    # Calculate additional metrics
    # 1. Average impairment by therapy type
    therapy_avg_impairment = report_data.groupby('recommended_therapy')['impairment_score'].mean()
    report_data['therapy_avg_impairment'] = report_data['recommended_therapy'].map(therapy_avg_impairment)
    
    # 2. Relative effectiveness (compared to avg for impairment category)
    category_avg_effectiveness = report_data.groupby('impairment_category')['predicted_effectiveness'].mean()
    report_data['category_avg_effectiveness'] = report_data['impairment_category'].map(category_avg_effectiveness)
    report_data['relative_effectiveness'] = report_data['predicted_effectiveness'] - report_data['category_avg_effectiveness']
    
    # 3. Priority score (higher for severe impairment + high effectiveness)
    # Scale: 1-10, weighted 70% by impairment, 30% by effectiveness
    report_data['priority_score'] = (7 * report_data['impairment_score'] * 10) + (3 * report_data['predicted_effectiveness'] / 10)
    
    # Sort by priority score (highest first)
    report_data = report_data.sort_values('priority_score', ascending=False)
    
    # Add ranks
    report_data['priority_rank'] = range(1, len(report_data) + 1)
    
    # 4. Estimated treatment duration (weeks)
    def estimate_duration(row):
        therapy = row['recommended_therapy']
        impairment = row['impairment_score']
        
        # Base durations from therapy details
        if 'constraint_induced' in therapy:
            base = 3  # 3 weeks
        elif 'robot' in therapy:
            base = 7  # 7 weeks
        elif 'bilateral' in therapy:
            base = 9  # 9 weeks
        elif 'electrical' in therapy:
            base = 5  # 5 weeks
        elif 'mirror' in therapy:
            base = 7  # 7 weeks
        else:
            base = 6  # Default
        
        # Adjust based on impairment (more severe = longer duration)
        adjustment = 1 + (impairment * 0.5)  # Up to 50% longer for severe impairment
        
        return round(base * adjustment)
    
    report_data['estimated_weeks'] = report_data.apply(estimate_duration, axis=1)
    
    # 5. Recommended follow-up interval (weeks)
    def recommend_followup(row):
        effectiveness = row['predicted_effectiveness']
        if effectiveness < 40:
            return 2  # More frequent follow-up for lower effectiveness
        elif effectiveness < 60:
            return 3
        else:
            return 4  # Less frequent for higher effectiveness
    
    report_data['followup_weeks'] = report_data.apply(recommend_followup, axis=1)
    
    # 6. Add summary statistics about cohort as separate columns
    report_data['total_patients_count'] = total_patients
    report_data['avg_cohort_impairment'] = report_data['impairment_score'].mean()
    report_data['avg_cohort_effectiveness'] = report_data['predicted_effectiveness'].mean()
    
    therapy_counts = report_data['recommended_therapy'].value_counts().to_dict()
    for therapy, count in therapy_counts.items():
        therapy_name = therapy.replace('_', ' ').title()
        column_name = f'count_{therapy.replace("_", "")}'
        report_data[column_name] = count
    
    # Save to CSV
    csv_path = 'consolidated_reports/detailed_rehabilitation_data.csv'
    report_data.to_csv(csv_path, index=False)
    print(f"Detailed CSV report generated: {csv_path}")
    
    # Generate a second, simplified CSV with just the essential columns for easy viewing
    essential_columns = [
        'patient_id', 
        'impairment_score', 
        'impairment_category',
        'therapy_name', 
        'predicted_effectiveness', 
        'effectiveness_category',
        'priority_rank', 
        'estimated_weeks',
        'followup_weeks', 
        'key_indicators'
    ]
    
    simplified_csv_path = 'consolidated_reports/simplified_rehabilitation_data.csv'
    report_data[essential_columns].to_csv(simplified_csv_path, index=False)
    print(f"Simplified CSV report generated: {simplified_csv_path}")
    
    return csv_path

def main():
    """
    Run the full pipeline and generate a consolidated CSV report.
    """
    # Set environment variables to control thread usage
    os.environ['OMP_NUM_THREADS'] = '4'  # Control OpenMP threads
    os.environ['MKL_NUM_THREADS'] = '4'  # Control Intel MKL threads
    np.set_printoptions(precision=3)     # Reduce numpy printing precision (saves memory)
    
    # Step 1: Initialize and run the EEG preprocessing pipeline
    print("Starting preprocessing of EEG data...")
    dataset_path = 'BCICIV_2a_all_patients.csv'
    preprocessor = EEGPreprocessing(dataset_path)
    
    # Run preprocessing pipeline to get epochs
    epochs = preprocessor.run_preprocessing_pipeline(
        epoch_duration=1.0,  # 1-second epochs
        overlap=0.5          # 50% overlap between epochs
    )
    
    # Step 2: Run the rehabilitation analysis pipeline with multiprocessing
    print("Starting rehabilitation analysis...")
    # Set multiprocessing usage
    num_cpus = max(1, mp.cpu_count() - 1)  # Leave one CPU free for system
    print(f"Using {num_cpus} CPU cores for processing")
    
    # Extract features, analyze patterns and estimate impairment
    print("Extracting features and analyzing patterns...")
    features_df, patient_ids, labels = run_rehabilitation_analysis.extract_rehabilitation_features(
        epochs, use_parallel=True, n_jobs=num_cpus
    )
    
    task_means = run_rehabilitation_analysis.analyze_motor_imagery_patterns(features_df, labels)
    
    # Estimate impairment levels
    print("Estimating impairment levels...")
    impairment_df = run_rehabilitation_analysis.estimate_impairment_level(features_df, labels, patient_ids)
    
    # Build therapy recommendations
    print("Building therapy recommendations...")
    patient_recommendations = run_rehabilitation_analysis.build_therapy_recommendation_model(
        features_df, impairment_df
    )
    
    # Predict therapy effectiveness
    print("Predicting therapy effectiveness...")
    effectiveness_predictions = run_rehabilitation_analysis.predict_therapy_effectiveness(
        features_df, patient_recommendations
    )
    
    # Generate consolidated report as CSV
    print("Generating consolidated CSV report...")
    csv_path = generate_consolidated_report(
        patient_recommendations, effectiveness_predictions, impairment_df
    )
    
    print("Analysis complete!")
    print(f"CSV report available at: {csv_path}")

if __name__ == "__main__":
    main()