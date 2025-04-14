import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import os
from scipy import signal
from joblib import Parallel, delayed
import multiprocessing as mp


def extract_rehabilitation_features(epochs, use_parallel=False, n_jobs=1):
    """
    Extract clinically relevant features from preprocessed EEG epochs.
    
    Parameters:
    epochs (list): List of dataframes containing preprocessed EEG epochs
    use_parallel (bool): Whether to use parallel processing
    n_jobs (int): Number of jobs for parallel processing
    
    Returns:
    features_df: DataFrame with extracted features
    patient_ids: Array of patient IDs
    labels: Array of task labels
    """
    # Define a helper function for parallel processing
    def process_epoch(epoch):
        # Get patient ID and label
        patient_id = epoch['patient'].iloc[0] if 'patient' in epoch.columns else None
        label = epoch['label'].iloc[0] if 'label' in epoch.columns else None
        
        # Select only EEG signal columns
        eeg_columns = [col for col in epoch.columns if col.startswith('EEG-')]
        
        # Extract features per channel
        channel_features = {}
        for channel in eeg_columns:
            signal_data = epoch[channel].values
            
            # Time domain features
            time_features = {
                f"{channel}_mean": np.mean(signal_data),
                f"{channel}_std": np.std(signal_data),
                f"{channel}_skew": scipy.stats.skew(signal_data),
                f"{channel}_kurtosis": scipy.stats.kurtosis(signal_data),
                f"{channel}_amplitude_range": np.max(signal_data) - np.min(signal_data)
            }
            
            # Frequency domain features - optimize FFT calculation
            fft = np.abs(np.fft.rfft(signal_data))
            freqs = np.fft.rfftfreq(len(signal_data), d=1/250)  # Assuming 250 Hz sampling rate
            
            # Pre-calculate indices for better performance
            delta_idx = np.logical_and(freqs >= 0.5, freqs < 4)
            theta_idx = np.logical_and(freqs >= 4, freqs < 8)
            alpha_idx = np.logical_and(freqs >= 8, freqs < 13)
            beta_idx = np.logical_and(freqs >= 13, freqs < 30)
            gamma_idx = np.logical_and(freqs >= 30, freqs < 100)
            
            # Faster summation by pre-filtering
            delta_sum = np.sum(fft[delta_idx]) if np.any(delta_idx) else 0
            theta_sum = np.sum(fft[theta_idx]) if np.any(theta_idx) else 0
            alpha_sum = np.sum(fft[alpha_idx]) if np.any(alpha_idx) else 0
            beta_sum = np.sum(fft[beta_idx]) if np.any(beta_idx) else 0
            gamma_sum = np.sum(fft[gamma_idx]) if np.any(gamma_idx) else 0
            
            freq_features = {
                f"{channel}_delta": delta_sum,
                f"{channel}_theta": theta_sum,
                f"{channel}_alpha": alpha_sum,
                f"{channel}_beta": beta_sum,
                f"{channel}_gamma": gamma_sum,
                f"{channel}_alpha_beta_ratio": alpha_sum / (beta_sum + 1e-10)
            }
            
            channel_features.update(time_features)
            channel_features.update(freq_features)
        
        # Create feature vector with patient ID
        feature_vector = {**channel_features, 'patient_id': patient_id}
        return feature_vector, patient_id, label
    
    # Run processing in parallel or sequentially
    if use_parallel and n_jobs > 1:
        print(f"Extracting features in parallel using {n_jobs} jobs...")
        results = Parallel(n_jobs=n_jobs)(delayed(process_epoch)(epoch) for epoch in epochs)
        features_data, patient_ids, labels = zip(*results)
    else:
        features_data = []
        patient_ids = []
        labels = []
        for i, epoch in enumerate(epochs):
            feature_vector, patient_id, label = process_epoch(epoch)
            features_data.append(feature_vector)
            patient_ids.append(patient_id)
            labels.append(label)
    
    # Convert to dataframe
    features_df = pd.DataFrame(features_data)
    
    return features_df, np.array(patient_ids), np.array(labels)

    unique_patient_ids = np.unique(patient_ids)
    id_mapping = {old_id: i+1 for i, old_id in enumerate(sorted(unique_patient_ids))}
    
    # Map the IDs to integers
    patient_ids = np.array([id_mapping[pid] for pid in patient_ids])
    features_df['patient_id'] = features_df['patient_id'].map(id_mapping)
    
    return features_df, np.array(patient_ids), np.array(labels)

def analyze_motor_imagery_patterns(features_df, labels):
    """
    Analyze patterns in motor imagery by comparing features across different tasks.
    
    Parameters:
    features_df: DataFrame with extracted features
    labels: Array of task labels
    
    Returns:
    task_means: DataFrame with average features per task
    """
    # Convert labels to categories
    label_map = {1: 'right_hand', 2: 'left_hand', 3: 'tongue', 4: 'foot'}
    task_labels = [label_map.get(label, str(label)) for label in labels]
    
    # Add task labels to features
    features_df = features_df.copy()
    features_df['task'] = task_labels
    
    # Calculate mean features per task
    task_means = features_df.groupby('task').mean()
    
    # Create output directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Create visualizations of key differences between tasks
    plt.figure(figsize=(15, 10))
    
    # Analyze alpha power across tasks for key channels (C3, C4, Cz)
    key_channels = ['EEG-C3', 'EEG-C4', 'EEG-Cz']
    for i, channel in enumerate(key_channels):
        alpha_col = f"{channel}_alpha"
        if alpha_col in task_means.columns:
            plt.subplot(2, 2, i+1)
            task_means[alpha_col].plot(kind='bar', title=f'Alpha Power - {channel}')
    
    # Compare beta/alpha ratio across tasks
    plt.subplot(2, 2, 4)
    ratio_cols = [col for col in task_means.columns if 'alpha_beta_ratio' in col]
    if ratio_cols:
        task_means[ratio_cols].plot(kind='bar', title='Alpha/Beta Ratio by Task')
    
    plt.tight_layout()
    plt.savefig('outputs/task_pattern_analysis.png')
    
    return task_means

def estimate_impairment_level(features_df, labels, patient_ids):
    """
    Estimate motor impairment level based on EEG patterns.
    
    Parameters:
    features_df: DataFrame with extracted features
    labels: Array of task labels
    patient_ids: Array of patient IDs
    
    Returns:
    impairment_df: DataFrame with estimated impairment levels per patient
    """
    # Map labels to tasks
    label_map = {1: 'right_hand', 2: 'left_hand', 3: 'tongue', 4: 'foot'}
    task_labels = [label_map.get(label, str(label)) for label in labels]
    
    # Add to features
    features_df = features_df.copy()
    features_df['task'] = task_labels
    features_df['patient_id'] = patient_ids
    
    # Calculate laterality index for each patient
    patients_data = []
    
    for patient in features_df['patient_id'].unique():
        patient_data = features_df[features_df['patient_id'] == patient]
        
        # Calculate for hand movements (comparing C3 vs C4 activation)
        right_hand = patient_data[patient_data['task'] == 'right_hand']
        left_hand = patient_data[patient_data['task'] == 'left_hand']
        
        # If we have both tasks for this patient
        if not right_hand.empty and not left_hand.empty:
            # Calculate C3/C4 alpha power ratio (indicates motor cortex activation)
            if 'EEG-C3_alpha' in patient_data.columns and 'EEG-C4_alpha' in patient_data.columns:
                right_c3c4_ratio = right_hand['EEG-C3_alpha'].mean() / (right_hand['EEG-C4_alpha'].mean() + 1e-10)
                left_c3c4_ratio = left_hand['EEG-C3_alpha'].mean() / (left_hand['EEG-C4_alpha'].mean() + 1e-10)
                
                # Calculate laterality index
                laterality_index = (right_c3c4_ratio - left_c3c4_ratio) / (right_c3c4_ratio + left_c3c4_ratio + 1e-10)
                
                # Higher absolute value = better differentiation between tasks
                # Lower value = potential impairment
                impairment_score = 1 - min(1, abs(laterality_index))  # 0 = no impairment, 1 = max impairment
                
                patients_data.append({
                    'patient_id': patient,
                    'laterality_index': laterality_index,
                    'impairment_score': impairment_score
                })
        else:
            # If we don't have both tasks, use alternative assessment
            # Based on overall alpha/beta ratio across all tasks
            if 'EEG-C3_alpha_beta_ratio' in patient_data.columns and 'EEG-C4_alpha_beta_ratio' in patient_data.columns:
                c3_ratio = patient_data['EEG-C3_alpha_beta_ratio'].mean()
                c4_ratio = patient_data['EEG-C4_alpha_beta_ratio'].mean()
                
                # Higher alpha/beta ratio in motor areas suggests less activation
                # Use average of both hemispheres as impairment indicator
                impairment_score = min(1, (c3_ratio + c4_ratio) / 4)
                
                patients_data.append({
                    'patient_id': patient,
                    'laterality_index': 0,  # Not applicable
                    'impairment_score': impairment_score
                })
    
    impairment_df = pd.DataFrame(patients_data)
    
    # Create visualization of impairment scores
    plt.figure(figsize=(10, 6))
    plt.bar(impairment_df['patient_id'].astype(str), impairment_df['impairment_score'])
    plt.xlabel('Patient ID')
    plt.ylabel('Estimated Impairment Score')
    plt.title('Motor Impairment Estimation Based on EEG Patterns')
    plt.ylim(0, 1)
    plt.savefig('outputs/impairment_scores.png')
    
    return impairment_df

def build_therapy_recommendation_model(features_df, impairment_df):
    """
    Build a model to recommend therapy based on EEG patterns and impairment.
    
    Parameters:
    features_df: DataFrame with extracted features
    impairment_df: DataFrame with estimated impairment levels
    
    Returns:
    patient_recommendations: DataFrame with therapy recommendations per patient
    """
    # Merge features with impairment data
    features_df = features_df.copy()
    merged_data = pd.merge(
        features_df, 
        impairment_df[['patient_id', 'impairment_score']], 
        on='patient_id'
    )
    
    # Define therapy categories based on impairment and EEG patterns
    def assign_therapy(row):
        # These rules are illustrative - customize based on clinical guidelines
        
        # Check if primarily hand motor imagery affected
        is_hand_issue = False
        if 'task' in row:
            if row['task'] in ['right_hand', 'left_hand']:
                # Compare alpha power in motor cortex regions
                if ('EEG-C3_alpha' in row and 'EEG-C4_alpha' in row and 
                    abs(row['EEG-C3_alpha'] - row['EEG-C4_alpha']) < 0.5):
                    is_hand_issue = True
        
        # Assign therapy based on impairment level and affected area
        if row['impairment_score'] < 0.3:  # Mild impairment
            if is_hand_issue:
                return "constraint_induced_movement_therapy"
            else:
                return "robot_assisted_therapy"
        elif row['impairment_score'] < 0.7:  # Moderate impairment
            if is_hand_issue:
                return "bilateral_training"
            else:
                return "functional_electrical_stimulation"
        else:  # Severe impairment
            return "mental_practice_with_mirror_therapy"
    
    # Apply the therapy assignment
    merged_data['recommended_therapy'] = merged_data.apply(assign_therapy, axis=1)
    
    # Aggregate to patient level to get final recommendations
    patient_recommendations = merged_data.groupby('patient_id')['recommended_therapy'].agg(
        lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else 'mental_practice_with_mirror_therapy'
    ).reset_index()
    
    return patient_recommendations

def predict_therapy_effectiveness(features_df, patient_recommendations):
    """
    Predict effectiveness of recommended therapies based on EEG patterns.
    
    Parameters:
    features_df: DataFrame with extracted features
    patient_recommendations: DataFrame with therapy recommendations
    
    Returns:
    effectiveness_predictions: DataFrame with predicted effectiveness per patient
    """
    # Merge features with recommendations
    features_df = features_df.copy()
    merged_data = pd.merge(
        features_df, 
        patient_recommendations, 
        on='patient_id'
    )
    
    # Define features that predict successful outcomes for each therapy
    therapy_success_indicators = {
        'constraint_induced_movement_therapy': [
            'EEG-C3_beta', 'EEG-C4_beta',  # Motor planning
            'EEG-FC3_alpha', 'EEG-FC4_alpha'  # Frontal connectivity
        ],
        'robot_assisted_therapy': [
            'EEG-C3_alpha_beta_ratio', 'EEG-C4_alpha_beta_ratio',  # Motor learning potential
            'EEG-P3_alpha', 'EEG-P4_alpha'  # Spatial processing
        ],
        'bilateral_training': [
            'EEG-C3_gamma', 'EEG-C4_gamma',  # Neural plasticity
            'EEG-Cz_theta'  # Midline connectivity
        ],
        'functional_electrical_stimulation': [
            'EEG-Cz_beta', 'EEG-Fz_beta',  # Sensorimotor integration
            'EEG-C3_delta', 'EEG-C4_delta'  # Cortical excitability
        ],
        'mental_practice_with_mirror_therapy': [
            'EEG-O1_alpha', 'EEG-O2_alpha',  # Visual processing
            'EEG-F3_theta', 'EEG-F4_theta'  # Executive function
        ]
    }
    
    # Calculate effectiveness score for each patient
    results = []
    for patient in merged_data['patient_id'].unique():
        patient_data = merged_data[merged_data['patient_id'] == patient]
        therapy = patient_data['recommended_therapy'].iloc[0]
        
        # Get relevant indicators for this therapy
        indicators = therapy_success_indicators.get(therapy, [])
        
        # Filter to available indicators
        available_indicators = [ind for ind in indicators if ind in patient_data.columns]
        
        if available_indicators:
            # Calculate average z-score of indicators
            indicator_values = patient_data[available_indicators].mean()
            all_patients = merged_data[available_indicators].mean()
            
            # Compare to overall population
            z_scores = (indicator_values - all_patients) / (merged_data[available_indicators].std() + 1e-10)
            
            # Calculate effectiveness score (0-100)
            effectiveness = min(100, max(0, 50 + 10 * z_scores.mean()))
            
            results.append({
                'patient_id': patient,
                'recommended_therapy': therapy,
                'predicted_effectiveness': effectiveness,
                'key_indicators': ', '.join(available_indicators[:3])  # Top 3 indicators
            })
        else:
            # No specific indicators available, use average effectiveness
            results.append({
                'patient_id': patient,
                'recommended_therapy': therapy,
                'predicted_effectiveness': 50.0,  # Average effectiveness
                'key_indicators': 'N/A'
            })
    
    return pd.DataFrame(results)

def generate_patient_reports(patient_recommendations, effectiveness_predictions, impairment_df):
    """
    Generate comprehensive reports for each patient.
    
    Parameters:
    patient_recommendations: DataFrame with therapy recommendations
    effectiveness_predictions: DataFrame with predicted effectiveness
    impairment_df: DataFrame with estimated impairment levels
    
    Returns:
    None (saves reports to files)
    """
    # Merge all data
    report_data = pd.merge(
        patient_recommendations,
        effectiveness_predictions[['patient_id', 'predicted_effectiveness', 'key_indicators']],
        on='patient_id'
    )
    
    report_data = pd.merge(
        report_data,
        impairment_df[['patient_id', 'impairment_score']],
        on='patient_id'
    )
    
    # Therapy descriptions
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
    
    # Generate reports directory
    os.makedirs('patient_reports', exist_ok=True)
    
    for _, row in report_data.iterrows():
        patient_id = row['patient_id']
        therapy = row['recommended_therapy']
        effectiveness = row['predicted_effectiveness']
        impairment = row['impairment_score']
        key_indicators = row['key_indicators']
        
        # Categorize impairment
        if impairment < 0.3:
            impairment_level = "Mild"
        elif impairment < 0.7:
            impairment_level = "Moderate"
        else:
            impairment_level = "Severe"
        
        # Categorize effectiveness
        if effectiveness > 75:
            effectiveness_level = "Excellent"
        elif effectiveness > 50:
            effectiveness_level = "Good"
        elif effectiveness > 25:
            effectiveness_level = "Moderate"
        else:
            effectiveness_level = "Limited"
        
        # Get therapy details
        therapy_info = therapy_details.get(therapy, {})
        
        # Create report
        report = f"""
        # Rehabilitation Recommendation Report - Patient {patient_id}
        
        ## Motor Impairment Assessment
        - **Impairment Level**: {impairment_level} (Score: {impairment:.2f})
        
        ## Recommended Rehabilitation Therapy
        - **Primary Recommendation**: {therapy.replace('_', ' ').title()}
        - **Predicted Effectiveness**: {effectiveness_level} ({effectiveness:.1f}%)
        
        ## Therapy Details
        - **Description**: {therapy_info.get('description', 'N/A')}
        - **Best For**: {therapy_info.get('best_for', 'N/A')}
        - **Recommended Frequency**: {therapy_info.get('frequency', 'N/A')}
        - **Expected Outcomes**: {therapy_info.get('expected_results', 'N/A')}
        
        ## Implementation Notes
        - Begin with supervised sessions to ensure proper technique
        - Gradually increase duration and intensity as tolerated
        - Monitor for fatigue and adjust accordingly
        - Combine with daily home exercises for optimal results
        
        ## Progress Monitoring
        - Reassess every 2-3 weeks
        - Track functional improvements using standardized assessments
        - Adjust therapy parameters based on progress
        
        ## Key EEG Indicators
        - {key_indicators}
        
        Generated based on EEG motor imagery pattern analysis
        """
        
        # Write to file
        with open(f'patient_reports/patient_{patient_id}_report.md', 'w') as f:
            f.write(report)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Effectiveness gauge
        plt.subplot(1, 2, 1)
        plt.pie([effectiveness, 100-effectiveness], colors=['#2ca02c', '#d3d3d3'], 
                startangle=90, counterclock=False)
        plt.title(f'Predicted Effectiveness: {effectiveness:.1f}%')
        plt.axis('equal')
        
        # Impairment level
        plt.subplot(1, 2, 2)
        plt.barh(['Impairment'], [impairment*100], color='#d62728')
        plt.barh(['Impairment'], [100-impairment*100], left=[impairment*100], color='#d3d3d3')
        plt.xlim(0, 100)
        plt.title(f'Impairment Level: {impairment_level}')
        
        plt.tight_layout()
        plt.savefig(f'patient_reports/patient_{patient_id}_visualization.png')
        plt.close()
    
    print(f"Generated reports for {len(report_data)} patients in 'patient_reports' directory.")

def run_rehabilitation_analysis(epochs, use_parallel=False, n_jobs=1):
    """
    Run the complete rehabilitation analysis pipeline on preprocessed EEG epochs.
    
    Parameters:
    epochs: List of dataframes containing preprocessed EEG epochs
    use_parallel: Whether to use parallel processing
    n_jobs: Number of jobs for parallel processing
    
    Returns:
    Dictionary with results of each analysis step
    """
    print(f"Working with {len(epochs)} preprocessed epochs")
    
    # Extract rehabilitation-relevant features with parallel option
    print("Extracting features...")
    features_df, patient_ids, labels = extract_rehabilitation_features(epochs, use_parallel, n_jobs)

# If this file is run directly, show a message
if __name__ == "__main__":
    print("This module contains functions for EEG-based rehabilitation analysis.")
    print("Import and use these functions in your main script.")

