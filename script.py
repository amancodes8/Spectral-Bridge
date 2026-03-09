import pandas as pd
import numpy as np

def generate_verification_test(file_path="test.csv", num_samples=3):
    """
    Generates a test.csv file in the format expected by the 
    Spectral Bridge ANP notebook.
    """
    all_data = []
    
    # Starting with Sample IDs outside the training range (10000-89999)
    sample_ids = [90001, 90002, 90003] 
    
    for sid in sample_ids:
        # 1. Create a 100ms timeline at 1kHz (100 points)
        time_ms = np.linspace(1, 100, 100)
        
        # 2. Create a realistic "audio" voltage signal (sine mix + noise)
        # Your model expects values roughly between -0.5 and 1.7
        base_signal = 0.6 + 0.4 * np.sin(2 * np.pi * 0.04 * time_ms)
        harmonic = 0.1 * np.sin(2 * np.pi * 0.12 * time_ms)
        noise = 0.05 * np.random.randn(100)
        true_values = base_signal + harmonic + noise
        
        # 3. Randomly select context points (approx 20% like the real test data)
        is_context = np.zeros(100, dtype=int)
        context_indices = np.random.choice(100, size=20, replace=False)
        is_context[context_indices] = 1
        
        # 4. Create the DataFrame
        df = pd.DataFrame({
            'Sample_ID': sid,
            'Time_ms': time_ms,
            'Is_Context': is_context,
            'Value': true_values
        })
        
        # 5. MASK the target values (set to NaN where Is_Context is 0)
        # This is how the notebook identifies what to predict
        df.loc[df['Is_Context'] == 0, 'Value'] = np.nan
        
        all_data.append(df)

    final_df = pd.concat(all_data)
    
    # Save with empty strings for NaNs to match your "3 point" CSV observation
    final_df.to_csv(file_path, index=False, na_rep="")
    print(f"File '{file_path}' generated with {len(final_df)} rows.")

if __name__ == "__main__":
    generate_verification_test()