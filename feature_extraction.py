import scipy.io
import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
input_file = 'fault_data_python_ready.mat'
output_csv = 'final_model_training_data.csv'

# Check if file exists
if not os.path.exists(input_file):
    print(f"ERROR: Could not find '{input_file}'")
    print("Did you run the MATLAB conversion snippet in Step 1?")
    exit()

print(f"Loading {input_file}...")
mat_data = scipy.io.loadmat(input_file)

# The data is stored under the variable name we saved in MATLAB
data_struct = mat_data['data_for_python']

# Get the number of simulations (rows in the struct)
num_samples = data_struct.shape[1] 
print(f"Successfully loaded {num_samples} simulation runs.")

# --- FEATURE ENGINEERING FUNCTIONS ---

def calculate_rms(signal):
    """Calculates Root Mean Square of a signal"""
    # signal shape is (Time, 3) -> (Phase A, B, C)
    # We square it, take the mean, then square root
    return np.sqrt(np.mean(signal**2, axis=0))

def calculate_peak(signal):
    """Calculates the Maximum Absolute Value (Peak)"""
    return np.max(np.abs(signal), axis=0)

# --- MAIN LOOP ---

dataset_list = []

print("Extracting features (RMS, Peak) from signals...")

for i in range(num_samples):
    # Get the data for this specific run
    # Note: [0, i] is required because of how scipy loads struct arrays
    run_data = data_struct[0, i]
    
    # 1. Extract Meta Data
    run_id = run_data['RunID'][0][0]
    fault_type = str(run_data['FaultType'][0])
    label = run_data['Label'][0][0] # 0 to 11
    
    # 2. Extract Raw Signals (Matrices)
    # V_source and I_source are critical for detection
    v_source = run_data['V_source'] 
    i_source = run_data['I_source']
    
    # 3. Calculate Features
    # RMS Voltage (3 values: Va, Vb, Vc)
    v_rms = calculate_rms(v_source)
    # RMS Current (3 values: Ia, Ib, Ic)
    i_rms = calculate_rms(i_source)
    
    # Peak Voltage (3 values)
    v_peak = calculate_peak(v_source)
    # Peak Current (3 values)
    i_peak = calculate_peak(i_source)
    
    # 4. Organize into a dictionary (One row for our final CSV)
    row = {
        'Run_ID': run_id,
        'Label': label,
        'Fault_Type': fault_type,
        
        # Voltage Features
        'Va_RMS': v_rms[0], 'Vb_RMS': v_rms[1], 'Vc_RMS': v_rms[2],
        'Va_Peak': v_peak[0], 'Vb_Peak': v_peak[1], 'Vc_Peak': v_peak[2],
        
        # Current Features
        'Ia_RMS': i_rms[0], 'Ib_RMS': i_rms[1], 'Ic_RMS': i_rms[2],
        'Ia_Peak': i_peak[0], 'Ib_Peak': i_peak[1], 'Ic_Peak': i_peak[2]
    }
    
    dataset_list.append(row)

# --- SAVE TO CSV ---
df = pd.DataFrame(dataset_list)

# Reorder columns to put Target (Label) at the end or beginning
cols = ['Run_ID', 'Fault_Type', 'Label'] + \
       [c for c in df.columns if c not in ['Run_ID', 'Fault_Type', 'Label']]
df = df[cols]

print("-" * 30)
print("FEATURE EXTRACTION COMPLETE")
print("-" * 30)
print(df.head()) # Show first few rows
print("-" * 30)

df.to_csv(output_csv, index=False)
print(f"Success! Training data saved to: {output_csv}")
print("You are now ready to train your Machine Learning model.")