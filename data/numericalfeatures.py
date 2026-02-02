import pandas as pd
import os
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from tqdm import tqdm
import time

def run_safe_featurization_v2():
    input_path = r"C:\Users\Mani Bharathi\Desktop\MQUBIT\quantum_gold_mine.csv"
    output_path = r"C:\Users\Mani Bharathi\Desktop\MQUBIT\quantum_features_ready.csv"
    
    # 1. Load the full database
    df = pd.read_csv(input_path)
    
    # 2. SELECT THE NEXT BATCH (Skip first 18000, take next 28000)
    # This ensures no duplicates!
    print("Selecting materials from index 18000 to 28000...")
    df_sample = df.iloc[18000:28000].copy()

    # Step 1: Conversion
    stc = StrToComposition()
    df_sample = stc.featurize_dataframe(df_sample, "Formula") 

    # Step 2: Batching with "Safety Saves"
    ep = ElementProperty.from_preset(preset_name="magpie")
    batch_size = 100
    chunks = [df_sample[i:i + batch_size] for i in range(0, df_sample.shape[0], batch_size)]
    
    full_results = []
    print(f"Starting Phase 2 (Next 10,000 materials)...")

    for i, chunk in enumerate(tqdm(chunks, desc="Processing Batches")):
        feathered_chunk = ep.featurize_dataframe(chunk, col_id="composition")
        full_results.append(feathered_chunk)
        
        # Save progress every 500 materials
        if (i + 1) % 5 == 0:
            temp_df = pd.concat(full_results)
            temp_df.to_csv(output_path + "_batch2.tmp", index=False)
            
        time.sleep(0.1)

    # Final Save and Combine
    df_new_batch = pd.concat(full_results)
    
    # Clean up columns
    cols_to_drop = ["ID", "Formula", "composition"]
    df_new_ml = df_new_batch.drop(columns=cols_to_drop)
    df_new_ml = pd.get_dummies(df_new_ml, columns=["Symmetry"])
    
    # --- THE MAGIC STEP: COMBINING ---
    if os.path.exists(output_path):
        print("Existing data found. Merging current 8,000 with new 10,000...")
        df_old = pd.read_csv(output_path)
        # Combine old and new, ensuring columns align correctly
        df_combined = pd.concat([df_old, df_new_ml], ignore_index=True, sort=False).fillna(0)
        df_combined.to_csv(output_path, index=False)
    else:
        df_new_ml.to_csv(output_path, index=False)
        
    print(f"\nPhase 2 Batch 3 Complete! You now have {len(df_combined)} materials in your dataset.")

if __name__ == "__main__":
    run_safe_featurization_v2()