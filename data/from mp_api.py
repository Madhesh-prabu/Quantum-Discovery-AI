from mp_api.client import MPRester
import pandas as pd
from tqdm import tqdm

API_KEY = "p2v9t7cSAeiIflZfnKEptxZx3TvA52mW"

def download_quantum_data():
    # Use the 'with' statement to ensure the connection closes properly
    with MPRester(API_KEY) as mpr:
        # 1. We use materials.summary.search (the new way)
        # 2. Changed 'crystal_system' to 'symmetry' in the fields list
        docs = mpr.materials.summary.search(
            energy_above_hull=(0, 0.1),
            fields=[
                "material_id", 
                "formula_pretty", 
                "band_gap", 
                "total_magnetization", 
                "energy_above_hull", 
                "symmetry" # Updated field name
            ]
        )
        
        data = []
        for doc in tqdm(docs, desc="Extracting Quantum Data"):
            # doc.symmetry is now an object, so we grab the crystal_system attribute from it
            data.append({
                "ID": doc.material_id,
                "Formula": doc.formula_pretty,
                "Band_Gap": doc.band_gap,
                "Magnetization": doc.total_magnetization,
                "Stability": doc.energy_above_hull,
                "Symmetry": str(doc.symmetry.crystal_system) # Extracting the specific string
            })
            
        return pd.DataFrame(data)

# Run the engine
df = download_quantum_data()
df.to_csv("quantum_gold_mine.csv", index=False)
print(f"Success! Data saved with {len(df)} rows.")
