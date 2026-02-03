import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# PROTECTION FOR WINDOWS
if __name__ == '__main__':
    # 1. Setup Paths
    BASE_PATH = r'C:\Users\Mani Bharathi\Desktop\MQUBIT'
    DATA_PATH = os.path.join(BASE_PATH, 'quantum_features_ready.csv')

    print("⚡ Starting SAFE Training Engine...")
    df = pd.read_csv(DATA_PATH)
    df['is_insulator'] = (df['Band_Gap'] > 0.001).astype(int)

    # 2. Prepare Features
    X = df.drop(columns=['Band_Gap', 'Stability', 'Magnetization', 'is_insulator'])
    feature_names = X.columns.tolist()

    # 3. Model A: Gatekeeper (Classification)
    # LIMIT: n_jobs=2 (Leaves 8 cores for Windows), n_estimators=50 (Faster)
    print("🛰️ Training Gatekeeper (Safety Mode)...")
    gatekeeper = RandomForestClassifier(n_estimators=50, n_jobs=2, random_state=42)
    gatekeeper.fit(X, df['is_insulator'])

    # 4. Model B: Specialist (Regression)
    # Filter for insulators to make the specialist faster
    ins_df = df[df['is_insulator'] == 1]
    X_ins = ins_df.drop(columns=['Band_Gap', 'Stability', 'Magnetization', 'is_insulator'])
    
    print("🎯 Training Specialist (Safety Mode)...")
    specialist = XGBRegressor(n_estimators=100, n_jobs=2, learning_rate=0.1)
    specialist.fit(X_ins, ins_df['Band_Gap'])

    # 5. Save the "Brains"
    print("💾 Saving models to disk...")
    with open(os.path.join(BASE_PATH, 'gatekeeper.pkl'), 'wb') as f:
        pickle.dump(gatekeeper, f)
    
    specialist.save_model(os.path.join(BASE_PATH, 'specialist.json'))
    
    with open(os.path.join(BASE_PATH, 'features.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)

    print("✅ Training Complete! Your PC survived. You can now close this and run app.py.")