from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os
import scipy.stats as stats
from xgboost import XGBRegressor
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty

app = Flask(__name__)

# ==========================================
# 1. LOAD ASSETS (The Brains)
# ==========================================
def load_models():
    try:
        with open('gatekeeper.pkl', 'rb') as f:
            gk = pickle.load(f)
        spec = XGBRegressor()
        spec.load_model('specialist.json')
        with open('features.pkl', 'rb') as f:
            feats = pickle.load(f)
        return gk, spec, feats
    except Exception as e:
        print(f"CRITICAL ERROR LOADING MODELS: {e}")
        return None, None, None

gatekeeper, spec_gap, features_list = load_models()

# ==========================================
# 2. ROUTES
# ==========================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    formula = request.form.get('formula')
    if not formula:
        return jsonify({"error": "Please enter a chemical formula."})

    try:
        # --- A. FEATURIZATION ---
        temp_df = pd.DataFrame({'Formula': [formula]})
        stc = StrToComposition()
        temp_df = stc.featurize_dataframe(temp_df, "Formula", ignore_errors=True)
        
        ep = ElementProperty.from_preset(preset_name="magpie")
        feats = ep.featurize_dataframe(temp_df, col_id="composition")
        
        # --- B. FEATURE ALIGNMENT ---
        X_input = feats.drop(columns=['Formula', 'composition'])
        
        # Ensure all 132 columns exist
        for col in features_list:
            if col not in X_input.columns:
                X_input[col] = 0
        
        X_input = X_input[features_list]

        # --- C. BALANCED PREDICTION LOGIC ---
        is_insulator = gatekeeper.predict(X_input)[0]
        
        if is_insulator == 0:
            # Material is Metallic
            g = 0.0
            final_score = 25.0 # Baseline conductivity score
            stability_val = 0.4
            magnet_val = 0.2
        else:
            # Material is an Insulator/Semiconductor
            g = float(spec_gap.predict(X_input)[0])
            
            # 1. New Balanced Gap Score (Wider Window)
            # Center: 1.75eV, StdDev: 2.0 (Generous)
            s_gap = stats.norm.pdf(g, 1.75, 2.0) / stats.norm.pdf(1.75, 1.75, 2.0)
            
            # 2. Stability Bonus Logic
            # Higher band gaps generally correlate with thermodynamic stability
            stability_bonus = 0.8 if g > 3.0 else 0.5
            
            # 3. Final Weighted Score (70% Gap match, 30% Stability)
            final_score = round((s_gap * 0.7 + stability_bonus * 0.3) * 100, 1)
            
            stability_val = stability_bonus
            magnet_val = 0.5 # Placeholder for magnetization prediction

        # --- D. RETURN RESULTS ---
        return jsonify({
            "gap": round(g, 3),
            "score": final_score,
            # Data specifically formatted for Chart.js Radar Chart
            "radar": [
                min(g/6, 1),       # Band Gap Vertex
                stability_val,     # Stability Vertex
                magnet_val,        # Magnetization Vertex
                final_score/100    # Electronic Score Vertex
            ]
        })

    except Exception as e:
        print(f"Error during prediction of {formula}: {e}")
        return jsonify({"error": "Failed to analyze formula. Check chemical syntax."})

if __name__ == '__main__':
    app.run(debug=True, port=5000)