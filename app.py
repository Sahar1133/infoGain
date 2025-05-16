import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import random
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Career Path Predictor with IG Feature Selection", layout="centered")

st.title("Career Path Predictor with Feature Selection via Information Gain")

# Step 1: Upload Dataset
uploaded_file = st.file_uploader("Upload your dataset CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(data.head())
    st.write(f"Shape: {data.shape}")
    st.write("Data Types:")
    st.write(data.dtypes)

    # Step 2: Select target column
    target_col = st.selectbox("Select the Target Column", options=data.columns)

    if target_col:
        # Separate features and target
        X = data.drop(columns=[target_col])
        y = data[target_col]

        # Encode target column
        target_encoder = LabelEncoder()
        y_enc = target_encoder.fit_transform(y.astype(str))

        # Encode categorical features for IG
        X_encoded = X.copy()
        encoders = {}
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object' or str(X_encoded[col].dtype).startswith('category'):
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                encoders[col] = le
            else:
                # Keep numeric as is
                try:
                    X_encoded[col] = X_encoded[col].astype(float)
                except:
                    X_encoded[col] = 0

        # Step 3: Compute Information Gain (Mutual Information)
        mi_scores = mutual_info_classif(X_encoded, y_enc, discrete_features='auto', random_state=42)
        mi_df = pd.DataFrame({"Feature": X_encoded.columns, "Information Gain": mi_scores})
        mi_df = mi_df.sort_values(by="Information Gain", ascending=False).reset_index(drop=True)

        st.subheader("Information Gain of Features")
        st.dataframe(mi_df.style.format({"Information Gain": "{:.4f}"}))

        # Step 4: Threshold slider
        max_ig = mi_df["Information Gain"].max()
        threshold = st.slider("Select Information Gain Threshold", 0.0, float(max_ig), 0.01, 0.005)

        # Select features above threshold
        selected_features = mi_df[mi_df["Information Gain"] >= threshold]["Feature"].tolist()

        st.write(f"Features selected (threshold >= {threshold}): {len(selected_features)}")
        st.write(selected_features)

        # Step 5: Load question bank
        with open("questions.json") as f:
            questions = json.load(f)

        # Step 6: Ask questions only for selected features present in question bank
        st.subheader("Answer Questions")

        responses = {}
        for feature in selected_features:
            if feature in questions:
                q = random.choice(questions[feature])
                responses[feature] = st.radio(
                    label=f"**{feature.replace('_', ' ').title()}**: {q['question']}",
                    options=q["options"],
                    key=feature
                )
            else:
                # If no question for feature, allow text input or skip
                responses[feature] = st.text_input(f"Enter value for {feature.replace('_', ' ').title()}:", key=feature)

        # Step 7: Load model
        with open("model.pkl", "rb") as f:
            model, model_encoders, model_target_encoder = pickle.load(f)

        if st.button("Predict Career Field"):
            # Check all responses provided
            if len(responses) < len(selected_features):
                st.warning("Please answer all questions.")
            else:
                try:
                    input_vector = []
                    for f in selected_features:
                        val = responses[f]
                        # Use model encoders to transform if available
                        if f in model_encoders:
                            val = model_encoders[f].transform([val])[0]
                        else:
                            # Try to convert to float else default 0
                            try:
                                val = float(val)
                            except:
                                val = 0
                        input_vector.append(val)

                    pred = model.predict([input_vector])[0]
                    career_pred = model_target_encoder.inverse_transform([pred])[0]

                    st.success(f"### Predicted Career Field: **{career_pred}**")
                    st.balloons()
                except Exception as e:
                    st.error(f"Prediction error: {e}")

else:
    st.info("Please upload a CSV dataset to begin.")
