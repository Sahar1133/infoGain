import pandas as pd
import numpy as np
import streamlit as st

# Attempt to import openpyxl, which is required for reading .xlsx files
try:
    import openpyxl
except ImportError:
    st.error("The 'openpyxl' library is required to read Excel files. Please install it by running `pip install openpyxl`.")
    st.stop()  # Stop further execution of the app if openpyxl is missing
# Streamlit Title
st.title("Feature Selection using Information Gain with Multiple Classifiers")

# File upload
uploaded_file = st.file_uploader("Upload Excel dataset (.xlsx)", type=["xlsx"])
results_df = pd.DataFrame(columns=["Model", "Threshold", "Accuracy", "Precision", "Recall"])

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(data.head())

    # Clean dataset by removing missing values
    data.dropna(inplace=True)

    # Label encoding for categorical variables
    label_encoders = {}
    for col in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # User selects the target column
    target_col = st.selectbox("Select the Target Column", options=data.columns)
    X = data.drop(target_col, axis=1)
    y = data[target_col]

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a DecisionTreeClassifier to calculate Information Gain
    base_tree = DecisionTreeClassifier(criterion='entropy', random_state=42)
    base_tree.fit(X_train, y_train)
    info_gain = base_tree.feature_importances_

    # Create DataFrame for feature importances
    feature_df = pd.DataFrame({'Feature': X.columns, 'Information_Gain': info_gain})
    feature_df.sort_values(by='Information_Gain', ascending=False, inplace=True)

    st.subheader("Feature Importances (Information Gain)")
    st.write(feature_df)

    # Threshold slider for Information Gain selection
    selected_threshold = st.slider("Select Information Gain Threshold", min_value=0.0, max_value=0.04, value=0.01, step=0.01)
    selected_features = feature_df[feature_df["Information_Gain"] > selected_threshold]["Feature"].tolist()

    st.markdown(f"### Selected Features with Threshold {selected_threshold}:")
    st.write(selected_features)

    results = []

    # Loop through different thresholds and evaluate models
    for thresh in np.arange(0.0, 0.041, 0.01):
        selected = feature_df[feature_df["Information_Gain"] > thresh]["Feature"].tolist()
        if not selected:
            continue

        X_train_sel = X_train[selected]
        X_test_sel = X_test[selected]

        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB()
        }

        for model_name, model in models.items():
            model.fit(X_train_sel, y_train)
            y_pred = model.predict(X_test_sel)

            # Calculate accuracy, precision, recall
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro')
            rec = recall_score(y_test, y_pred, average='macro')

            # Append results
            results.append({
                "Model": model_name,
                "Threshold": thresh,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec
            })

    results_df = pd.DataFrame(results)

    # Filter results based on the selected threshold
    filtered_df = results_df[results_df["Threshold"] == selected_threshold] if 'selected_threshold' in locals() else \
                  pd.DataFrame(columns=["Model", "Threshold", "Accuracy", "Precision", "Recall"])

    if not filtered_df.empty:
        st.subheader(f"Model Comparison at Threshold = {selected_threshold}")

        # Melt DataFrame for plotting
        melted_df = filtered_df.melt(id_vars=["Model"],
                                     value_vars=["Accuracy", "Precision", "Recall"],
                                     var_name="Metric",
                                     value_name="Score")

        fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
        metrics = melted_df['Metric'].unique()
        models = melted_df['Model'].unique()
        bar_width = 0.2
        x = np.arange(len(models))

        max_score = melted_df["Score"].max()
        y_limit = max(1.0, round(max_score + 0.05, 2))

        # Create bar plot
        for i, metric in enumerate(metrics):
            scores = melted_df[melted_df['Metric'] == metric]['Score'].tolist()
            ax_bar.bar(x + i * bar_width, scores, width=bar_width, label=metric)

        ax_bar.set_xticks(x + bar_width)
        ax_bar.set_xticklabels(models)
        ax_bar.set_ylim(0, y_limit)
        ax_bar.set_ylabel("Score")
        ax_bar.set_title("Accuracy, Precision & Recall Comparison by Model")
        ax_bar.legend()
        ax_bar.grid(True)
        st.pyplot(fig_bar)
    else:
        st.warning("No features selected for this threshold. Please choose a lower threshold.")

    # Performance summary
    st.subheader("Performance Summary")
    st.dataframe(results_df)

    # Plot performance vs. threshold
    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in ['Accuracy', 'Precision', 'Recall']:
        for model in results_df['Model'].unique():
            subset = results_df[results_df['Model'] == model]
            ax.plot(subset['Threshold'], subset[metric], marker='o', label=f"{model} - {metric}")

    ax.set_title("Model Performance vs. Information Gain Threshold")
    ax.set_xlabel("Information Gain Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)

    # Best Model by Metric
    st.subheader("Best Model by Metric")
    for metric in ["Accuracy", "Precision", "Recall"]:
        idx = results_df[metric].idxmax()
        st.markdown(f"*{metric}:* Best Model: {results_df.loc[idx, 'Model']}, "
                    f"Threshold: {results_df.loc[idx, 'Threshold']}, "
                    f"Score: {results_df.loc[idx, metric]:.2f}")

    # Visualization Pilot
    st.subheader("Visualization Pilot: Accuracy vs. Information Gain Threshold")
    if not results_df.empty:
        fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
        for model in results_df['Model'].unique():
            model_data = results_df[results_df['Model'] == model]
            ax_acc.plot(model_data['Threshold'], model_data['Accuracy'], marker='o', label=model)

        ax_acc.set_title("Accuracy vs. Information Gain Threshold")
        ax_acc.set_xlabel("Information Gain Threshold")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0, 1.1)
        ax_acc.legend()
        ax_acc.grid(True)
        st.pyplot(fig_acc)
    else:
        st.write("Please upload a dataset to generate the visualization.")
