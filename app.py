import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

st.title("Feature Selection using Information Gain with Multiple Classifiers")

uploaded_file = st.file_uploader("Upload dataset (.xlsx or .csv)", type=["xlsx", "csv"])

results_df = pd.DataFrame(columns=["Model", "Threshold", "Accuracy", "Precision", "Recall"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(data.head())

    data.dropna(inplace=True)

    label_encoders = {}
    for col in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    target_col = st.selectbox("Select the Target Column", options=data.columns)
    X = data.drop(target_col, axis=1)
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    base_tree = DecisionTreeClassifier(criterion='entropy', random_state=42)
    base_tree.fit(X_train, y_train)

    info_gain = base_tree.feature_importances_
    feature_df = pd.DataFrame({'Feature': X.columns, 'Information_Gain': info_gain})
    feature_df.sort_values(by='Information_Gain', ascending=False, inplace=True)

    st.subheader("Feature Importances (Information Gain)")
    st.write(feature_df)

    selected_threshold = st.slider("Select Information Gain Threshold", min_value=0.0, max_value=0.05, value=0.01, step=0.01)
    selected_features = feature_df[feature_df["Information_Gain"] > selected_threshold]["Feature"].tolist()
    st.markdown(f"### Selected Features with Threshold {selected_threshold}:")
    st.write(selected_features)

    results = []

    for thresh in np.arange(0.0, 0.21, 0.01):
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

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)

            results.append({
                "Model": model_name,
                "Threshold": round(thresh, 3),
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec
            })

    results_df = pd.DataFrame(results)
    filtered_df = results_df[results_df["Threshold"] == round(selected_threshold, 3)]

    if not filtered_df.empty:
        st.subheader(f"Model Comparison at Threshold = {selected_threshold}")
        melted_df = filtered_df.melt(id_vars=["Model"], value_vars=["Accuracy", "Precision", "Recall"],
                                     var_name="Metric", value_name="Score")
        fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
        metrics = melted_df['Metric'].unique()
        models = melted_df['Model'].unique()
        bar_width = 0.2
        x = np.arange(len(models))

        for i, metric in enumerate(metrics):
            scores = melted_df[melted_df['Metric'] == metric]['Score']
            ax_bar.bar(x + i * bar_width, scores, width=bar_width, label=metric)

        ax_bar.set_xticks(x + bar_width)
        ax_bar.set_xticklabels(models)
        ax_bar.set_ylim(0, 1.1)
        ax_bar.set_ylabel("Score")
        ax_bar.set_title("Accuracy, Precision & Recall Comparison by Model")
        ax_bar.legend()
        ax_bar.grid(True)
        st.pyplot(fig_bar)
    else:
        st.warning("No features selected for this threshold. Please choose a lower threshold.")

    st.subheader("Performance Summary")
    st.dataframe(results_df)

    fig, ax = plt.subplots(figsize=(12, 6))
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

    st.subheader("Best Model by Metric")
    for metric in ["Accuracy", "Precision", "Recall"]:
        idx = results_df[metric].idxmax()
        st.markdown(f"**{metric}:** Best Model: `{results_df.loc[idx, 'Model']}`, "
                    f"Threshold: `{results_df.loc[idx, 'Threshold']}`, "
                    f"Score: `{results_df.loc[idx, metric]:.2f}`")

    st.subheader("Accuracy vs. Information Gain Threshold")
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
                                        
