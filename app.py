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

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(data.head())

    # Data preprocessing
    data.dropna(inplace=True)

    # Label encoding for categorical variables
    label_encoders = {}
    for col in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Target selection
    target_col = st.selectbox("Select the Target Column", options=data.columns)
    X = data.drop(target_col, axis=1)
    y = data[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Calculate information gain
    base_tree = DecisionTreeClassifier(criterion='entropy', random_state=42)
    base_tree.fit(X_train, y_train)
    info_gain = base_tree.feature_importances_
    feature_df = pd.DataFrame({'Feature': X.columns, 'Information_Gain': info_gain})
    feature_df.sort_values(by='Information_Gain', ascending=False, inplace=True)

    st.subheader("Feature Importances (Information Gain)")
    st.write(feature_df)

    # Dynamic threshold range based on actual feature importances
    max_ig = feature_df['Information_Gain'].max()
    min_ig = feature_df['Information_Gain'].min()
    step_size = max(0.001, round((max_ig - min_ig)/20, 3))
    
    selected_threshold = st.slider(
        "Select Information Gain Threshold",
        min_value=0.0,
        max_value=float(max_ig * 1.1),
        value=float(max_ig * 0.1),
        step=step_size,
        format="%.4f"
    )
    
    selected_features = feature_df[feature_df["Information_Gain"] > selected_threshold]["Feature"].tolist()
    st.markdown(f"### Selected Features with Threshold {selected_threshold:.4f}:")
    st.write(selected_features)

    # Evaluate models at different thresholds
    results = []
    threshold_values = np.linspace(0.0, max_ig * 1.1, 20)
    
    for thresh in threshold_values:
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
                "Threshold": round(thresh, 6),
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec
            })

    results_df = pd.DataFrame(results)
    
    # Filter for current threshold with tolerance for floating point precision
    tolerance = 1e-6
    filtered_df = results_df[np.isclose(results_df["Threshold"], selected_threshold, atol=tolerance)]

    # Visualization improvements
    st.subheader("Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not filtered_df.empty:
            st.subheader(f"Model Comparison at Threshold = {selected_threshold:.4f}")
            melted_df = filtered_df.melt(id_vars=["Model"], value_vars=["Accuracy", "Precision", "Recall"],
                                         var_name="Metric", value_name="Score")
            
            # Calculate dynamic y-axis limits
            min_score = melted_df['Score'].min()
            max_score = melted_df['Score'].max()
            y_lower = max(0, min_score - 0.1)
            y_upper = min(1, max_score + 0.1)

            fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
            metrics = melted_df['Metric'].unique()
            models = melted_df['Model'].unique()
            bar_width = 0.25
            x = np.arange(len(models))

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            for i, metric in enumerate(metrics):
                scores = melted_df[melted_df['Metric'] == metric]['Score']
                ax_bar.bar(x + i * bar_width, scores, width=bar_width, label=metric, color=colors[i])

            ax_bar.set_xticks(x + bar_width)
            ax_bar.set_xticklabels(models, rotation=45)
            ax_bar.set_ylim(y_lower, y_upper)
            ax_bar.set_ylabel("Score")
            ax_bar.set_title(f"Performance at Threshold = {selected_threshold:.4f}")
            ax_bar.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax_bar.grid(True, alpha=0.3)

            # Add value labels
            for i, metric in enumerate(metrics):
                scores = melted_df[melted_df['Metric'] == metric]['Score']
                for j, score in enumerate(scores):
                    ax_bar.text(x[j] + i * bar_width, score + 0.01, f"{score:.2f}", 
                               ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            st.pyplot(fig_bar)
        else:
            st.warning("No features selected for this threshold. Please choose a lower threshold.")

    with col2:
        st.subheader("Accuracy vs. Threshold")
        fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
        
        # Calculate dynamic y-axis limits for accuracy plot
        acc_min = results_df['Accuracy'].min()
        acc_max = results_df['Accuracy'].max()
        acc_y_lower = max(0, acc_min - 0.1)
        acc_y_upper = min(1, acc_max + 0.1)
        
        for model in results_df['Model'].unique():
            model_data = results_df[results_df['Model'] == model]
            ax_acc.plot(model_data['Threshold'], model_data['Accuracy'], 
                      marker='o', linestyle='-', label=model, linewidth=2)
        
        ax_acc.axvline(x=selected_threshold, color='r', linestyle='--', alpha=0.5, label='Current Threshold')
        ax_acc.set_title("Accuracy vs. Information Gain Threshold")
        ax_acc.set_xlabel("Information Gain Threshold")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(acc_y_lower, acc_y_upper)
        ax_acc.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_acc.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_acc)

    # Performance trends plot
    st.subheader("Performance Trends")
    fig_trends, ax_trends = plt.subplots(figsize=(12, 6))
    
    # Calculate dynamic y-axis limits for trends plot
    trend_min = min(results_df['Accuracy'].min(), results_df['Precision'].min(), results_df['Recall'].min())
    trend_max = max(results_df['Accuracy'].max(), results_df['Precision'].max(), results_df['Recall'].max())
    trend_y_lower = max(0, trend_min - 0.1)
    trend_y_upper = min(1, trend_max + 0.1)
    
    line_styles = ['-', '--', ':']
    for i, metric in enumerate(['Accuracy', 'Precision', 'Recall']):
        for j, model in enumerate(results_df['Model'].unique()):
            subset = results_df[results_df['Model'] == model]
            ax_trends.plot(subset['Threshold'], subset[metric], 
                          linestyle=line_styles[i], marker='o', markersize=4,
                          label=f"{model} - {metric}", linewidth=1.5)
    
    ax_trends.axvline(x=selected_threshold, color='k', linestyle='--', alpha=0.5, label='Current Threshold')
    ax_trends.set_title("Model Performance vs. Information Gain Threshold")
    ax_trends.set_xlabel("Information Gain Threshold")
    ax_trends.set_ylabel("Score")
    ax_trends.set_ylim(trend_y_lower, trend_y_upper)
    ax_trends.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_trends.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_trends)

    # Best models summary
    st.subheader("Best Model by Metric")
    cols = st.columns(3)
    for i, metric in enumerate(["Accuracy", "Precision", "Recall"]):
        with cols[i]:
            best_idx = results_df[metric].idxmax()
            st.metric(
                label=f"Best {metric}",
                value=f"{results_df.loc[best_idx, metric]:.3f}",
                help=f"Model: {results_df.loc[best_idx, 'Model']}\nThreshold: {results_df.loc[best_idx, 'Threshold']:.4f}"
            )

    # Full results table
    st.subheader("Detailed Performance Results")
    st.dataframe(results_df.style.format({
        'Threshold': '{:.4f}',
        'Accuracy': '{:.3f}',
        'Precision': '{:.3f}',
        'Recall': '{:.3f}'
    }))
