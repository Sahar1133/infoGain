# Import necessary libraries
import streamlit as st  # For creating the web app interface
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualization

# Import scikit-learn modules for machine learning
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import LabelEncoder  # For encoding categorical variables
from sklearn.tree import DecisionTreeClassifier  # Decision Tree model
from sklearn.neighbors import KNeighborsClassifier  # KNN model
from sklearn.naive_bayes import GaussianNB  # Naive Bayes model
from sklearn.metrics import accuracy_score, precision_score, recall_score  # Evaluation metrics

# Set up the Streamlit app title
st.title("Feature Selection using Information Gain with Multiple Classifiers")

# Create file uploader widget
uploaded_file = st.file_uploader("Upload dataset (.xlsx or .csv)", type=["xlsx", "csv"])

# Initialize empty DataFrame to store results
results_df = pd.DataFrame(columns=["Model", "Threshold", "Accuracy", "Precision", "Recall"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the file based on its extension
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    # Display dataset preview
    st.subheader("Dataset Preview")
    st.write(data.head())

    # Clean data by dropping rows with missing values
    data.dropna(inplace=True)

    # Encode categorical variables using LabelEncoder
    label_encoders = {}
    for col in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Let user select target column
    target_col = st.selectbox("Select the Target Column", options=data.columns)
    
    # Split data into features (X) and target (y)
    X = data.drop(target_col, axis=1)
    y = data[target_col]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train base Decision Tree to calculate information gain
    base_tree = DecisionTreeClassifier(criterion='entropy', random_state=42)
    base_tree.fit(X_train, y_train)

    # Get feature importances (information gain)
    info_gain = base_tree.feature_importances_
    
    # Create DataFrame to display feature importances
    feature_df = pd.DataFrame({'Feature': X.columns, 'Information_Gain': info_gain})
    feature_df.sort_values(by='Information_Gain', ascending=False, inplace=True)

    # Display feature importances
    st.subheader("Feature Importances (Information Gain)")
    st.write(feature_df)
    
    # Create slider for selecting information gain threshold
    selected_threshold = st.slider(
        "Select Threshold",
        min_value=0.0,
        max_value=0.05,  
        value=0.025,
        step=0.001
    )
    
    # Get selected features based on threshold
    selected_features = feature_df[feature_df["Information_Gain"] > selected_threshold]["Feature"].tolist()
    st.markdown(f"### Selected Features with Threshold {selected_threshold}:")
    st.write(selected_features)

    # Initialize list to store results
    results = []  

    # Test different threshold values
    for thresh in np.arange(0.0, 0.051, 0.001): 
        selected = feature_df[feature_df["Information_Gain"] > thresh]["Feature"].tolist()
        if not selected:
            continue
            
        # Select features based on current threshold
        X_train_sel = X_train[selected]
        X_test_sel = X_test[selected]

        # Define models to test
        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB()
        }

        # Train and evaluate each model
        for model_name, model in models.items():
            model.fit(X_train_sel, y_train)
            y_pred = model.predict(X_test_sel)

            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)

            # Store results
            results.append({
                "Model": model_name,
                "Threshold": round(thresh, 3),
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec
            })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Filter results for the selected threshold
    filtered_df = results_df[results_df["Threshold"] == round(selected_threshold, 3)]

    # Display model comparison at selected threshold
    if not filtered_df.empty:
        st.subheader(f"Model Comparison at Threshold = {selected_threshold}")
        
        # Prepare data for visualization
        melted_df = filtered_df.melt(id_vars=["Model"], 
                                   value_vars=["Accuracy", "Precision", "Recall"],
                                   var_name="Metric", 
                                   value_name="Score")
        
        # Create bar plot
        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        metrics = melted_df['Metric'].unique()
        models = melted_df['Model'].unique()
        bar_width = 0.25
        x = np.arange(len(models))

        # Set dynamic y-axis limits
        min_score = melted_df['Score'].min()
        max_score = melted_df['Score'].max()
        y_lower = max(0, min_score - 0.1) if min_score > 0.1 else 0
        y_upper = min(1, max_score + 0.1) if max_score < 0.9 else 1.1

        # Plot bars for each metric
        for i, metric in enumerate(metrics):
            scores = melted_df[melted_df['Metric'] == metric]['Score']
            ax_bar.bar(x + i * bar_width, scores, width=bar_width, label=metric)

        # Configure plot
        ax_bar.set_xticks(x + bar_width)
        ax_bar.set_xticklabels(models)
        ax_bar.set_ylim(y_lower, y_upper)
        ax_bar.set_ylabel("Score")
        ax_bar.set_title("Accuracy, Precision & Recall Comparison by Model")
        ax_bar.legend()
        ax_bar.grid(True)
        st.pyplot(fig_bar)
    else:
        st.warning("No features selected for this threshold. Please choose a lower threshold.")

    # Display performance summary
    st.subheader("Performance Summary")
    st.dataframe(results_df)

    # Create line plot showing performance vs. threshold
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot each metric for each model
    for metric in ['Accuracy', 'Precision', 'Recall']:
        for model in results_df['Model'].unique():
            subset = results_df[results_df['Model'] == model]
            ax.plot(subset['Threshold'], subset[metric], marker='o', markersize=8, label=f"{model} - {metric}")
    
    # Set dynamic y-axis limits
    min_score = results_df[['Accuracy', 'Precision', 'Recall']].min().min()
    max_score = results_df[['Accuracy', 'Precision', 'Recall']].max().max()
    y_lower = max(0, min_score - 0.1) if min_score > 0.1 else 0
    y_upper = min(1, max_score + 0.1) if max_score < 0.9 else 1.1
    
    # Configure plot
    ax.set_title("Model Performance vs. Information Gain Threshold", pad=20)
    ax.set_xlabel("Information Gain Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(y_lower, y_upper)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)

    # Display best models for each metric
    st.subheader("Best Model by Metric")
    for metric in ["Accuracy", "Precision", "Recall"]:
        idx = results_df[metric].idxmax()
        st.markdown(f"**{metric}:** Best Model: `{results_df.loc[idx, 'Model']}`, "
                    f"Threshold: `{results_df.loc[idx, 'Threshold']}`, "
                    f"Score: `{results_df.loc[idx, metric]:.2f}`")
    import matplotlib.pyplot as plt

# Create figure
plt.figure(figsize=(12, 6))

# Extract best scores for each metric
best_accuracy = results_df.loc[results_df['Accuracy'].idxmax()]
best_precision = results_df.loc[results_df['Precision'].idxmax()]
best_recall = results_df.loc[results_df['Recall'].idxmax()]

# Create bar positions
x = np.arange(3)  # For 3 metrics
width = 0.25  # Width of bars

# Plot bars for each model's best scores
plt.bar(x - width, 
        [best_accuracy['Accuracy'], best_precision['Precision'], best_recall['Recall']],
        width, 
        label='Decision Tree',
        color='#1f77b4')

# Add threshold text on each bar
for i, (val, thresh) in enumerate(zip(
    [best_accuracy['Accuracy'], best_precision['Precision'], best_recall['Recall']],
    [best_accuracy['Threshold'], best_precision['Threshold'], best_recall['Threshold']]
)):
    plt.text(x[i] - width, val + 0.01, 
             f"Thresh: {thresh:.3f}", 
             ha='center')

# Customize plot
plt.xticks(x, ['Accuracy', 'Precision', 'Recall'])
plt.ylabel('Score')
plt.title('Best Performance by Metric (Decision Tree)')
plt.ylim(0, 1.1)
plt.grid(True, axis='y', alpha=0.3)

# Add horizontal line at 0.5 for reference
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
st.pyplot(plt.gcf())

    # Create separate plot for accuracy vs. threshold
    st.subheader("Accuracy vs. Information Gain Threshold")
    fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
    
    # Plot accuracy for each model
    for model in results_df['Model'].unique():
        model_data = results_df[results_df['Model'] == model]
        ax_acc.plot(model_data['Threshold'], model_data['Accuracy'], marker='o', markersize=8, label=model)
    
    # Set dynamic y-axis limits
    min_acc = results_df['Accuracy'].min()
    max_acc = results_df['Accuracy'].max()
    y_lower_acc = max(0, min_acc - 0.1) if min_acc > 0.1 else 0
    y_upper_acc = min(1, max_acc + 0.1) if max_acc < 0.9 else 1.1
    
    # Configure plot
    ax_acc.set_title("Accuracy vs. Information Gain Threshold", pad=20)
    ax_acc.set_xlabel("Information Gain Threshold")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(y_lower_acc, y_upper_acc)
    ax_acc.legend()
    ax_acc.grid(True)
    st.pyplot(fig_acc)
