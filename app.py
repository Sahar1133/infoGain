\[media pointer="file-service://file-9Uk6Tc1LvKVjMAqz9SjZGJ"]

# Install Streamlit (if not already installed) in your environment

!pip install streamlit

# Importing required libraries

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Machine learning and preprocessing libraries

from sklearn.model\_selection import train\_test\_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive\_bayes import GaussianNB
from sklearn.metrics import accuracy\_score, precision\_score, recall\_score

# Title of the Streamlit Web Application

st.title("Feature Selection using Information Gain with Multiple Classifiers")

# Upload Excel file containing the dataset

uploaded\_file = st.file\_uploader("Upload Excel dataset (.xlsx)", type=\["xlsx"])

# Initialize an empty DataFrame to store model results

results\_df = pd.DataFrame(columns=\["Model", "Threshold", "Accuracy", "Precision", "Recall"])

# MAIN PROCESS STARTS HERE

if uploaded\_file is not None:

# Read the uploaded Excel file

data = pd.read\_excel(uploaded\_file)

# Show preview of the dataset

st.subheader("Dataset Preview")
st.write(data.head())

# Drop rows with any missing values to ensure clean training data

data.dropna(inplace=True)

# Label Encode categorical features

label\_encoders = {}
for col in data.select\_dtypes(include='object').columns:
le = LabelEncoder()
data\[col] = le.fit\_transform(data\[col])
label\_encoders\[col] = le # Store encoder for potential inverse transform

# Select Target Column

target\_col = st.selectbox("Select the Target Column", options=data.columns)
X = data.drop(target\_col, axis=1) # Features
y = data\[target\_col] # Target label

# Split data into train and test sets

X\_train, X\_test, y\_train, y\_test = train\_test\_split(X, y, test\_size=0.3, random\_state=42)

# Calculate Information Gain using Decision Tree

base\_tree = DecisionTreeClassifier(criterion='entropy', random\_state=42)
base\_tree.fit(X\_train, y\_train)

# Get feature importances (Information Gain)

info\_gain = base\_tree.feature\_importances\_

# Create DataFrame of features and their information gain

feature\_df = pd.DataFrame({'Feature': X.columns, 'Information\_Gain': info\_gain})
feature\_df.sort\_values(by='Information\_Gain', ascending=False, inplace=True)

# Display feature importance table

st.subheader("Feature Importances (Information Gain)")
st.write(feature\_df)

# Allow user to select Information Gain threshold

selected\_threshold = st.slider("Select Information Gain Threshold", min\_value=0.0, max\_value=0.2,
value=0.01, step=0.01)

# Filter features based on selected threshold

selected\_features = feature\_df\[feature\_df\["Information\_Gain"] > selected\_threshold]\["Feature"].tolist()
st.markdown(f"### Selected Features with Threshold {selected\_threshold}:")
st.write(selected\_features)

# Initialize results list

results = \[]

# Iterate through thresholds and train classifiers

for thresh in np.arange(0.0, 0.21, 0.01): # From 0.0 to 0.2 in 0.01 steps
selected = feature\_df\[feature\_df\["Information\_Gain"] > thresh]\["Feature"].tolist()
if not selected: # Skip if no features meet the threshold
continue

# Subset train and test data based on selected features

X\_train\_sel = X\_train\[selected]
X\_test\_sel = X\_test\[selected]

# Initialize three models

models = {
"Decision Tree": DecisionTreeClassifier(random\_state=42),
"KNN": KNeighborsClassifier(n\_neighbors=5),
"Naive Bayes": GaussianNB()
}

# Train and evaluate each model

for model\_name, model in models.items():
model.fit(X\_train\_sel, y\_train)
y\_pred = model.predict(X\_test\_sel)

# Calculate performance metrics

acc = accuracy\_score(y\_test, y\_pred)
prec = precision\_score(y\_test, y\_pred, average='macro')
rec = recall\_score(y\_test, y\_pred, average='macro')

# Append result for current model and threshold

results.append({
"Model": model\_name,
"Threshold": thresh,
"Accuracy": acc,
"Precision": prec,
"Recall": rec
})

# Convert list of results to DataFrame

results\_df = pd.DataFrame(results)

# Filter and visualize model results for current threshold

filtered\_df = results\_df\[results\_df\["Threshold"] == selected\_threshold] if 'selected\_threshold' in locals() else
pd.DataFrame(columns=\["Model", "Threshold", "Accuracy", "Precision", "Recall"])
if not filtered\_df.empty:
st.subheader(f"Model Comparison at Threshold = {selected\_threshold}")

# Prepare data for grouped bar plot

melted\_df = filtered\_df.melt(id\_vars=\["Model"],
value\_vars=\["Accuracy", "Precision", "Recall"],
var\_name="Metric",
value\_name="Score")

# Plot bar chart

fig\_bar, ax\_bar = plt.subplots(figsize=(8, 5))
metrics = melted\_df\['Metric'].unique()
models = melted\_df\['Model'].unique()
bar\_width = 0.2
x = np.arange(len(models))
for i, metric in enumerate(metrics):
scores = melted\_df\[melted\_df\['Metric'] == metric]\['Score']
ax\_bar.bar(x + i \* bar\_width, scores, width=bar\_width, label=metric)
ax\_bar.set\_xticks(x + bar\_width)
ax\_bar.set\_xticklabels(models)
ax\_bar.set\_ylim(0, 1.1)
ax\_bar.set\_ylabel("Score")
ax\_bar.set\_title("Accuracy, Precision & Recall Comparison by Model")
ax\_bar.legend()
ax\_bar.grid(True)
st.pyplot(fig\_bar)
else:
st.warning("No features selected for this threshold. Please choose a lower threshold.")

# Full Results Summary and Overall Model Performance

if uploaded\_file is not None:
st.subheader("Performance Summary")
st.dataframe(results\_df)

# Line plot for Accuracy/Precision/Recall over thresholds

fig, ax = plt.subplots(figsize=(10, 6))
for metric in \['Accuracy', 'Precision', 'Recall']:
for model in results\_df\['Model'].unique():
subset = results\_df\[results\_df\['Model'] == model]
ax.plot(subset\['Threshold'], subset\[metric], marker='o', label=f"{model} - {metric}")

ax.set\_title("Model Performance vs. Information Gain Threshold")

ax.set\_xlabel("Information Gain Threshold")

ax.set\_ylabel("Score")

ax.set\_ylim(0, 1.1)

ax.legend(bbox\_to\_anchor=(1.05, 1), loc='upper left')

ax.grid(True)

fig.tight\_layout()

st.pyplot(fig)

# Highlight the best score across all models and metrics

st.subheader("Best Model by Metric")

for metric in \["Accuracy", "Precision", "Recall"]:

idx = results\_df\[metric].idxmax()

st.markdown(f"**{metric}:** Best Model: `{results_df.loc[idx, 'Model']}`, "

f"Threshold: `{results_df.loc[idx, 'Threshold']}`, "

f"Score: `{results_df.loc[idx, metric]:.2f}`")

# Additional Visualization: Accuracy vs Threshold

st.subheader("Visualization Pilot: Accuracy vs. Information Gain Threshold")

if not results\_df.empty:

fig\_acc, ax\_acc = plt.subplots(figsize=(8, 5))

for model in results\_df\['Model'].unique():

model\_data = results\_df\[results\_df\['Model'] == model]

ax\_acc.plot(model\_data\['Threshold'], model\_data\['Accuracy'], marker='o', label=model)

ax\_acc.set\_title("Accuracy vs. Information Gain Threshold")

ax\_acc.set\_xlabel("Information Gain Threshold")

ax\_acc.set\_ylabel("Accuracy")

ax\_acc.set\_ylim(0, 1.1)

ax\_acc.legend()

ax\_acc.grid(True)

st.pyplot(fig\_acc)

else:

st.write("Please upload a dataset to generate the visualization.")

Don't change anything only Update threshold range in slider, in all 3 graphd bcz my data information Gain value is short
So add range according to that check range value in figure
