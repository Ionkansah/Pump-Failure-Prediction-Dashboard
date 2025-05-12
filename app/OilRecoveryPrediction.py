import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # For saving and loading the model

# --- Load Data and Preprocess ---
try:
    df = pd.read_csv(r"C:\Users\HP\Documents\DATASCIENCE PRODUCT FOR PREDICTIVE MAINTENANCE\Jupyter env\data\ai4i2020.csv")
except FileNotFoundError:
    st.error("Error: ai4i2020.csv not found. Please ensure the file is in the correct directory.")
    st.stop()

label_encoder = LabelEncoder()
df['Type'] = label_encoder.fit_transform(df['Type'])
X = df.drop(['UDI', 'Product ID', 'Type', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
y = df['Machine failure']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Train Model (or load if you've saved it) ---
# For this example, we'll train the model every time the app runs.
# In a real-world scenario, you'd likely train it once and save it.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# --- Streamlit App ---
st.title("Predictive Maintenance Dashboard")
st.subheader("Analyzing Machine Failure Data")

# --- Data Overview ---
st.subheader("Data Overview")
if st.checkbox("Show Raw Data"):
    st.dataframe(df)

st.subheader("Descriptive Statistics")
st.dataframe(df.describe())

st.subheader("Machine Failure Distribution")
failure_counts = df['Machine failure'].value_counts()
st.bar_chart(failure_counts)
st.write(df['Machine failure'].value_counts(normalize=True))

# --- Interactive Data Exploration ---
st.subheader("Interactive Data Exploration")

# Histograms
st.subheader("Feature Histograms")
selected_feature_hist = st.selectbox("Select a numerical feature for histogram:", X.columns)
fig_hist, ax_hist = plt.subplots()
sns.histplot(df[selected_feature_hist], kde=True, ax=ax_hist)
st.pyplot(fig_hist)

# Scatter Plots
st.subheader("Feature Scatter Plots")
feature_x = st.selectbox("Select the first feature for scatter plot:", X.columns)
feature_y = st.selectbox("Select the second feature for scatter plot:", X.columns, index=1)
fig_scatter, ax_scatter = plt.subplots()
sns.scatterplot(data=df, x=feature_x, y=feature_y, hue='Machine failure', ax=ax_scatter)
st.pyplot(fig_scatter)

# Box Plots
st.subheader("Feature Box Plots by Product Type")
selected_feature_box = st.selectbox("Select a numerical feature for box plot:", X.columns)
fig_box, ax_box = plt.subplots()
sns.boxplot(data=df, x='Type', y=selected_feature_box, ax=ax_box)
ax_box.set_xticklabels(['Low', 'Medium', 'High']) # Assuming 0, 1, 2 correspond to L, M, H
st.pyplot(fig_box)

# --- Model Evaluation ---
st.subheader("Model Evaluation")
st.write(f"**Accuracy:** {accuracy:.2f}")
st.subheader("Classification Report")
st.text(report)

st.subheader("Confusion Matrix")
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Failure', 'Failure'], yticklabels=['No Failure', 'Failure'], ax=ax_cm)
st.pyplot(fig_cm)

# --- Feature Importance ---
st.subheader("Feature Importance")
importances = model.feature_importances_
feature_names = X.columns
sorted_indices = np.argsort(importances)[::-1]

fig_importance, ax_importance = plt.subplots()
sns.barplot(x=importances[sorted_indices], y=feature_names[sorted_indices], ax=ax_importance)
st.pyplot(fig_importance)

st.subheader("Insights and Further Analysis")
st.write("This dashboard provides a visual overview of the machine failure data and the performance of the predictive model.")
st.write("- Explore the distributions of different features and their relationships with machine failure.")
st.write("- Observe the model's accuracy, precision, recall, and F1-score from the classification report.")
st.write("- The confusion matrix shows the counts of true positives, true negatives, false positives, and false negatives.")
st.write("- Feature importance highlights which factors the model deemed most significant in predicting failures.")
st.write("Further analysis could involve exploring different machine learning models, performing more in-depth feature engineering, and deploying this dashboard for real-time monitoring.")