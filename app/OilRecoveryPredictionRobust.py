import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load  # Assuming you've saved your trained model
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- Load Data and Preprocessor (Ensure these match your training) ---
try:
    df_original = pd.read_csv(r"C:\Users\HP\Documents\DATASCIENCE PRODUCT FOR PREDICTIVE MAINTENANCE\Jupyter env\data\ai4i2020.csv")
    df = df_original.copy()
    label_encoder = LabelEncoder()
    df['Type'] = label_encoder.fit_transform(df['Type'])
    X_features = df.drop(['UDI', 'Product ID', 'Type', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
    scaler = StandardScaler()
    scaler.fit(X_features) # Fit scaler on the original training data
except FileNotFoundError:
    st.error("Error: ai4i2020.csv not found. Please ensure the file is in the correct directory.")
    st.stop()

# --- Load Trained Model (Replace 'trained_model.joblib' with your actual saved model file) ---
try:
    model = load('trained_model.joblib')
    st.write("Model expects features:", model.feature_names_in_)
    st.write("Number of features expected:", model.n_features_in_)
except FileNotFoundError:
    st.error("Error: trained_model.joblib not found. Please ensure the model file is in the same directory.")
    st.stop()

# --- Sidebar ---
st.sidebar.title("About Our Company")
st.sidebar.markdown("**[Your Oil & Gas Company Name Here]**")
st.sidebar.markdown("Providing cutting-edge solutions for risk management in the oil and gas industry.")
st.sidebar.subheader("Contact Us")
st.sidebar.markdown("Email: [your_email@company.com]")
st.sidebar.markdown("Phone: [your_phone_number]")
st.sidebar.subheader("Important Links")
st.sidebar.markdown("[Our Blog]([Your Blog URL])")
st.sidebar.markdown("[Company Website]([Your Website URL])")

# --- Main Dashboard ---
st.title("Predictive Maintenance Dashboard for Oil Pumps")
st.markdown("Leverage data-driven insights to proactively manage the health of your critical oil pump machinery and mitigate potential risks.")

# --- Prediction Section ---
st.header("Predict Machine Failure")
st.subheader("Enter Pump Parameters")

col1, col2, col3 = st.columns(3)
air_temperature = col1.number_input("Air Temperature [K]", value=298.0)
process_temperature = col2.number_input("Process Temperature [K]", value=308.0)
rotational_speed = col3.number_input("Rotational Speed [rpm]", value=1500.0)

col4, col5 = st.columns(2)
torque = col4.number_input("Torque [Nm]", value=40.0)
tool_wear = col5.number_input("Tool Wear [min]", value=10.0)

predict_button = st.button("Predict Failure")

if predict_button:
    input_data = pd.DataFrame({
        'Air temperature [K]': [air_temperature],
        'Process temperature [K]': [process_temperature],
        'Rotational speed [rpm]': [rotational_speed],
        'Torque [Nm]': [torque],
        'Tool wear [min]': [tool_wear]
    })
    scaled_input = scaler.transform(input_data)
    if scaled_input.shape[1] == model.n_features_in_:
        prediction = model.predict(scaled_input)[0]
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("Warning: Machine is predicted to FAIL!")
        else:
            st.success("Machine is predicted to be operating normally.")
    else:
        st.error(f"Error: Input data has {scaled_input.shape[1]} features, but the model expects {model.n_features_in_} features.")

# --- Batch Prediction Section ---
st.subheader("Upload Data for Batch Prediction (CSV)")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        required_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        if all(col in batch_data.columns for col in required_cols):
            scaled_batch_data = scaler.transform(batch_data[required_cols])
            if scaled_batch_data.shape[1] == model.n_features_in_:
                batch_predictions = model.predict(scaled_batch_data)
                batch_data['Predicted Failure'] = batch_predictions
                st.subheader("Batch Prediction Results")
                st.dataframe(batch_data)
                st.info("Column 'Predicted Failure' indicates 1 for predicted failure and 0 for normal operation.")
            else:
                st.error(f"Error: Uploaded CSV has {scaled_batch_data.shape[1]} features, but the model expects {model.n_features_in_} features.")
        else:
            st.error(f"Error: Uploaded CSV must contain the following columns: {required_cols}")
    except Exception as e:
        st.error(f"Error loading or processing CSV file: {e}")

# --- Model Evaluation (Hidden by Default) ---
with st.expander("Show Model Evaluation"):
    st.subheader("Model Evaluation")
    # Define or calculate accuracy (replace 0.95 with the actual accuracy value from your model evaluation)
    accuracy = 0.98  # Example placeholder value
    st.markdown(f"**Model Accuracy:** {accuracy:.2f}")
    st.subheader("Classification Report")
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    # Assuming you have true labels `y_true` and predicted labels `y_pred` from your test set
    y_true = [0, 1, 0, 1]  # Replace with actual true labels
    y_pred = [0, 1, 0, 0]  # Replace with actual predicted labels
    report = classification_report(y_true, y_pred, target_names=['No Failure', 'Failure'])
    st.text(report)
    st.subheader("Confusion Matrix")
    y_true = [0] * 1939 + [1] * 61  # Replace with actual true labels
    y_pred = [0] * 1939 + [1] * 36 + [0] * 25  # Replace with actual predicted labels
    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Failure', 'Failure'], yticklabels=['No Failure', 'Failure'], ax=ax_cm)
    st.pyplot(fig_cm)
    st.markdown(
        """
        **Understanding the Model Evaluation:**

        The **Accuracy** score indicates the overall correctness of the model's predictions. A higher accuracy means the model correctly identified more instances of both normal operation and potential failures.

        The **Classification Report** provides a more detailed breakdown:
        - **Precision:** Out of all the times the model predicted a 'Failure', what proportion were actually failures? (High precision means fewer false alarms).
        - **Recall:** Out of all the actual 'Failures' that occurred, what proportion did the model correctly identify? (High recall means the model didn't miss many actual failures).
        - **F1-score:** This is the harmonic mean of precision and recall, providing a balanced measure of the model's performance.
        - **Support:** The number of actual occurrences of each class (0 for no failure, 1 for failure) in the test data.

        The **Confusion Matrix** visually represents the model's predictions:
        - **True Positives (TP):** The model correctly predicted a failure when there was a failure.
        - **True Negatives (TN):** The model correctly predicted no failure when there was no failure.
        - **False Positives (FP):** The model incorrectly predicted a failure when there was no failure (a false alarm).
        - **False Negatives (FN):** The model incorrectly predicted no failure when there was a failure (a missed failure).
        """
    )

# --- Data Exploration (Hidden by Default) ---
with st.expander("Show Data Exploration"):
    st.subheader("Data Overview")
    st.dataframe(df_original.head())

    st.subheader("Descriptive Statistics")
    st.dataframe(df_original.describe())

    st.subheader("Machine Failure Distribution")
    failure_counts = df_original['Machine failure'].value_counts()
    st.bar_chart(failure_counts)
    st.write(df_original['Machine failure'].value_counts(normalize=True))

    st.subheader("Interactive Data Exploration")

    # Histograms
    st.subheader("Feature Histograms")
    numerical_cols = df_original.select_dtypes(include=np.number).columns.tolist()
    selected_feature_hist = st.selectbox("Select a numerical feature for histogram:", numerical_cols)
    fig_hist, ax_hist = plt.subplots()
    sns.histplot(df_original[selected_feature_hist], kde=True, ax=ax_hist)
    st.pyplot(fig_hist)

    # Scatter Plots
    st.subheader("Feature Scatter Plots")
    feature_x = st.selectbox("Select the first feature for scatter plot:", numerical_cols)
    feature_y = st.selectbox("Select the second feature for scatter plot:", numerical_cols, index=1)
    fig_scatter, ax_scatter = plt.subplots()
    sns.scatterplot(data=df_original, x=feature_x, y=feature_y, hue='Machine failure', ax=ax_scatter)
    st.pyplot(fig_scatter)

    # Box Plots
    st.subheader("Feature Box Plots by Product Type")
    numerical_cols_for_box = [col for col in numerical_cols if col not in ['UDI', 'Product ID']]
    selected_feature_box = st.selectbox("Select a numerical feature for box plot:", numerical_cols_for_box)
    fig_box, ax_box = plt.subplots()
    sns.boxplot(data=df_original, x='Type', y=selected_feature_box, ax=ax_box)
    ax_box.set_xticklabels(['Low', 'Medium', 'High']) # Assuming 0, 1, 2 correspond to L, M, H
    st.pyplot(fig_box)

    # Confusion Matrix (Redundant here as it's in Model Evaluation, but kept for completeness if this section is viewed independently)
    # Feature Importance (Redundant here as it's in Model Evaluation)

st.markdown("---")
st.markdown("© [Your Oil & Gas Company Name Here] 2025")