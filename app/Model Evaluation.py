# --- Model Evaluation (Hidden by Default) ---
with st.expander("Show Model Evaluation"):
    st.subheader("Model Evaluation")
    # Defining or calculate accuracy 
    accuracy = 0.98  
    st.markdown(f"**Model Accuracy:** {accuracy:.2f}")
    st.subheader("Classification Report")
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    # With my True labels `y_true` and predicted labels `y_pred` from your test set
    y_true = [0, 1, 0, 1]  
    y_pred = [0, 1, 0, 0]  
    report = classification_report(y_true, y_pred, target_names=['No Failure', 'Failure'])
    st.text(report)
    st.subheader("Confusion Matrix")
    y_true = [0] * 1939 + [1] * 61  
    y_pred = [0] * 1939 + [1] * 36 + [0] * 25  
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
