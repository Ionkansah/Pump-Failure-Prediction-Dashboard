from fpdf import FPDF
import tempfile
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.preprocessing import StandardScaler, LabelEncoder
import streamlit.components.v1 as components
from PIL import Image  # For handling images my added background image
import base64

# --- Loading Data and Preprocessor matching my training ---
try:
    data_file_path = "data/ai4i2020.csv"
    df_original = pd.read_csv(data_file_path)
    df = df_original.copy()
    label_encoder = LabelEncoder()
    df['Type'] = label_encoder.fit_transform(df['Type'])
    X_features = df.drop(['UDI', 'Product ID', 'Type', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
    scaler = StandardScaler()
    scaler.fit(X_features) # This fits scaler on the original training data
except FileNotFoundError:
    st.error("Error: ai4i2020.csv not found. Please ensure the file is in the correct directory.")
    st.stop()

# --- Load my Trained Model saved as 'trained_model.joblib' ---
try:
    model_path = "app/notebooks/trained_model.joblib" # Relative path to the model
    model = load(model_path)
except FileNotFoundError:
    st.error("Error: trained_model.joblib not found. Please ensure the model file is in the same directory.")
    st.stop()

# --- My preferred Sidebar Styling ---
st.sidebar.title("IONARTS Projects Consult")
st.sidebar.markdown("**✨ Dream it..We Deliver it ✨**")
st.sidebar.markdown("Designed by **bi95cz**. Providing cutting-edge Data Science solutions for analytics and risk management in the oil and gas industry.")
st.sidebar.subheader("Contact")
# Using Font Awesome icons requires internet access for the user's browser
st.sidebar.markdown("<link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css'>", unsafe_allow_html=True) # Add Font Awesome CSS
st.sidebar.markdown("<i class='fas fa-envelope'></i> Email: isaac3g@outlook.com", unsafe_allow_html=True)
st.sidebar.markdown("<i class='fas fa-phone'></i> Phone: +447392615042", unsafe_allow_html=True)
st.sidebar.subheader("Links")
st.sidebar.markdown("[<i class='fab fa-github'></i> Our Blog](https://github.com/Ionkansah)", unsafe_allow_html=True)
st.sidebar.markdown("[<i class='fas fa-globe'></i> Company Website](https://github.com/Ionkansah)", unsafe_allow_html=True)


# --- ** Added sidebar Visualization Customization** ---
st.sidebar.subheader("🎨 Visualization Customization")

# Define available palettes
qualitative_palettes = ['Set1', 'Set2', 'Set3', 'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'tab10', 'tab20', 'tab20b', 'tab20c']
sequential_palettes = ['coolwarm', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
diverging_palettes = ['coolwarm', 'bwr', 'seismic', 'Spectral', 'RdBu', 'RdGy', 'PiYG', 'PRGn', 'BrBG', 'PuOr']

# Selectbox for plots using categorical palettes (hist with hue, boxplot)
selected_palette_categorical = st.sidebar.selectbox(
    "Categorical Palette (Histograms/Boxplots):",
    qualitative_palettes + sequential_palettes, # Offer sequential too just in case
    index=qualitative_palettes.index('Set1') # Default to Set1
)

# Selectbox for plots using sequential/diverging palettes (heatmap)
selected_palette_heatmap = st.sidebar.selectbox(
    "Heatmap Palette:",
    sequential_palettes + diverging_palettes,
    index=sequential_palettes.index('inferno') # Default to inferno colour
)

# Color picker for the single line plot
selected_line_color = st.sidebar.color_picker("Line Plot Color:", "#1f77b4") # Default blue



# --- Main Dashboard Background Image ---
def set_background(image_file):
    try:
        with open(image_file, "rb") as f:
            img_data = f.read()
        b64_encoded = base64.b64encode(img_data).decode("utf-8")
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{b64_encoded}");
                background-repeat: repeat-y;
                background-position: left top, right top;
                background-size: 100%, auto; /* Adjust as needed, e.g., '20% auto' */
            }}
            /* Reduce padding around the main block container */
            .main .block-container {{
                padding-top: 1rem;
                padding-right: 1rem;
                padding-left: 1rem;
                padding-bottom: 1rem;
            }}
            /* Target elements inside the main container if needed */
            .main .block-container > div {{
                 max-width: 90%; /* Adjust content width */
                 margin: auto;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.warning(f"Could not set background image: {e}")

# --- Set background image ---
try:
    image_path = "data/Pipes7.jpeg"
    image = Image.open(image_path)
    # target_width = 300 # Adjust this width value
    # aspect_ratio = image.height / image.width
    # target_height = int(target_width * aspect_ratio)
    # image = image.resize((target_width, target_height)) # Optional resize
    resized_image_path = 'pump_network_background_resized.png' # Use a consistent name
    image.save(resized_image_path) # Save the original or resized image
    set_background(resized_image_path)
except FileNotFoundError:
    st.warning("Warning: Background image not found. Background will be default.")
except Exception as e:
    st.error(f"Error loading or setting background image: {e}")

# --- Main Dashboard ---
st.title("Predictive Failure Dashboard for Oil Pumps")
st.markdown("You are welcome to this Dashboard. This predictive oil pump failure dashboard provides real time prediction based on the parameter inputs specified below. For batch inputs, you can just upload your CSV file with the specified column headings for prediction. For more directions, kindly chat with ION, the Dasboard Agent")

# --- AI Agent Integration (Lightbox Button) ---
jotform_lightbox_html = """
<iframe id="JotFormIFrame-01967c62b26b7ba1bdf1890a77124fcfd3e5"
  title="Isaac: Prediction Support Agent" onload="window.parent.scrollTo(0,0)"
  allowtransparency="true" allow="geolocation; microphone; camera; fullscreen"
  src="https://eu.jotform.com/agent/01967c62b26b7ba1bdf1890a77124fcfd3e5?embedMode=iframe&background=1&shadow=1"
  frameborder="0" style="
    min-width:100%;
    max-width:100%;
    height:600px;
    border:none;
    width:40%;
  " scrolling="no">
</iframe>
<script src='https://cdn.jotfor.ms/s/umd/latest/for-form-embed-handler.js'></script>
<script>
  window.jotformEmbedHandler("iframe[id='JotFormIFrame-01967c62b26b7ba1bdf1890a77124fcfd3e5']",
    "https://eu.jotform.com")
</script>
"""

st.markdown(jotform_lightbox_html, unsafe_allow_html=True)

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
        'Tool wear [min]': [tool_wear],
        'TWF': [0], 'HDF': [0], 'PWF': [0], 'OSF': [0], 'RNF': [0]
    })
    numerical_features = ['Air temperature [K]', 'Process temperature [K]',
                            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    binary_features = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    scaled_numerical = scaler.transform(input_data[numerical_features])
    final_input = scaled_numerical
    
    if final_input.shape[1] == model.n_features_in_:
        prediction = model.predict(final_input)[0]
        st.subheader("Prediction Result")
        if prediction == 1: st.error("⚠️ Warning: Machine is predicted to FAIL! The Torque, Rotational Speed or Tool Wear fatally high")
        else: st.success("✅ Machine is predicted to be operating normally. Please maintain similar conditions")
    else: st.error(f"Error: Input data shape mismatch.")

# --- ION(bi95cz) Poduct for Batch Prediction Section ---
st.subheader("Upload Data for Batch Prediction (CSV)")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

batch_data_display = None # Initialize

if uploaded_file is not None:
    try:
        batch_data_input = pd.read_csv(uploaded_file)
        st.success("CSV file uploaded successfully!")

        # --- Input Data Visualization Section ---
        with st.expander("📊 **Visualize Uploaded Input Data**"):
            st.markdown("Explore the characteristics of the data you uploaded.")
            required_viz_cols = ['Product ID', 'Type', 'Air temperature [K]', 'Process temperature [K]',
                                 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
            if all(col in batch_data_input.columns for col in required_viz_cols):
                st.markdown("### Product ID Distribution (Top 20)")
                if batch_data_input['Product ID'].nunique() > 20:
                    st.bar_chart(batch_data_input['Product ID'].value_counts().nlargest(20))
                else:
                    st.bar_chart(batch_data_input['Product ID'].value_counts())
                st.markdown("### Product Type Distribution")
                st.bar_chart(batch_data_input['Type'].value_counts())
                st.markdown("### Numerical Feature Trends (vs. Record Index)")
                numerical_to_plot = ['Air temperature [K]', 'Process temperature [K]',
                                     'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
                selected_num_feature = st.selectbox("Select numerical feature to plot:", numerical_to_plot)
                fig_line, ax_line = plt.subplots(figsize=(10, 4))
                
                # Apply selected color to line plot
                sns.lineplot(data=batch_data_input[selected_num_feature], ax=ax_line, color=selected_line_color)
                ax_line.set_title(f'{selected_num_feature} Trend')
                ax_line.set_xlabel("Record Index")
                ax_line.set_ylabel(selected_num_feature)
                st.pyplot(fig_line)
            else:
                missing_viz_cols = [col for col in required_viz_cols if col not in batch_data_input.columns]
                st.warning(f"Cannot generate input visualizations. Missing columns: {missing_viz_cols}")

        # --- Prediction Logic and Output ---
        required_numerical = ['Air temperature [K]', 'Process temperature [K]',
                                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        required_binary = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        all_required = required_numerical + required_binary
        if all(col in batch_data_input.columns for col in all_required):
            scaled_numerical_batch = scaler.transform(batch_data_input[required_numerical])
            binary_data = batch_data_input[required_binary].values
            final_batch_input = scaled_numerical_batch
            
            if final_batch_input.shape[1] == model.n_features_in_:
                batch_predictions = model.predict(final_batch_input)
                batch_data_display = batch_data_input.copy()
                batch_data_display['Predicted Failure'] = batch_predictions
                st.subheader("Batch Prediction Results")
                st.dataframe(batch_data_display)
                st.info("Column 'Predicted Failure' indicates 1 for predicted failure and 0 for normal operation.")
                st.subheader("Interpretation & Solutions for Failures")
                st.markdown("""
                            - **Column 'Predicted Failure'** indicates:
                                - `1`: Predicted failure.
                                - `0`: Normal operation.
                            - **For failures**, please consider:
                                - Greasing pump levers and insulating pump exteriors against below zero temperatures.
                                - Checking for leaks and loose joints.
                                """)
                # --- Interactive Data Analysis Section (Post-Prediction) ---
                st.subheader("📊 Exploratory Insights (Post-Prediction)")
                st.markdown("### 🔍 Failure Distribution")
                failure_counts = batch_data_display['Predicted Failure'].value_counts()
                st.bar_chart(failure_counts) # Bar charts don't use seaborn palettes directly
                st.markdown("### 📈 Feature Distributions by Prediction")
                selected_hist_feature_post = st.selectbox("Select numerical feature for post-prediction histogram:", required_numerical, key="post_hist_select")
                fig_hist_post, ax_hist_post = plt.subplots()

                # Apply selected categorical palette
                sns.histplot(data=batch_data_display, x=selected_hist_feature_post, hue='Predicted Failure', multiple="stack", palette=selected_palette_categorical, ax=ax_hist_post)
                st.pyplot(fig_hist_post)
                st.markdown("### 🧠 Correlation Heatmap (Numerical Features)")
                fig_corr, ax_corr = plt.subplots(figsize=(8, 5))

                # Apply selected heatmap palette (cmap)
                sns.heatmap(batch_data_display[required_numerical + ['Predicted Failure']].corr(), annot=True, cmap=selected_palette_heatmap, ax=ax_corr)
                st.pyplot(fig_corr)
                st.markdown("### 📦 Boxplot of Feature by Failure Prediction")
                selected_box_feature_post = st.selectbox("Select numerical feature for post-prediction boxplot:", required_numerical, key="post_box_select")
                fig_box_post, ax_box_post = plt.subplots()

                # Apply selected categorical palette
                sns.boxplot(data=batch_data_display, x='Predicted Failure', y=selected_box_feature_post, palette=selected_palette_categorical, ax=ax_box_post)
                ax_box_post.set_xticklabels(['No Failure', 'Failure'])
                st.pyplot(fig_box_post)
            else:
                st.error(f"Error: Input data shape mismatch for prediction.")
        else:
            missing = [col for col in all_required if col not in batch_data_input.columns]
            st.error(f"Error: Uploaded CSV is missing required columns for prediction: {missing}")
    except Exception as e:
        st.error(f"Error loading or processing CSV file: {e}")
        st.write(f"Detailed error: {e}")

def generate_pdf_report(data, failure_counts, selected_feature, correlation_matrix, box_feature,
                        palette_categorical, palette_heatmap): # Pass palettes here
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Machine Failure Prediction Report", ln=True, align='C')
    pdf.ln(5)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="Failure Summary:", ln=True)
    for key, value in failure_counts.items():
        label = 'Failure' if key == 1 else 'No Failure'
        pdf.cell(200, 8, txt=f"{label}: {value} instances", ln=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Histogram - Apply categorical palette
        fig1, ax1 = plt.subplots()
        sns.histplot(data=data, x=selected_feature, hue='Predicted Failure', multiple='stack', palette=palette_categorical, ax=ax1)
        hist_path = os.path.join(tmpdir, "histogram.png")
        fig1.savefig(hist_path); plt.close(fig1)
        pdf.image(hist_path, x=10, w=180)

        # Correlation heatmap - Apply heatmap palette (cmap)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.heatmap(correlation_matrix, annot=True, cmap=palette_heatmap, ax=ax2)
        corr_path = os.path.join(tmpdir, "correlation.png")
        fig2.savefig(corr_path); plt.close(fig2)
        pdf.add_page()
        pdf.cell(200, 10, txt="Correlation Heatmap:", ln=True)
        pdf.image(corr_path, x=10, w=180)

        # Boxplot - Apply categorical palette
        fig3, ax3 = plt.subplots()
        sns.boxplot(data=data, x='Predicted Failure', y=box_feature, palette=palette_categorical, ax=ax3)
        ax3.set_xticklabels(['No Failure', 'Failure'])
        box_path = os.path.join(tmpdir, "boxplot.png")
        fig3.savefig(box_path); plt.close(fig3)
        pdf.add_page()
        pdf.cell(200, 10, txt=f"Boxplot of {box_feature} by Predicted Failure:", ln=True)
        pdf.image(box_path, x=10, w=180)
    pdf.set_y(-20)
    pdf.set_font("Arial", style='I', size=8)
    pdf.cell(0, 10, "Report Generated by IONARTS PROJECTS CONSULT", 0, 0, 'C')
    return pdf.output(dest='S').encode('latin1')

# --- Toggle and Download PDF Report ---
with st.expander("📥 Generate Downloadable Report (PDF)"):
    st.markdown("Customize and download an automatic visual report of your uploaded data.")
    required_numerical = ['Air temperature [K]', 'Process temperature [K]',
                            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    required_binary = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    all_required = required_numerical + required_binary

    if uploaded_file is not None and batch_data_display is not None:
        try:
            if all(col in batch_data_display.columns for col in all_required + ['Predicted Failure']):
                correlation_matrix = batch_data_display[required_numerical + ['Predicted Failure']].corr()
                failure_counts = batch_data_display['Predicted Failure'].value_counts().to_dict()
                selected_feature = st.session_state.get("post_hist_select", required_numerical[0])
                box_feature = st.session_state.get("post_box_select", required_numerical[1])

                if st.button("Generate & Download PDF Report"):
                    pdf_bytes = generate_pdf_report(batch_data_display.copy(),
                                                    failure_counts,
                                                    selected_feature,
                                                    correlation_matrix,
                                                    box_feature,
                                                    selected_palette_categorical, # Pass selected palette
                                                    selected_palette_heatmap) # Pass selected heatmap palette
                    st.download_button(
                        label="📄 Download Report as PDF",
                        data=pdf_bytes,
                        file_name="failure_report_IONARTS.pdf",
                        mime='application/pdf'
                    )
            else:
                 st.error("Error: Could not find necessary columns for report generation.")
        except Exception as e:
            st.error(f"Error generating PDF report: {e}")
            st.write(f"Detailed error: {e}")
    elif uploaded_file is not None:
         st.warning("Prediction step might have failed. Cannot generate report.")
    else:
        st.warning("Please upload a CSV file and run predictions to generate the report.")

# --- User Review Section (Embedded Jotform) ---
st.markdown("---")
st.header("Leave Your Feedback")
st.markdown("We value your feedback! Please share your experience using the dashboard below.")
jotform_embed_url = "https://form.jotform.com/251204859375361"
components.iframe(jotform_embed_url, height=400, scrolling=True)

# --- Footer ---
st.markdown("---")
st.markdown("© Developed by **bi95cz** with pseudonym IONARTS PROJECTS CONSULT- 2025")
