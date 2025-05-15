# Oil Pump Predictive Failure Dashboard

## Project Overview

The Oil Pump Predictive Failure Dashboard  is a bespoke data science product dsigned and developed by Isaac Opoku Nkansah with a pseudonym prototype domain business IONARTS Projects Consult. It is designed to empower maintenance engineers and reliability analysts in the oil and gas sector with an easy-to-use tool for predicting potential failures in oil pumps. With the use of machine learning and data analytics, the dashboard facilitates a proactive approach to maintenance, aiming to reduce unplanned downtime, optimize operational efficiency, and support informed decision-making.

This dashboard addresses the industry's need to move beyond traditional reactive or scheduled maintenance by providing data-driven informaition into equipment health. It offers functionalities for both real-time single-pump prediction and comprehensive batch analysis of historical or fleet data.

## Features

* **Real-time Single Prediction:** Predict the failure status of a single oil pump by manually entering its key operational parameters.
* **Batch Prediction:** Upload a CSV file containing data for multiple pumps to get failure predictions for the entire batch.
* **Input Data Visualization:** Explore the characteristics and trends of your uploaded batch data *before* prediction.
* **Post-Prediction Analysis:** Gain insights into the predicted failures within your batch data through various visualizations (failure distribution, feature distributions, correlation heatmap, boxplots).
* **Downloadable PDF Reports:** Generate and download a PDF report summarizing the batch prediction analysis and key visualizations.
* **Customizable Visualizations:** Adjust the color palettes of charts and plots via controls in the sidebar.
* **AI Support Agent:** Access "Isaac," an AI-powered and technically trained support agent, via a lightbox button for assistance with the dashboard.
* **User Feedback:** Provide your expert feedback through an embedded form to help improve the dashboard.
* **Responsive UI:** A user-friendly interface built with Streamlit.

## Project Motivation and Background

Traditional maintenance methods in the oil and gas industry often lead to significant operational inefficiencies, unplanned downtime, and increased costs (Achouch et al., 2022). The literature emphasizes the transformative potential of data science and predictive maintenance in addressing these challenges (Saxena, 2025). When operational data are analysed, it's possible to anticipate equipment failures and schedule maintenance proactively, leading to enhanced reliability and reduced environmental impact (Ozowe et al., 2024).

This project was undertaken as an individual R&D initiative to design and develop a practical data science product that makes these advanced capabilities accessible to domain experts who may not have extensive data science backgrounds. The goal was to create a user-friendly tool that directly supports the transition to a data-driven maintenance strategy within the oil and gas sector.

## Technical Report Summary

A comprehensive technical report detailing the design, development, and project management of this dashboard is available in the repository data/Technical-Report

Key aspects covered in the report include:

* **Product Design:** Discusses the selection of the AI4I 2020 dataset, the analysis of end-user requirements (need for ease of use, actionable insights), functional and non-functional requirements specifications, the Streamlit-based software architecture, and detailed use case specifications (single prediction, batch analysis, customization, support, feedback).
* **Product Development:** Covers the selection of Python and key libraries (Pandas, Scikit-learn, Seaborn, FPDF, Streamlit) as development tools, highlights the use of an Agile/Iterative software engineering methodology, describes the system testing approach (primarily manual/UAT), and outlines a plan for formal user evaluation.
* **Project Management:** Addresses time management (illustrated hypothetically with a Gantt chart structure), discusses risk assessment focusing on data security and personal information protection, outlines quality control measures during development, describes basic customer/user relationship management via integrated feedback and support channels, and proposes a basic product marketing strategy targeting the oil and gas maintenance sector.

The report provides a critical discussion of the choices made throughout the project lifecycle, supported by references from academic literature.

## Getting Started

To run the ION Oil Pump Predictive Failure Dashboard locally, follow these steps:

### Prerequisites

* Python 3.7 or higher installed.
* Git installed.

### 1. Clone the Repository

Open your terminal or command prompt and clone the project repository:

```bash
git clone <https://github.com/Ionkansah/Pump-Failure-Prediction-Dashboard/tree/main>
cd <Pump-Failure-Prediction-Dashboard>
```

### 2. Set up a Virtual Environment (I personally recommend this method)
Creating a virtual environment helps manage project dependencies.
### On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

### On Windows
python -m venv .venv
.venv\Scripts\activate

### 3. Install Dependencies
With your virtual environment activated, install the required libraries:
```bash
pip install --upgrade pip  # You need to ensure pip is up-to-date
pip install -r requirements.txt
```

The requirements.txt file contains the list of necessary packages:
streamlit
pandas
numpy
matplotlib
seaborn
joblib
scikit-learn
Pillow
fpdf
plotly
imbalanced-learn

### 4. Obtain Data and Model Files
You need the dataset (ai4i2020.csv) used for the scaler and the trained model file (trained_model.joblib). 
* **ai4i2020.csv**: This file should be placed in a directory named data in the root of your project. Create the folder if it doesn't exist.
![image](https://github.com/user-attachments/assets/23b890a4-2e77-433a-8ff6-4a6ff7c51924)


* **trained_model.joblib**: This file containing your trained machine learning model should be placed in the root directory of your project (the same directory as NewRecoveryApp.py).
your-project-root/
├── trained_model.joblib  <-- Place the model file here
└── ...

* **Background Image** (You can choose to add or not. I love aesthetical appeal so I recommend): If you used a custom background image, ensure that file is also in your project, and the path in NewRecoveryApp.py is correct relative to where you run the script. Make sure these files are in the correct relative locations as expected by the Python script.

### 5. Run the Streamlit App
From the root directory of your project, with your virtual environment activated:
```bash
streamlit run NewRecoveryApp.py
```

Your default web browser should open a new tab with the dashboard running.

## How to Use the Dashboard
The dashboard provides a straightforward interface:

*Sidebar: Provides information about IONARTS Projects Consult, contact details, links, visualization customization controls, and a company description pop-out.

*Real-time Prediction: Use the input fields under "Predict Machine Failure" to enter parameters for a single pump and get an instant prediction.

*Batch Prediction: Use the file uploader under "Upload Data for Batch Prediction (CSV)" to upload a CSV file with your data (ensure it has the required columns).

* The dashboard will display visualizations of your input data in the "Visualize Uploaded Input Data" expander. It will then show the prediction results in a table with a "Predicted Failure" column.
* Explore post-prediction insights using the provided charts (Failure Distribution, Feature Distributions, Correlation Heatmap, Boxplots).

* **Customize Visualizations**: Use the controls in the sidebar under "🎨 Visualization Customization" to change the color palettes of the charts and plots.
* **Download Report**: If you've uploaded batch data and predictions were made, the "📥 Generate Downloadable Report (PDF)" expander will allow you to download a PDF summary.
* **AI Agent Support**: Click the "Isaac: Prediction Support Agent" button to open a chat window for assistance.
* **Leave Feedback**: Use the embedded form at the bottom to provide feedback on the dashboard.

Ensure your trained_model.joblib is either in the root or in the notebooks/ folder and the path in NewRecoveryApp.py is updated accordingly.

Dependencies
The project relies on the libraries listed in requirements.txt, including:
* **streamlit**: For building the web application.
* **pandas, numpy**: For data manipulation.
* **scikit-learn**: For the ML model and preprocessing (StandardScaler, LabelEncoder).
* **joblib**: For loading the trained model.
* **matplotlib, seaborn, plotly**: For data visualization.
* **Pillow**: For image handling (used for the background).
* **fpdf**: For PDF report generation.
* **imbalanced-learn**: (Included in requirements, potentially used during model training).

## Future Enhancements
* Integration with real-time sensor data streams.
* Development of more advanced machine learning models.
* Enhanced error handling and data validation for uploaded CSVs.
* Implementation of user authentication and authorization for secure deployment.
* More sophisticated visualization options and interactivity.
* Training on larger and more diverse oil pump datasets.
* A/B testing of different UI layouts and features.
* Deployment to a production environment (e.g., Streamlit Community Cloud, AWS, Azure).

License: MIT

## Contact
Developed by Isaac Opoku Nkansah for IONARTS Projects Consult.

## For inquiries, please contact:
Email: isaac3g@outlook.com
Phone: +447392615042
Website: https://github.com/Ionkansah/
