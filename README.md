# Oil Pump Predictive Failure Dashboard

## Project Overview

The Oil Pump Predictive Failure Dashboard  is a bespoke data science product dsigned and developed by Isaac Opoku Nkansah with a pseudonym prototype domain business IONARTS Projects Consult. It is designed to empower maintenance engineers and reliability analysts in the oil and gas sector with an easy-to-use tool for predicting potential failures in oil pumps. With the use of machine learning and data analytics, the dashboard facilitates a proactive approach to maintenance, aiming to reduce unplanned downtime, optimize operational efficiency, and support informed decision-making.

This dashboard addresses the industry's need to move beyond traditional reactive or scheduled maintenance by providing data-driven informaition into equipment health. It offers functionalities for both real-time single-pump prediction and comprehensive batch analysis of historical or fleet data.

## Title and Description

**Project Title:** ION Oil Pump Predictive Failure Dashboard

**Description:** The ION Oil Pump Predictive Failure Dashboard is a user-friendly data science application developed by Isaac Opoku Nkansah for IONARTS Projects Consult. It provides tools for predicting potential failures in oil pumps based on operational data, facilitating proactive maintenance and risk management within the oil and gas sector. The dashboard supports both real-time single predictions and batch analysis of historical data, offering valuable insights through visualizations and downloadable reports.

## API Reference

This project is a web-based dashboard application and can be accessed for integrations [here](https://pumpfailureprediction-bi95cz.streamlit.app/?embed_options=show_toolbar,light_theme,show_colored_line,show_padding,disable_scrolling,show_footer,dark_theme)

## Appendix

Appendices related to the project, such as detailed dataset descriptions, model training specifics, or comprehensive evaluation results, are included in the [Project Technical Report](https://drive.google.com/file/d/10VpD8HawbZKFxOwlMkuZFlYx8P5I1I3_/view?usp=sharing).

## Author

Isaac Opoku Nkansah (bi95cz)

## Color Reference

The dashboard includes a **Visualization Customization** section in the sidebar allowing users to select color palettes for categorical plots (histograms, boxplots) and heatmaps, as well as a specific color for the numerical feature line plot. 
The default palettes are 'Set1' for categorical plots and 'coolwarm' for heatmaps, with a default blue for the line plot. Users can interactively change these via the sidebar controls.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add YourFeature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Open a Pull Request.

Please ensure your code adheres to standard Python style guides like PEP 8 and include appropriate documentation and tests if applicable.

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
streamlit,
pandas,
numpy,
matplotlib,
seaborn,
joblib,
scikit-learn,
Pillow,
fpdf,
plotly,
imbalanced-learn

### 4. Obtain Data and Model Files
You need the dataset (ai4i2020.csv) used for the scaler and the trained model file (trained_model.joblib). 
* **ai4i2020.csv**: This file should be placed in a directory named data in the root of your project. Create the folder if it doesn't exist.
* **trained_model.joblib**: This file containing your trained machine learning model should be placed in the root directory of your project (the same directory as NewRecoveryApp.py).
* **Background Image** (You can choose to add or not. I love aesthetical appeal so I recommend): If you used a custom background image, ensure that file is also in your project, and the path in NewRecoveryApp.py is correct relative to where you run the script. Make sure these files are in the correct relative locations as expected by the Python script.
![Screenshot 2025-05-17 232729](https://github.com/user-attachments/assets/c89ce6ef-0cd2-45bc-afca-6324919db9b6)


### 5. Run the Streamlit App
From the root directory of your project, with your virtual environment activated:
```bash
streamlit run NewRecoveryApp.py
```

Your default web browser should open a new tab with the dashboard running.

## How to Use the Dashboard
The dashboard provides a straightforward interface:

* **Sidebar**: Provides information about IONARTS Projects Consult, contact details, links, visualization customization controls, and a company description pop-out.

* **Real-time Prediction**: Use the input fields under "Predict Machine Failure" to enter parameters for a single pump and get an instant prediction.

* **Batch Prediction**: Use the file uploader under "Upload Data for Batch Prediction (CSV)" to upload a CSV file with your data (ensure it has the required columns).

* The dashboard will display visualizations of your input data in the "Visualize Uploaded Input Data" expander. It will then show the prediction results in a table with a "Predicted Failure" column.
* Explore post-prediction insights using the provided charts (Failure Distribution, Feature Distributions, Correlation Heatmap, Boxplots).

* **Customize Visualizations**: Use the controls in the sidebar under "🎨 Visualization Customization" to change the color palettes of the charts and plots.
* **Download Report**: If you've uploaded batch data and predictions were made, the "📥 Generate Downloadable Report (PDF)" expander will allow you to download a PDF summary.
* **AI Agent Support**: Click the "Isaac: Prediction Support Agent" button to open a chat window for assistance.
* **Leave Feedback**: Use the embedded form at the bottom to provide feedback on the dashboard.

Ensure your trained_model.joblib is either in the root or in the notebooks/ folder and the path in NewRecoveryApp.py is updated accordingly.

**Dependencies**-
The project relies on the libraries listed in requirements.txt, including:
* **streamlit**: For building the web application.
* **pandas, numpy**: For data manipulation.
* **scikit-learn**: For the ML model and preprocessing (StandardScaler, LabelEncoder).
* **joblib**: For loading the trained model.
* **matplotlib, seaborn, plotly**: For data visualization.
* **Pillow**: For image handling (used for the background).
* **fpdf**: For PDF report generation.
* **imbalanced-learn**: (Included in requirements, potentially used during model training).

## Documentation

The primary documentation for this project is the [Project Technical Report](https://drive.google.com/file/d/10VpD8HawbZKFxOwlMkuZFlYx8P5I1I3_/view?usp=sharing). This report provides a critical, referenced discussion on the project's design, development, and management.

Additional documentation is available through:

* **Code Comments:** Explanations within the `NewRecoveryApp.py` script.
* **README File:** This document serves as a quick start guide and overview.

## FAQ

* **Q: What kind of data does the dashboard accept for batch prediction?**
    * **A:** The dashboard accepts CSV files with specific column headings corresponding to the operational parameters and binary failure flags used by the model. Required columns include: `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`. It also uses `Product ID` and `Type` for visualization.
* **Q: Is the model retrained by the dashboard?**
    * **A:** No, the dashboard loads a pre-trained model (`trained_model.joblib`). The scaler is fit on the `ai4i2020.csv` dataset each time the app runs to ensure consistency, but the ML model itself is static.
* **Q: How accurate are the predictions?**
    * **A:** The accuracy depends on the performance of the `trained_model.joblib` file used. This model was trained on a specific dataset (AI4I 2020). For real-world use, retraining on your specific operational data is recommended to improve relevance and accuracy.

  ## Usage/Examples

Watch the [Project Video Tutorial](https://drive.google.com/file/d/10VpD8HawbZKFxOwlMkuZFlYx8P5I1I3_/view?usp=sharing), for detailed instructions on using the dashboard's features, including manual input, batch data upload, and visualization customization.You can also view this Readme 

## Used By
- Majorly Maintenance Managers, Reliability Engineers, Operations Supervisors in Oil & Gas companies
- Other interested users who find its need in their own domains.

### License: MIT

### Contact
Developed by Isaac Opoku Nkansah for IONARTS Projects Consult.

### For inquiries, please contact:
* Email: isaac3g@outlook.com
* Phone: +447392615042
* Website: https://github.com/Ionkansah/
