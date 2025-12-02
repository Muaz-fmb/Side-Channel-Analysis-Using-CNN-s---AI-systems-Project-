# Deep Learning Side-Channel Analysis (SCA) - Project Overview

This project implements a full Deep Learning pipeline using Azure Databricks to perform Side-Channel Analysis (SCA) on the ASCAD benchmark dataset. The repository is organized into distinct phases, covering data ingestion/exploration, model development, and deployment.

## Project Structure

```text
├── Project Phase 1/
│   └── EDA.ipynb               # Data mounting with the storage account & the EDA part
├── Project Phase 2/
│   └── CNN_Model_Training.ipynb
└── ascad-side-channel-analyzer/ # Deployment Source Code
    ├── dataset/                # Place the ASCAD dataset here
    ├── model/                  # Place the trained .h5/.tflite model here
    ├── templates/
    │   └── index.html          # Web interface template
    ├── app.py                  # Main application entry point
    └── requirements.txt        # Python dependencies
```

Deployment Instructions
To run the deployment interface locally, follow these steps:

Navigate to the analyzer directory:

```
cd ascad-side-channel-analyzer
```

Install Dependencies: Ensure you have the necessary libraries installed by running:

```
pip install -r requirements.txt
```

Setup Data & Model:

Move your trained model file into the ascad-side-channel-analyzer/model/ directory.

Move the ASCAD dataset file into the ascad-side-channel-analyzer/dataset/ directory.

Run the Application: Start the server with the following command:
```
python app.py
```

### Next Step
Would you like me to help you create the content for the `requirements.txt` file now, so it matches the imports used in your `app.py`?



