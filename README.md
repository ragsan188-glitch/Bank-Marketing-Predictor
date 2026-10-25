# TermWise — Bank Marketing Prediction

<p align="center">
  A machine learning web application that predicts customer subscription likelihood for bank term deposits.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit" alt="Streamlit">
</p>

---

## About the Project

**TermWise** is a machine learning application built using Python and Streamlit. It uses a trained K-Nearest Neighbors (KNN) model to estimate whether a customer is likely to subscribe to a bank term deposit based on customer information and marketing campaign attributes.

The project focuses on turning a trained machine learning model into an interactive web application.

## Key Features

- Predicts customer subscription likelihood.
- Interactive form for customer and campaign inputs.
- Integrates a trained Scikit-learn pipeline.
- Displays predictions as **Potential subscriber** or **Less likely to subscribe**.
- Includes a customizable interface theme.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| Scikit-learn | Model training and prediction |
| Streamlit | Web application |
| Joblib | Saving and loading the model |

## Machine Learning Model

The application uses a **K-Nearest Neighbors (KNN) classifier** trained on the Bank Marketing dataset.

A saved Scikit-learn pipeline handles preprocessing and prediction. The `duration` feature is excluded to avoid relying on call duration when predicting subscription likelihood before a marketing call.

## Getting Started

### Prerequisites

- Python installed on your system
- Git
- The trained model file: `bank_marketing_knn.pkl`

### Installation

Clone the repository:

    git clone https://github.com/YOUR-USERNAME/termwise-bank-marketing.git

Navigate to the project directory:

    cd termwise-bank-marketing

Install dependencies:

    pip install -r requirements.txt

Run the application:

    streamlit run app.py

## Deployment

The application can be deployed using [Streamlit Community Cloud](https://share.streamlit.io/).

Connect the GitHub repository, select the `main` branch, specify `app.py` as the main file, and deploy.

Ensure that the trained model file is available at the path expected by the application.

## Future Improvements

- Add model performance metrics and evaluation visualizations.
- Develop an interactive analytics dashboard.
- Improve prediction transparency and explainability.
- Expand model comparison and experimentation.

## Author

**Ragul Y. S.**

- LinkedIn: [Ragul Y. S.](https://www.linkedin.com/in/ragul-y-s-a05253394/)
- Email: [ragsan188@gmail.com](mailto:ragsan188@gmail.com)

---

*Developed as a machine learning and deployment portfolio project.*
