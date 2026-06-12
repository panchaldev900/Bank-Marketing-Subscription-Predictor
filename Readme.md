# 🏦 Bank Marketing Subscription Prediction System

An end-to-end Machine Learning project that predicts whether a bank customer is likely to subscribe to a term deposit based on demographic, financial, and marketing campaign data.

## 📌 Project Overview

Banks often conduct marketing campaigns to promote term deposits. Contacting every customer can be costly and inefficient. This project uses Machine Learning to identify customers who are more likely to subscribe, helping improve marketing efficiency and reduce operational costs.

The project includes:

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning Model Development
- Model Evaluation
- Interactive Streamlit Web Application
- Deployment Ready Architecture

---

## 🎯 Objective

To predict whether a customer will subscribe to a term deposit using historical banking and campaign data.

Target Variable:

- **Yes** → Customer Subscribed
- **No** → Customer Did Not Subscribe

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Pickle

### Web Framework
- Streamlit

### Version Control
- Git
- GitHub

---

## 📊 Dataset Features

The dataset contains customer information such as:

- Age
- Job
- Marital Status
- Education
- Account Balance
- Housing Loan
- Personal Loan
- Campaign Contacts
- Previous Contacts
- Previous Campaign Outcome
- Month of Contact

---

## 🔍 Project Workflow

### 1. Data Collection
- Loaded Bank Marketing Dataset

### 2. Data Preprocessing
- Missing Value Analysis
- Duplicate Value Check
- Data Type Verification
- Feature Transformation

### 3. Feature Engineering
Created additional features such as:
- Age Groups
- Balance Groups
- Month Mapping

### 4. Exploratory Data Analysis (EDA)
Performed:
- Age Analysis
- Balance Analysis
- Campaign Analysis
- Correlation Heatmap
- Subscription Trend Analysis

### 5. Encoding
Converted categorical variables into numerical format using One-Hot Encoding.

### 6. Model Training
Implemented:
- Logistic Regression
- Random Forest Classifier

### 7. Model Evaluation
Evaluated models using:
- Accuracy Score
- Confusion Matrix
- Classification Metrics

### 8. Deployment
Built an interactive Streamlit web application for real-time prediction.

---

## 📈 Model Performance

| Model | Accuracy |
|---------|----------|
| Logistic Regression | ~89.57% |
| Random Forest | ~89.57% |

> Note: Initial 100% accuracy was caused by data leakage (`response_yes`). After removing leakage, the model achieved realistic and reliable performance.

---

## 💻 Web Application Features

### Customer Prediction Form
Users can enter:

- Age
- Job
- Balance
- Housing Loan Status
- Personal Loan Status
- Campaign Information
- Previous Outcome

### Prediction Output

The application provides:

- Subscription Prediction
- Probability Score
- Confidence Visualization
- Customer Summary

---

## 📂 Project Structure

```text
Bank-Marketing-Subscription-Predictor/
│
├── app.py
├── model.pkl
├── features.pkl
├── notebook.ipynb
├── requirements.txt
├── README.md
└── images/
```

---

## 🚀 How to Run Locally

### Clone Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
```

### Navigate to Project Folder

```bash
cd Bank-Marketing-Subscription-Predictor
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit Application

```bash
streamlit run app.py
```

---

## 🌐 Deployment

This project can be deployed using:

- Streamlit Community Cloud
- Render
- Railway

---

## 💡 Business Impact

This solution helps banks:

- Identify high-potential customers
- Reduce marketing costs
- Improve campaign effectiveness
- Increase conversion rates
- Support data-driven decision making

---

## 📚 Learning Outcomes

Through this project, I gained practical experience in:

- Data Analysis
- Feature Engineering
- Machine Learning
- Model Evaluation
- Data Leakage Detection
- Streamlit Development
- Git & GitHub
- Model Deployment

---

## 👨‍💻 Author

**Dev Panchal**


---

## ⭐ If you found this project useful

Please consider giving it a star on GitHub!
