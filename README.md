# ai-employee-attrition-prediction
AI-based employee attrition prediction and workforce analytics platform.
# AI Employee Attrition Prediction

A Machine Learning web application that predicts whether an employee is likely to leave the company based on HR-related factors such as age, job level, income, overtime, and years of experience.

The project integrates a trained ML model with a Flask web interface to provide real-time predictions.

---

## Project Overview

Employee attrition is a significant challenge for organizations. High attrition increases hiring costs, reduces productivity, and results in the loss of experienced employees.

This project aims to predict employee attrition using machine learning so HR teams can identify at-risk employees and take proactive retention measures.

---

## Features

* Predict employee attrition using a trained ML model
* Web interface for entering employee information
* Displays attrition probability and retention probability
* Shows important factors influencing prediction
* Interactive frontend using HTML, CSS, and JavaScript

---

## Tech Stack

Backend

* Python
* Flask

Machine Learning

* scikit-learn
* imbalanced-learn (SMOTE)

Data Processing

* pandas
* numpy

Visualization

* matplotlib
* seaborn

Model Persistence

* joblib

Explainability

* shap

---

## Project Structure

```
ai-employee-attrition-prediction
│
├── models
│   └── employee_attrition_model.pkl
│
├── notebooks
│   └── Employee_Attrition_EDA.ipynb
│
├── static
│   ├── css
│   │   └── style.css
│   └── js
│       └── script.js
│
├── templates
│   └── index.html
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/maheshwariakshat34/ai-employee-attrition-prediction.git
```

### 2. Navigate to project folder

```
cd ai-employee-attrition-prediction
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run the application

```
python app.py
```

### 5. Open in browser

```
http://127.0.0.1:5000
```

---

## Machine Learning Workflow

1. Data cleaning and preprocessing
2. Exploratory Data Analysis (EDA)
3. Handling class imbalance using SMOTE
4. Model training using scikit-learn
5. Model evaluation using metrics and confusion matrix
6. Saving the trained model with joblib
7. Integration with Flask API for prediction

---

## Example Prediction Output

Prediction: Employee Likely to Leave

Attrition Probability: 0.72
Retention Probability: 0.28

Top Contributing Factors:

* Overtime
* Job Level
* Monthly Income

---

## Future Improvements

* Deploy the application online
* Add interactive HR analytics dashboard
* Improve UI with charts and visualizations
* Add more model explainability features

---

## Author

Akshat Lakhotiya

Lakshay Bindal

Ashish Kumar Yadav

---

## License

This project is licensed under the MIT License.
