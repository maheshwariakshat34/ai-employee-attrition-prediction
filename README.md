#  AI Employee Attrition Prediction

> A machine learning–powered web application that predicts whether an employee is likely to leave an organization, enabling HR teams to take proactive retention actions.

🔗 **Live Demo:** [https://employee-attrition-predictor-jxcw.onrender.com/signup](https://employee-attrition-predictor-jxcw.onrender.com/signup)

---

## Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [ML Model](#ml-model)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running Locally](#running-locally)
- [Screenshots](#screenshots)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## About the Project

Employee attrition is one of the most costly challenges organizations face. Losing a skilled employee means not just losing their expertise but also incurring recruitment, onboarding, and training expenses — often exceeding 1.5× the employee's annual salary.

This project uses **Artificial Intelligence and Machine Learning** to predict the probability of an employee leaving an organization based on various HR factors. HR managers and business decision-makers can use this tool to:

- Identify at-risk employees early
- Understand the key drivers of attrition
- Design targeted retention strategies

---

##  Features

- 🔐 **User Authentication** — Secure Signup & Login system
- 📊 **Attrition Prediction** — Real-time prediction based on employee data input
- 📈 **Model Insights** — View which factors most influence the prediction
- 🖥️ **Clean Web UI** — Simple, intuitive interface for HR professionals
- ☁️ **Cloud Deployed** — Hosted live on Render for anywhere access

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML, CSS, JavaScript (Jinja2 Templates) |
| **Backend** | Python, Flask |
| **ML / Data** | Scikit-learn, Pandas, NumPy |
| **Database** | SQLite / SQLAlchemy |
| **Deployment** | Render |
| **Version Control** | Git & GitHub |

---

## 📂 Dataset

The model is trained on the **IBM HR Analytics Employee Attrition & Performance** dataset, widely used for HR analytics research.

- **Rows:** 1,470 employee records
- **Columns:** 35 features
- **Target Variable:** `Attrition` (Yes / No)

**Key Features Used:**

| Feature | Description |
|---------|-------------|
| `Age` | Employee age |
| `MonthlyIncome` | Monthly salary |
| `OverTime` | Whether the employee works overtime |
| `JobSatisfaction` | Job satisfaction rating (1–4) |
| `WorkLifeBalance` | Work-life balance score |
| `YearsAtCompany` | Total years spent at the company |
| `JobRole` | Current job role |
| `EnvironmentSatisfaction` | Satisfaction with work environment |
| `NumCompaniesWorked` | Number of companies previously worked at |
| `DistanceFromHome` | Distance from home to office |

---

## 🤖 ML Model

The prediction pipeline follows these steps:

1. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical variables (Label Encoding / One-Hot Encoding)
   - Feature scaling (StandardScaler)

2. **Model Training**
   - Multiple classifiers evaluated: Logistic Regression, Decision Tree, Random Forest
   - Best model selected based on **Accuracy**, **Precision**, **Recall**, and **F1-Score**
   - Handling class imbalance with SMOTE / class weights

3. **Model Evaluation**
   - Confusion Matrix
   - Classification Report
   - ROC-AUC Score

4. **Serialization**
   - Trained model saved as `model.pkl` using `joblib` / `pickle`
   - Loaded in Flask for real-time inference

---

## 🗂 Project Structure

```
ai-employee-attrition-prediction/
│
├── app.py                   # Main Flask application
├── model.pkl                # Trained ML model (serialized)
├── requirements.txt         # Python dependencies
│
├── templates/               # HTML templates (Jinja2)
│   ├── index.html
│   ├── signup.html
│   ├── login.html
│   └── predict.html
│
├── static/                  # CSS, JS, images
│   ├── css/
│   └── js/
│
├── notebook/                # Jupyter Notebooks for EDA & Model Training
│   └── attrition_model.ipynb
│
├── dataset/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.8+
- pip
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/maheshwariakshat34/ai-employee-attrition-prediction.git

# 2. Navigate to the project directory
cd ai-employee-attrition-prediction

# 3. Create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
python app.py
```

Open your browser and go to: [http://localhost:5000](http://localhost:5000)

---

## 📸 Screenshots

| Page | Preview |
|------|---------|
| **Signup Page** | User registration to access the app |
| **Login Page** | Secure authentication |
| **Prediction Form** | Enter employee data to get prediction |
| **Result Page** | View attrition prediction with confidence |

---

## ☁️ Deployment

This app is deployed on **Render** (free tier).

**Live URL:** [https://employee-attrition-predictor-jxcw.onrender.com](https://employee-attrition-predictor-jxcw.onrender.com)

> ⚠️ Note: The app may take **30–60 seconds** to load on first visit as Render spins up the free-tier server from sleep.

To deploy your own instance on Render:
1. Push your project to GitHub
2. Go to [render.com](https://render.com) and create a new **Web Service**
3. Connect your GitHub repository
4. Set the **Start Command** to: `python app.py`
5. Add environment variables if needed
6. Deploy!

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the project
2. Create your feature branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m 'Add YourFeature'`
4. Push to the branch: `git push origin feature/YourFeature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

1)Akshat Lakhotiya
2)Lakshay Bindal
3)Ashish Kumar Yadav

- 🐙 GitHub: [@maheshwariakshat34](https://github.com/maheshwariakshat34)
- 🌐 Live Project: [Employee Attrition Predictor](https://employee-attrition-predictor-jxcw.onrender.com/signup)

---

> ⭐ If you found this project helpful, please consider giving it a star on GitHub!