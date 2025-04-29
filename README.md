# 🧠 Customer Conversion Prediction – ML Project

A machine learning project designed to help a new-age insurance company predict whether a client is likely to subscribe to term insurance. The objective is to optimize telephonic marketing campaigns by targeting high-conversion customers, reducing cost, and improving efficiency.

---

## 📌 Problem Statement

The company uses telephonic marketing campaigns to reach potential customers. However, calls are resource-intensive, so it's crucial to identify customers with a higher likelihood of converting. The task is to build a **classification model** using historical marketing data to predict whether a customer will subscribe to term insurance.

---

## 🎯 Objectives

- Build and validate a predictive model to identify potential customers.
- Improve marketing efficiency by focusing efforts on likely converters.
- Use **F1-score** as the main evaluation metric for model performance.

---

## 🧰 Skills & Tools Applied

- **Domain**: Data Science / Machine Learning  
- **Techniques**: Classification, EDA, Feature Engineering, Model Evaluation  
- **Tools**: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

---

## 📁 Dataset Overview

The dataset includes customer demographics, contact history, and past campaign performance.

### 🔹 Features

| Feature         | Description |
|----------------|-------------|
| age            | Age of the client (numeric) |
| job            | Type of job |
| marital        | Marital status |
| educational_qual | Education level |
| call_type      | Contact communication type |
| day            | Last contact day of the month |
| mon            | Last contact month |
| dur            | Duration of last contact (in seconds) |
| num_calls      | Number of contacts performed in this campaign |
| prev_outcome   | Outcome of the previous marketing campaign (`unknown`, `other`, `failure`, `success`) |

### 🎯 Target

| Target Variable | Description |
|----------------|-------------|
| y              | Has the client subscribed to the insurance? (`yes` or `no`) |

---

## 🧪 Project Workflow

1. **Data Cleaning** – Handle missing values, correct data types.
2. **Exploratory Data Analysis** – Discover patterns, visualize distributions.
3. **Data Preprocessing** – Encode categorical variables, scale features.
4. **Model Building** – Train classification models (e.g., Logistic Regression, Random Forest, etc.).
5. **Model Validation** – Evaluate using F1-score.
6. **Feature Importance** – Interpret key drivers of customer conversion.
7. **Deployment (optional)** – Save the model for future use.

---

## 📊 Evaluation Metric

- **F1 Score**: Chosen for its balance between precision and recall in an imbalanced classification scenario.

---

## 📦 Deliverables

- 🧾 Cleaned and preprocessed dataset  
- 🧠 Trained ML model(s) with performance metrics  
- 📈 Visualizations and insights  
- 📄 Final project report (PDF or Notebook)


## 📎 References

- [Project Dataset (Google Drive Link)](https://drive.google.com/file/d/1BJ_Q8Q-kDRisAQyLltBQggeb0QmdWGZy/view?usp=sharing)

---

## 👨‍💻 Project By

**Yogeshwaran M**


