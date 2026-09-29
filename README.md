  # 🩺 Multiple Disease Prediction System

A machine learning–based application that predicts the likelihood of multiple diseases using patient clinical data. The project uses independent supervised learning models for different diseases and integrates them into a unified prediction system.

## 📌 Project Overview

The main objective of this project is to demonstrate the complete **machine learning lifecycle**, including:

* Data preprocessing
* Data cleaning and missing-value handling
* Feature engineering
* Feature selection
* Model training
* Model evaluation
* Model integration
* Deployment-ready application development

### Diseases Covered

| Disease                       | Example Clinical Features                    |
| ----------------------------- | -------------------------------------------- |
| 🩸 **Diabetes Mellitus**      | Glucose, BMI, Insulin, Age                   |
| ❤️ **Heart Disease**          | Blood Pressure, Cholesterol, Heart Rate, Age |
| 🧪 **Chronic Kidney Disease** | Blood Urea, Creatinine, Hemoglobin, Albumin  |

---

## 📊 Dataset Description

The project uses structured healthcare datasets stored in **CSV format**. Each disease has its own dataset to allow disease-specific preprocessing and modeling.

### Diabetes Dataset

Key features include:

* Glucose level
* BMI
* Insulin
* Age
* Blood pressure
* Other clinical measurements

### Heart Disease Dataset

Key features include:

* Blood pressure
* Cholesterol
* Maximum heart rate
* Age
* Chest-pain related attributes
* Other cardiovascular indicators

### Chronic Kidney Disease Dataset

Key features include:

* Blood urea
* Serum creatinine
* Hemoglobin
* Albumin
* Blood pressure
* Other kidney-related clinical measurements

---

## 🔄 Machine Learning Pipeline

The project follows a standard supervised machine learning workflow:

```text
Dataset
   ↓
Data Loading & Exploration
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Feature Encoding & Scaling
   ↓
Feature Selection
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Selection
   ↓
Application Integration
```

---

## 🤖 Models Implemented

Multiple supervised learning algorithms are evaluated for each disease prediction task.

### 1. Logistic Regression

Used as a classification model for predicting disease presence based on clinical features.

**Advant**
