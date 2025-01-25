
# **Life Expectancy Prediction with Machine Learning**

## **Table of Contents**

1. [Project Overview](#project-overview)
   - [Objective](#objective)
   - [Significance](#significance)
   - [Dataset](#dataset)
   - [Key Features](#key-features)
   - [Technologies Used](#technologies-used)
2. [How to Run](#how-to-run)
   - [Setting Up Environment](#setting-up-environment)
   - [Running the Notebook](#running-the-notebook)
3. [Detailed Project Workflow](#detailed-project-workflow)
   - [Data Preprocessing](#data-preprocessing)
   - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Feature Engineering](#feature-engineering)
   - [Model Training](#model-training)
   - [Hyperparameter Tuning](#hyperparameter-tuning)
   - [Model Evaluation](#model-evaluation)
4. [Results and Insights](#results-and-insights)
5. [Future Enhancements](#future-enhancements)
6. [References](#references)

---

## **Project Overview**

### **Objective**
The goal of this project is to develop a machine learning pipeline that predicts life expectancy based on various health-related, demographic, and socioeconomic factors. By identifying the most significant contributors to life expectancy, this project provides actionable insights that can influence public health policies and socioeconomic strategies.

### **Significance**
Life expectancy serves as a critical indicator of overall public health and socioeconomic stability. Accurate predictions and insights into factors influencing life expectancy can help:
- Governments allocate healthcare resources effectively.
- Organizations design better policies to tackle public health challenges.
- Businesses strategize for emerging markets, focusing on healthcare, insurance, and retirement planning.

### **Dataset**
The dataset used in this project is the **Life Expectancy Dataset**, published by the WHO and the United Nations. It covers:
- **183 countries** over **16 years** (2000–2015).
- Health, demographic, and socioeconomic indicators, such as:
  - Life expectancy (target variable)
  - Adult mortality, infant deaths
  - BMI, schooling, GDP, immunization coverage
  - HIV/AIDS prevalence, thinness among children
- Available on [Kaggle](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who).

### **Key Features**
- **22 features**, categorized as:
  - **Target Variable**: Life expectancy (in years)
  - **Demographic**: Population, schooling, income composition
  - **Health Indicators**: BMI, HIV/AIDS prevalence, immunization coverage
  - **Economic**: GDP, health expenditure
  - **Mortality Rates**: Adult, infant, and under-five mortality
- Features were preprocessed, with unreliable columns excluded (e.g., population data) and missing values imputed.

### **Technologies Used**
- **Programming Language**: Python
- **Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn, SHAP
- **Notebook Environment**: Google Colab / Jupyter Notebook
- **Deployment Tools**: Flask, Docker

---

## **How to Run**

### **Setting Up Environment**

1. Clone the repository:
   ```bash
   git clone https://github.com/<username>/life_expectancy_prediction.git
   cd life_expectancy_prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset:
   Download the [Life Expectancy Dataset](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who) and save it as `data/life_expectancy_data.csv`.

### **Running the Notebook**

1. Open the notebook:
   ```bash
   jupyter notebook Life_Expectency_Prediction_with_Machine_Learning.ipynb
   ```

2. Follow the notebook's step-by-step structure for:
   - Data preprocessing
   - Exploratory Data Analysis
   - Model training and evaluation

3. To deactivate the virtual environment:
   ```bash
   deactivate
   ```

---

## **Detailed Project Workflow**

### **1. Data Preprocessing**
- **Normalize Column Names**: Converted column names to lowercase with underscores.
- **Handle Missing Values**:
  - Used median imputation for numeric features.
  - Dropped unreliable or MNAR features (e.g., hepatitis B coverage).
- **Encode Categorical Variables**: Transformed `country` and `status` using `LabelEncoder`.

### **2. Exploratory Data Analysis (EDA)**
- **Visualized Data Distribution**:
  - Histograms for all numeric columns.
  - Pairplots for key features like life expectancy, HIV/AIDS, BMI, and GDP.
- **Correlation Heatmap**:
  - Revealed strong correlations between mortality rates and life expectancy.
- **Feature Importance Analysis**:
  - Identified adult mortality, income composition of resources, and schooling as top predictors.

### **3. Feature Engineering**
- Created derived features, such as `bmi_to_hiv_ratio`, for better representation.
- Scaled numeric data using `MinMaxScaler` to normalize ranges.

### **4. Model Training**
- Trained and evaluated multiple regression models:
  - **Linear Regression**
  - **Random Forest Regressor**
  - **Gradient Boosting Regressor**
  - **Support Vector Regressor (SVR)**
  - **K-Nearest Neighbors (KNN)**
  - **Decision Tree Regressor**

### **5. Hyperparameter Tuning**
- Performed hyperparameter tuning using `GridSearchCV` for models like Random Forest, Gradient Boosting, and SVR.
- Used time-aware cross-validation (expanding window) to avoid data leakage.

### **6. Model Evaluation**
- Compared models using **R² scores**:
  - **Training R²**
  - **Testing R²**
- Evaluated feature importance using SHAP and permutation importance.

---

## **Results and Insights**

### **Best Model**
- **Random Forest Regressor**:
  - **Training R²**: 0.95
  - **Testing R²**: 0.92

### **Key Predictors**
- **Adult mortality** (negative correlation with life expectancy)
- **Income composition of resources**
- **Schooling**

### **Feature Importance**
- Socioeconomic and demographic features were more influential than immunization or health expenditure.

### **Recommendations**
- Governments should prioritize investments in education, employment, and childhood nutrition.
- Focus on reducing mortality rates to improve life expectancy.

---

## **Future Enhancements**
1. Incorporate additional datasets (e.g., global economic indices) for enhanced predictions.
2. Deploy the model via a Flask API for real-time predictions.
3. Implement ensemble methods (e.g., stacking) for improved accuracy.
4. Automate dataset preprocessing and feature engineering pipelines.

---

## **References**
- **WHO Life Expectancy Dataset**: [Kaggle](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)
- **SHAP Library Documentation**: [SHAP](https://shap.readthedocs.io/)
