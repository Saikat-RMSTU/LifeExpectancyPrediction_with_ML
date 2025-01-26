
# **Life Expectancy Prediction Using Machine Learning**

## **Table of Contents**

1. [Contributors and Instructor](#contributors-and-instructor)
2. [Project Overview](#project-overview)
   - [Objective](#objective)
   - [Significance](#significance)
   - [Dataset](#dataset)
   - [Key Features](#key-features)
   - [The Columns of the Dataset](#the-columns-of-the-dataset)
   - [Technologies Used](#technologies-used)
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
## **Contributors and Instructor**
### **Contributors**
- **Saikat Das Roky**
  
  **Reg No**- 2018-15-18
- **Punam Kanungoe**
  
  **Reg No**- 2018-15-24


### **Instructor**
- **Md Mynoddin**
  
  **Assistant Professor**


Department of CSE, RMSTU.

---
## **Project Overview**

### **Objective**
The primary goal of this project is to develop a robust machine learning pipeline to predict life expectancy using various health, demographic, and socioeconomic features. The model aims to provide insights into the critical factors influencing life expectancy, enabling data-driven decision-making to enhance public health strategies and resource allocation. This project also explores feature importance to guide policymakers on prioritizing impactful interventions.

### **Significance**
Life expectancy serves as a vital benchmark for the overall health, development, and well-being of a population. By predicting life expectancy and analyzing the underlying contributors, this project:
- Empowers governments to allocate healthcare resources more effectively.
- Supports organizations in formulating policies to combat health inequities.
- Helps businesses identify market opportunities in healthcare, insurance, and retirement planning.
- Provides insights into how socioeconomic, demographic, and health-related factors interact to shape global health outcomes.

### **Dataset**
The dataset used in this project is the **Life Expectancy Dataset**, published by the WHO and the United Nations. It covers:
- **183 countries** over **16 years** (2000–2015).
- Health, demographic, and socioeconomic indicators, such as:
  - Life expectancy (target variable)
  - Adult mortality, infant deaths
  - BMI, schooling, GDP, immunization coverage
  - HIV/AIDS prevalence, thinness among children

### **Key Features**
  - **Target Variable**: Life expectancy (in years)
  - **Demographic**: Population, schooling, income composition
  - **Health Indicators**: BMI, HIV/AIDS prevalence, immunization coverage
  - **Economic**: GDP, health expenditure
  - **Mortality Rates**: Adult, infant, and under-five mortality

### **The Columns of the Dataset**
  - **Country**: Country
  - **Year**: Year
  - **Status**: Country Developed or Developing status
  - **Life expectancy**: Life expectancy in age
  - **Adult Mortality**: Adult Mortality Rates of both sexes (probability of dying between 15 and 60 years per 1000 population)
  - **infant deaths**: Number of Infant Deaths per 1000 population
  - **Alcohol** : Alcohol, recorded per capita (15+) consumption (in litres of pure alcohol) -percentage expenditure: Expenditure on 
                  health as a percentage of Gross Domestic Product per capita(%)
  - **Hepatitis B**: Hepatitis B (HepB) immunization coverage among 1-year-olds (%)
  - **Measles**: Measles - number of reported cases per 1000 population
  - **BMI**: Average Body Mass Index of entire population
  - **under-five deaths**: Number of under-five deaths per 1000 population
  - **Polio**: Polio (Pol3) immunization coverage among 1-year-olds (%)
  - **Total expenditure**: General government expenditure on health as a percentage of total government expenditure (%)
  - **Diphtheria**: Diphtheria tetanus toxoid and pertussis (DTP3) immunization coverage among 1-year-olds (%)
  - **HIV/AIDS**: Deaths per 1 000 live births HIV/AIDS (0-4 years)
  - **GDP**: Gross Domestic Product per capita (in USD)
  - **Population**: Population of the country
  - **thinness 1-19 years**: Prevalence of thinness among children and adolescents for Age 10 to 19 (%)
  - **thinness 5-9 years**: Prevalence of thinness among children for Age 5 to 9(%)
  - **Income composition of resources**: Human Development Index in terms of income composition of resources (index ranging from 0 to 1)
  - **Schooling**: Number of years of Schooling(years)

### **Technologies Used**
- **Programming Language**: Python
- **Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn, SHAP
- **Notebook Environment**: Google Colab / Jupyter Notebook

---

## **Detailed Project Workflow**

### **Data Preprocessing**
- **Column Name Normalization**: Simplified column names by converting them to lowercase and replacing spaces with underscores.
- **Handling Missing Values**:
  - Imputed missing values for numeric features using median imputation.
  - Dropped unreliable columns such as Hepatitis B due to high missing values and suspected bias.
- **Categorical Encoding**:
  - Transformed `country` and `status` columns using `LabelEncoder`.
- **Feature Scaling**:
  - Scaled numeric features using `MinMaxScaler` to standardize the range for consistent model performance.

### **Exploratory Data Analysis (EDA)**
- Analyzed the distributions of key numeric variables using histograms.
- Visualized pairwise relationships among features using pairplots (e.g., life expectancy vs. HIV/AIDS prevalence, BMI, and GDP).
- Generated a correlation heatmap to uncover strong relationships, such as:
  - Negative correlation between adult mortality and life expectancy.
  - Positive correlation between income composition of resources, schooling, and life expectancy.

### **Feature Engineering**
- Engineered a derived feature: `bmi_to_hiv_ratio` for better representation of the relationship between BMI and HIV/AIDS prevalence.
- Focused on reducing multicollinearity among features.

### **Model Training**
Trained and evaluated the following regression models:
- **Linear Regression**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **Support Vector Regressor (SVR)**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree Regressor**

### **Hyperparameter Tuning**
- Used `GridSearchCV` to optimize hyperparameters for models like Random Forest, Gradient Boosting, and SVR.
- Applied time-aware cross-validation (expanding window method) to prevent data leakage and ensure realistic performance evaluation.

### **Model Evaluation**
- Compared models using:
  - **Training R²**: How well the model fits the training data.
  - **Testing R²**: How well the model generalizes to unseen data.
- Analyzed residual distributions for model validation.
- Conducted feature importance analysis using SHAP and permutation importance.

---

## **Results and Insights**

### **Best Model**
- **Gradient Boosting Regressor**:
  - **Training R²**: 0.9875
  - **Testing R²**: 0.9593

### **Key Predictors**
- **HIV/AIDS Prevalence**: Strongest negative impact on life expectancy.
- **Income Composition of Resources**: Indicates the effectiveness of income utilization for human development.
- **Adult Mortality**: Significant negative correlation with life expectancy.

### **Feature Importance**
- Socioeconomic and demographic features like schooling, GDP, and income composition emerged as the most influential predictors.
- Health indicators, such as immunization coverage, had relatively lower importance due to widespread implementation globally.

### **Recommendations**
- Governments should focus on:
  - Reducing HIV/AIDS prevalence.
  - Improving education systems to ensure equitable access to quality schooling.
  - Increasing investments in health infrastructure and income equality.

---

## **Future Enhancements**
1. Incorporate additional datasets (e.g., global economic indices) for enhanced predictions.
2. Deploy the model via a Flask API for real-time predictions.
3. Implement ensemble methods (e.g., stacking) for improved accuracy.
4. Automate dataset preprocessing and feature engineering pipelines.

---


## **References**
- **WHO Life Expectancy Dataset**: [Kaggle](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)
