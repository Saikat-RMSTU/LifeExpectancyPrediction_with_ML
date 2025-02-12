
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
5. [Recommendations](#recommendations)
6. [Future Enhancements](#future-enhancements)
7. [References](#references)

---
## **Contributors and Instructor**
### **Contributors**
- **Saikat Das Roky**
  
  **Reg No**- 2018-15-18

    Department of CSE, RMSTU
  
- **Punam Kanungoe**
  
  **Reg No**- 2018-15-24

   Department of CSE, RMSTU

### **Instructor**
- **Md Mynoddin**
  
  **Assistant Professor**

    Department of CSE, RMSTU.

---
## **Project Overview**

This project serves as a valuable tool for governments, researchers, and organizations, offering deep insights into the key factors shaping global life expectancy trends. By leveraging machine learning, we move towards a future where data-driven strategies can significantly improve public health and societal well-being.

### **Objective**
The primary goal of this project is to develop a robust machine-learning pipeline to predict life expectancy using various health, demographic, and socioeconomic features. The model aims to uncover critical life expectancy factors, enabling data-driven decision-making to enhance public health strategies and resource allocation. Additionally, this project delves into feature importance analysis to provide actionable insights for policymakers, guiding them in prioritizing impactful interventions that improve overall well-being.

### **Significance**

Life expectancy is a key indicator of a nation's health, economic stability, and development progress. Predicting life expectancy and understanding its underlying contributors offer numerous benefits:
- **Policy Development:** Empowers governments and public health officials to allocate healthcare resources more effectively.
- **Health Equity:** Supports organizations in formulating policies to reduce health disparities across different populations.
- **Business Applications:** Assists businesses in identifying market opportunities in healthcare, insurance, and retirement planning sectors.
- **Scientific Insights:** Provides researchers with a deeper understanding of how socioeconomic, demographic, and health-related factors influence global health outcomes.

### **Dataset**
The dataset used in this project is the **Life Expectancy Dataset**, published by the WHO and the United Nations. It covers:
- **193 countries** over **16 years** (2000–2015).
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
  - **HIV/AIDS**: Deaths per 1000 live births HIV/AIDS (0-4 years)
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

Ensuring clean and structured data is crucial for accurate predictions. The preprocessing steps include:
- **Column Name Normalization:** Standardizing column names by converting them to lowercase and replacing spaces with underscores.
- **Handling Missing Values:**
  - Numerical features were imputed using median values to preserve data integrity.
  - Features with excessive missing values (e.g., Hepatitis B immunization) were dropped due to potential bias.
- **Categorical Encoding:**
  - Encoded categorical variables such as `country` and `status` using `LabelEncoder`.
- **Feature Scaling:**
  - Applied `MinMaxScaler` to ensure consistent feature scaling and improve model performance.

### **Exploratory Data Analysis (EDA)**

A thorough EDA was conducted to understand feature distributions and relationships:
- **Distribution Analysis:** Histograms and boxplots were used to examine data distribution and detect outliers.
- **Pairwise Relationships:** Scatter plots and pairplots were generated to identify interactions between features, such as life expectancy vs. GDP or HIV/AIDS prevalence.
- **Correlation Heatmap:** A correlation matrix was plotted to assess relationships between variables:
- **Negative Correlation:** Adult mortality and HIV/AIDS prevalence were found to be negatively correlated with life expectancy.
- **Positive Correlation:** Schooling years, GDP, and income composition index showed strong positive correlations with life expectancy.

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

Each model's performance was assessed using:
- **R² Score:** Measures how well the model fits the data.
- **Residual Analysis:** Examined error distributions to detect patterns or biases.
- **Feature Importance Analysis:** Used SHAP values and permutation importance to identify the most influential predictors.

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

---

## **Recommendations**

Based on the findings, the following policy recommendations are proposed:
- **Healthcare Infrastructure:** Strengthen healthcare facilities in low-income regions to improve access to medical care.
- **Disease Prevention Programs:** Prioritize campaigns targeting major health threats like HIV/AIDS through awareness and preventive measures.
- **Education & Awareness:** Expand access to quality education, particularly in underprivileged areas, as schooling is a crucial determinant of life expectancy.
- **Public Health Initiatives:** Promote healthier lifestyles and early screening for chronic diseases to reduce mortality rates and enhance quality of life.

---


## **Future Enhancements**

To further refine the model and broaden its applicability, the following enhancements are planned:
1. **Integrate Additional Datasets:** Incorporate economic and healthcare quality indices to improve predictions.
2. **Model Deployment:** Implement a Flask-based API for real-time life expectancy predictions.
3. **Ensemble Learning:** Combine multiple models using stacking techniques for improved accuracy.
4. **Automated Data Pipeline:** Develop an automated system for real-time data updates and preprocessing.


---



## **References**
- **WHO Life Expectancy Dataset**: [Kaggle](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)
- **Mahumud, R.A., Hossain, G., Hossain, R., Islam, N. and Rawal, L.,;, “Impact of Life Expectancy on Economic Growth and Health Care 
    Expenditures in Bangladesh.,” Universal Journal of Public Health, vol. 1, no. 4, pp. 180-186, 2013.**
- **Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.**
