# World Happiness Report: Machine Learning Analysis

![image](https://github.com/user-attachments/assets/d83bb320-a5ce-4222-8b1d-7d2235881dac)

## Project Overview
This project analyzes the World Happiness Report dataset to predict life satisfaction scores ("Happiness") based on socioeconomic factors. Using various machine learning techniques, we identify key drivers of happiness and build predictive models to quantify their impact.

## Dataset
**Source:** World Happiness Report 2018 (WHR2018Chapter2OnlineData.csv)  
**Target Variable:** `Life Ladder` (Happiness score)  
**Key Features:**
- Economic factors (Log GDP per capita, Gini index)
- Social factors (Social support, Freedom, Generosity)
- Health metrics (Healthy life expectancy)
- Governance metrics (Democratic Quality, Corruption perceptions)

## Key Steps

### 1. Data Preparation
- Handled missing values with mean imputation
- Renamed columns for clarity
- Winsorized outliers in Gini index
- Selected relevant features through correlation analysis

### 2. Models Implemented
1. **Linear Regression** (Baseline)
2. **Decision Tree Regressor** (with GridSearchCV tuning)
3. **Stacking Regressor** (Decision Tree + Linear Regression)
4. **Gradient Boosting Regressor**
5. **Random Forest Regressor**

### 3. Performance Comparison
| Model          | RMSE   | R² Score |
|----------------|--------|----------|
| Linear Regression | 0.534 | 0.781    |
| Decision Tree  | 0.508  | 0.801    |
| Stacking       | 0.471  | 0.829    |
| GBDT           | 0.408  | 0.872    |
| Random Forest  | **0.358** | **0.901** |

## Key Findings
- **Top 3 Happiness Predictors:**
  1. Social Support
  2. GDP per capita
  3. Healthy Life Expectancy
- Random Forest achieved best performance (RMSE = 0.358, R² = 0.901)
- Income inequality (Gini index) showed complex dual impact

## How to Run
1. Clone repository
2. Install requirements: `pip install -r requirements.txt`
3. Run Jupyter notebook: `jupyter notebook DefineAndSolveMLProblem.ipynb`

## Dependencies
- Python 3.9+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook

## Future Work
- Incorporate time-series analysis for year-over-year trends
- Experiment with neural networks
- Develop country-specific happiness prediction models
