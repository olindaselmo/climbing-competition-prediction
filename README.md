# Climbing Competition Prediction (Machine Learning Project)

This project predicts the top climbers in international competitions using machine learning models.

The model is trained on historical competition results and athlete characteristics to estimate the probability that an athlete finishes in the top 10 of a discipline.

## Dataset

The dataset contains historical climbing competition results and athlete information.

Main features include:
- age
- height
- arm span
- discipline
- gender
- sport class
- competition season
- previous rankings

Two datasets are used:
data/
cleaned_athlete_information.csv
vrairesultWC.csv

## Methods

Several machine learning models were tested:

- Random Forest
- XGBoost
- Logistic Regression

Key steps of the pipeline:

1. Data cleaning and preprocessing
2. Feature engineering
3. Encoding categorical variables (OneHotEncoder)
4. Model training and hyperparameter tuning (GridSearchCV)
5. Model evaluation
6. Prediction of top athletes for future competitions

## Results

The models predict the probability of finishing in the top 10 for each athlete.

Example outputs:
results/
gridtop10_hommes_2026.csv
gridtop10_femmes_2026.csv
top10_hommes_XGB_2026.csv
top10_femmes_XGB_2026.csv

These files contain predicted rankings of the top athletes by discipline.

## Visualization

Visualization scripts generate bar plots showing the predicted top athletes per discipline.

## Tools

Python  
Pandas  
NumPy  
Scikit-learn  
XGBoost  
Matplotlib  
Seaborn