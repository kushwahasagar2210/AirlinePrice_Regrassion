# AirlinePrice_Regrassion
Flight Price Prediction using Machine Learning. This project predicts airline ticket prices based on features like airline, source, destination, and travel date. Techniques used include data preprocessing, regression models, and hyperparameter tuning. Optional deployment using Flask for real-time predictions.
# ✈️ Flight Price Prediction using Machine Learning

## 📋 Project Overview

This project involves predicting airline flight prices using machine learning models. By analyzing various features like airline, source, destination, and date of journey, we build a model that helps estimate flight fares. The project utilizes regression models, and different techniques like data preprocessing, feature extraction, and hyperparameter tuning.

## 🛠️ Key Features

- **Data Preprocessing**: Handling missing values, feature extraction, and categorical encoding.
- **Exploratory Data Analysis (EDA)**: Visualizations to explore relationships between features and prices.
- **Model Building**: Using regression models such as Linear Regression, Random Forest, and Gradient Boosting.
- **Evaluation**: Assessing model performance using R² score, MAE, and MSE.
- **Hyperparameter Tuning**: Optimizing model performance through grid search and cross-validation.
- **Deployment**: Optional model deployment using Flask for predictions.

## 🚀 Technologies Used

- **Python** 🐍
- **Pandas** for data manipulation
- **NumPy** for numerical operations
- **Matplotlib** & **Seaborn** for data visualization
- **Scikit-learn** for machine learning algorithms
- **Flask** (Optional for deployment)

## 📂 Dataset

The dataset used for this project is available on [Kaggle](https://www.kaggle.com/nikhilmittal/flight-fare-prediction-mh). It contains the following features:

- **Airline**: Name of the airline
- **Date of Journey**: Date on which the journey was scheduled
- **Source**: Starting point of the journey
- **Destination**: Ending point of the journey
- **Route**: The route taken by the flight
- **Duration**: Total time taken by the flight
- **Price**: Ticket price

## ⚙️ Steps Performed

1. **Data Preprocessing**:
   - Converted `Date_of_Journey` to day and month features.
   - Encoded categorical variables like `Airline`, `Source`, and `Destination`.
   - Removed irrelevant columns like `Route` and `Additional_Info`.

2. **Exploratory Data Analysis (EDA)**:
   - Visualized the distribution of flight prices.
   - Examined correlations between features using heatmaps.

3. **Model Building**:
   - Built models using **Linear Regression**, **Random Forest**, and **Gradient Boosting**.
   - Split the dataset into training and testing sets.

4. **Model Evaluation**:
   - Evaluated the model using **MAE**, **MSE**, and **R² score**.
   - Random Forest provided the best results after hyperparameter tuning.

5. **Deployment** (Optional):
   - Created a Flask web app to predict flight prices based on user input.

  # 🎯 Results
  - RandomForestRegressor have maximum R2Score and choose as best model.
  - The deployment allows users to predict flight prices based on their inputs.
  # 📝 Future Work
  - Adding more advanced models like XGBoost and LightGBM.
  - Expanding the dataset to include more cities and airlines.
  - Improving the deployment interface with more interactive features.
