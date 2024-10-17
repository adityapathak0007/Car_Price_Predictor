# 🚗 Car Price Prediction App

This **Car Price Prediction App** allows users to predict the price of used cars based on various factors such as the car's brand, model, year of purchase, fuel type, and kilometers driven. The prediction is powered by a pre-trained linear regression model.

## 🛠️ Features

- 🔍 Predict the price of a used car based on:
  - Car company (brand)
  - Car model
  - Year of purchase
  - Fuel type (Petrol, Diesel, etc.)
  - Kilometers driven
- 🌐 Responsive web interface built using **Streamlit**.
- 📊 Real-time price predictions based on user input.
- 💻 Simple and clean UI for an intuitive user experience.

## 🧠 How It Works

The app collects input from the user and calculates the predicted car price using a **linear regression model**. The model was trained on a dataset containing features like car brand, model, year, fuel type, kilometers driven, and price.

## 📄 Data

The model is trained on a dataset (`Cleaned Car.csv`) with the following key columns:
- `name`: The name of the car model.
- `company`: The car manufacturer or brand.
- `year`: The year the car was purchased.
- `kms_driven`: The number of kilometers the car has been driven.
- `fuel_type`: The type of fuel the car uses (Petrol, Diesel, etc.).
- `Price`: The actual selling price of the car (used for model training).

The dataset is cleaned and pre-processed to remove outliers and missing values before training the model.


## ⚙️ Installation

To run this app locally, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/adityapathak0007/car-price-prediction-app.git

2. Navigate to the project directory:
   ```bash
   cd car-price-prediction-app

3. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt

### Ensure the following files are present in the root directory:

1. **Cleaned Car.csv**: This is the cleaned dataset used to predict car prices. Ensure it is placed in the root directory for the app to access.
   
   - You can download the cleaned dataset or prepare it using the original data by following the data cleaning steps mentioned in the code.
   - The dataset contains columns like `name`, `company`, `year`, `kms_driven`, `fuel_type`, and `Price`.

2. **LinearRegressionModel.pkl**: This is the pre-trained machine learning model that will be used for predicting car prices based on user input.

   - The model has been serialized using **Pickle**. Make sure the file is located in the root directory so the app can load it when running predictions.
   - If you want to train the model yourself, refer to the code provided to train and save the model using Scikit-learn.


---

### 6. Files
## 📁 Files

- **app2.py**: The main Python script for the Streamlit web app.
- **Cleaned Car.csv**: The dataset used for predictions.
- **LinearRegressionModel.pkl**: The saved model used to predict car prices.
- **requirements.txt**: The list of Python packages required to run the app.
- **README.md**: Documentation file (this file).

## 🎯 Usage

1. Select the car company (e.g., Maruti, Honda) from the dropdown.
2. Choose the car model based on the selected company.
3. Enter the year of purchase and select the fuel type.
4. Input the number of kilometers the car has traveled.
5. Click the "Predict Price" button to get the estimated price of the car.

The predicted car price will be displayed on the sidebar.


## 🛠️ Technologies Used

- **Python**: The core programming language used to build the app.
- **Streamlit**: A fast and interactive way to create web apps with Python.
- **Pandas**: Used for data manipulation and analysis.
- **Scikit-learn**: Used to build and save the machine learning model.
- **Pickle**: Used to serialize and deserialize the trained model.


