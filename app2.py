import streamlit as st
import pandas as pd
import pickle

# Load the cleaned car data
car = pd.read_csv("D:\\Aditya's Notes\\All Projects\\Cars Price Prediction\\Cleaned Car.csv")

# Load the pre-trained model
model = pickle.load(open("D:\\Aditya's Notes\\All Projects\\Cars Price Prediction\\LinearRegressionModel.pkl", "rb"))

# Streamlit app layout
st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("Car Price Predictor")

# Sidebar for user inputs
st.sidebar.header("User Input")

# Select the company
companies = sorted(car['company'].unique())
selected_company = st.sidebar.selectbox("Select The Company", companies)

# Select the car model
car_models = sorted(car[car['company'] == selected_company]['name'].unique())
selected_car_model = st.sidebar.selectbox("Select The Car Model", car_models)

# Select the year of purchase
year = sorted(car['year'].unique(), reverse=True)
selected_year = st.sidebar.selectbox("Select The Year of Purchase", year)

# Select the fuel type
fuel_types = sorted(car['fuel_type'].unique())
selected_fuel_type = st.sidebar.selectbox("Select The Fuel Type", fuel_types)

# Input for kilometers driven
kilo_driven = st.sidebar.number_input("Enter the Number of Kilometers Travelled", min_value=0)

# Predict button
if st.sidebar.button("Predict Price"):
    # Prepare the input data for the model
    input_data = pd.DataFrame({
        'name': [selected_car_model],
        'company': [selected_company],
        'year': [selected_year],
        'kms_driven': [kilo_driven],
        'fuel_type': [selected_fuel_type]
    })

    # Predict the car price using the model
    predicted_price = model.predict(input_data)[0]
    st.sidebar.success(f"The predicted price for the selected car is: ₹{round(predicted_price, 2)}")

# Styling to mimic Bootstrap aesthetics
st.markdown(
    """
    <style>
    body {
        background-color: #f8f9fa;
        font-family: 'Arial', sans-serif;
        color: #333;
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    .stSelectbox, .stNumberInput {
        background-color: #f1f1f1;
        border: 1px solid #ced4da;
        border-radius: 5px;
        padding: 10px;
        font-size: 1rem;
    }
    </style>
    """, unsafe_allow_html=True
)

# Additional information can be added here
st.markdown("""
### Instructions
1. Select the company from the dropdown.
2. Choose the car model based on the selected company.
3. Select the year of purchase and fuel type.
4. Enter the number of kilometers traveled.
5. Click on "Predict Price" to see the predicted car price.
""")
