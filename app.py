from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the cleaned car data
car = pd.read_csv("D:\\Aditya's Notes\\All Projects\\Cars Price Prediction\\Cleaned Car.csv")

# Load the pre-trained model (assuming it's saved as 'car_price_model.pkl')
model = pickle.load(open("D:\\Aditya's Notes\\All Projects\\Cars Price Prediction\\LinearRegressionModel.pkl", "rb"))

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()
    return render_template('index.html', companies=companies, years=year, fuel_type=fuel_type)

@app.route('/get_car_models', methods=['POST'])
def get_car_models():
    company = request.form.get('company')
    car_models = sorted(car[car['company'] == company]['name'].unique())
    return jsonify(car_models)

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kilo_driven = int(request.form.get('kilo_driven'))

    # Prepare the input data for the model
    input_data = pd.DataFrame({
        'name': [car_model],
        'company': [company],
        'year': [year],
        'kms_driven': [kilo_driven],
        'fuel_type': [fuel_type]
    })

    # Predict the car price using the model
    predicted_price = model.predict(input_data)[0]

    return render_template('index.html', predicted_price=round(predicted_price, 2), companies=sorted(car['company'].unique()),
                           years=sorted(car['year'].unique(), reverse=True), fuel_type=car['fuel_type'].unique())

if __name__ == "__main__":
    app.run(debug=True)
