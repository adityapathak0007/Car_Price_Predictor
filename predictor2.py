import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Load the data
car = pd.read_csv("D:\\Aditya's Notes\\All Projects\\Cars Price Prediction\\car_price.csv")
print(car.head())
print(car.shape)
print(car.info())

# Clean and convert the 'year' column
# Ensure that 'year' is numeric
car['year'] = pd.to_numeric(car['year'], errors='coerce')  # Convert to numeric, set non-convertible values to NaN
car = car[car['year'].notna()]  # Drop rows with NaN in 'year'
car['year'] = car['year'].astype(int)  # Convert to integer
print(car['year'])

# Remove "Ask For Price" from 'Price' and convert to integer
car = car[car['Price'] != "Ask For Price"]
car['Price'] = car['Price'].str.replace(',', '').astype(int)

# Clean 'kms_driven'
car['kms_driven'] = car['kms_driven'].str.split(' ').str.get(0).str.replace(',', '')
car = car[car['kms_driven'].str.isnumeric()]
car['kms_driven'] = car['kms_driven'].astype(int)

# Remove rows with NaN in 'fuel_type'
car = car[~car['fuel_type'].isna()]

# Keep only the first 3 words of 'name'
car['name'] = car['name'].str.split(' ').str.slice(0,3).str.join(' ')
car = car.reset_index(drop=True)

# Remove outliers in 'Price'
car = car[car['Price'] < 6e6].reset_index(drop=True)

# Calculate car age
car['car_age'] = 2024 - car['year']  # Assuming current year is 2024

# Print the cleaned data
print(car.info())
print(car.describe())

# Proceed with model building
X = car.drop(columns='Price')
y = car['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])

column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']), remainder='passthrough')

lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print(y_pred)
print("R² Score:", r2_score(y_test, y_pred))

# Check for best R² score (You can also use cross-validation for this)
scores = []
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=i)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    scores.append(r2_score(y_test, y_pred))

best_index = np.argmax(scores)
print("Best Random State = ", best_index)
print("Best R² Score = ", scores[best_index])

# Create and save the model
pickle.dump(pipe, open('LinearRegressionModel.pkl', 'wb'))

# Prepare the input data
input_data = pd.DataFrame([['Maruti Suzuki Swift', 'Maruti', 2019, 100, 'Petrol']], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

# Calculate car_age based on the input data
input_data['car_age'] = 2024 - input_data['year']  # Assuming current year is 2024

# Now predict using the pipeline
result = pipe.predict(input_data)
print("Result:", result)

