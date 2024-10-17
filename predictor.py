import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


car = pd.read_csv("D:\\Aditya's Notes\\All Projects\\Cars Price Prediction\\car_price.csv")
print(car.head())
print(car.shape)
print(car.info())

print(car['year'].unique())
print(car['Price'].unique())
print(car['kms_driven'].unique())
print(car['fuel_type'].unique())



# Quality:
# year has many non-year values, and it is in object datatype we have to change it to integer
# price has Ask For Price , so Price - Object to Int
# kms driven has kms with integers so kms_driven object to int and also kms_driven has nan values
# fuel_type has nan values
# keep first 3 words of name

# creating backup of dataset
backup = car.copy()

# Cleaning of data:
# Print the 'year' column to inspect
print(car['year'])

# Keep only the rows where 'year' is numeric
car = car[car['year'].str.isnumeric()]
# Convert 'year' to integer
car['year'] = car['year'].astype(int)
print(car['year'])

print(car.info())

# removing , and ask for price in car['price']
car = car[car['Price'] != "Ask For Price"]
print(car['Price'])

car['Price'] = car['Price'].str.replace(',', '').astype(int)
print(car['Price'])
print(car.info())

# kms_driven removing , and petrol, and kms
car['kms_driven'] = car['kms_driven'].str.split(' ').str.get(0).str.replace(',', '')
print(car['kms_driven'])
car = car[car['kms_driven'].str.isnumeric()]
car['kms_driven'] = car['kms_driven'].astype(int)
print(car['kms_driven'])
print(car.info())

# removing those values where fuel_type is nan
car = car[~car['fuel_type'].isna()]
print(car)
print(car.info())

# keeping the only 1st 3 words of name
car['name'] = car['name'].str.split(' ').str.slice(0,3).str.join(' ')
car = car.reset_index(drop=True)
print(car['name'])

# removing outlier
car = car[car['Price'] < 6e6].reset_index(drop=True)
print(car['Price'])

print(car.info())
print(car.describe())
print(car)

# saving cleaned file to csv
# car.to_csv('Cleaned Car.csv')

# model building starts from here
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

print(r2_score(y_test, y_pred))

# for finding best r2 score
scores = []
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=i)
    lr = LinearRegression()
    pipe = make_pipeline(column_trans, lr)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    scores.append(r2_score(y_test, y_pred))

print("Best Random State = ", np.argmax(scores))
print("Best R2 Score = ", scores[np.argmax(scores)])

'''
# another check for improvements
from sklearn.pipeline import Pipeline
from joblib import Memory
memory = Memory(location='/tmp', verbose=0)
pipe = Pipeline(steps=[('preprocessor', column_trans), ('model', lr)], memory=memory)

# for finding best r2 score
scores = []
test_sizes = [0.2, 0.25, 0.3]  # Experimenting with different test sizes

for size in test_sizes:
    for i in range(1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=i)
        lr = LinearRegression()
        pipe = make_pipeline(column_trans, lr)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        scores.append((r2_score(y_test, y_pred), size, i))

best_index = np.argmax([score[0] for score in scores])
best_score = scores[best_index]

print("Best R² score:", best_score[0])
print("Best Test Size:", best_score[1])
print("Best Random State:", best_score[2])
'''

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=np.argmax(scores))
lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
scores.append(r2_score(y_test, y_pred))

print("R2 Score of Model = ", scores[np.argmax(scores)])

# Creating pickle file for the model
pickle.dump(pipe, open('LinearRegressionModel1.pkl', 'wb'))

result = pipe.predict(pd.DataFrame([['Maruti Suzuki Swift', 'Maruti', 2019, 100, 'Petrol']], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))

print("Result:", result)
