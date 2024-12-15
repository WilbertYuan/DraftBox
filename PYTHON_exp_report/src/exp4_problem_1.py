import pandas as pd
import numpy as np
from keras import models
from keras import layers
from sklearn.preprocessing import OneHotEncoder, StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

data = pd.read_csv("data/experiment_4/题目1数据.txt", delim_whitespace=True,header = 0,names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'])
data = data.replace('?', np.nan)
data = data.dropna()

data['mpg'] = data['mpg'].astype(float)
data['cylinders'] = data['cylinders'].astype(int)
data['displacement'] = data['displacement'].astype(float)
data['horsepower'] = data['horsepower'].astype(float)
data['weight'] = data['weight'].astype(float)
data['acceleration'] = data['acceleration'].astype(float)
data['model_year'] = data['model_year'].astype(int)
data['origin'] = data['origin'].astype(int)

encoder = OneHotEncoder(sparse=False, drop='first')
origin_encoded = pd.DataFrame(encoder.fit_transform(data[['origin']]))
origin_encoded.index = data.index


X = pd.concat([data.drop(['mpg', 'car_name', 'origin'], axis=1).reset_index(drop=True), origin_encoded.reset_index(drop=True)], axis=1)
y = data['mpg'].reset_index(drop=True)

X.columns = X.columns.astype(str)

# scaler = StandardScaler()
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(512, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), verbose=1)

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs',style="italic")
plt.ylabel('Mean Squared Error (MSE)',style="italic")
plt.title('Training and Validation Loss',fontsize=16,style="oblique")
plt.legend()
plt.grid(True)
plt.savefig('img/problem4-1.png', dpi=300)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R^2 Score: {r2}")
print(f"Mean Squared Error (MSE): {mse}")
plt.show()



