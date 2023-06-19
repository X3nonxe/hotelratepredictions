import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import joblib

# Load data train_features.csv, train_labels.csv, dan test_features.csv
train_features = pd.read_csv('dataset/train_features.csv')
train_labels = pd.read_csv('dataset/train_label.csv')
test_features = pd.read_csv('dataset/test_feature.csv')

# Menggabungkan data train_features dan train_labels berdasarkan indeks
train_data = pd.merge(train_features, train_labels, left_index=True, right_index=True)

# Mengkonversi kolom 'Price' menjadi numerik
train_data['Price'] = train_data['Price'].str.replace('Rp', '').str.replace(',', '').str.replace('avg/night', '').astype(float)

# Membersihkan kolom 'rating' dari karakter non-numerik
train_data['rating'] = train_data['rating'].str.extract('(\d+\.\d+|\d+)').astype(float)

# Memisahkan fitur dan label pada data train
X_train = train_data[['facilities', 'rating', 'location']]
y_train = train_data['Price']

# Memisahkan kolom 'Price' pada data test
X_test = test_features[['facilities', 'rating', 'location']]

# Membersihkan kolom 'rating' dari karakter non-numerik pada data uji (X_test)
X_test['rating'] = X_test['rating'].str.extract('(\d+\.\d+|\d+)').astype(float)

# Praproses data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['rating']),
        ('cat', categorical_transformer, ['facilities', 'location'])
    ])

# Membangun model Random Forest Regressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Definisikan grid parameter untuk penyetelan parameter
param_grid = {
    'regressor__n_estimators': [50, 100],
    'regressor__max_depth': [None, 5, 10]
}

# Melakukan penyetelan parameter (hyperparameter tuning) dengan stratified k-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=cv)
grid_search.fit(X_train, y_train)

# Memilih model terbaik setelah penyetelan parameter
best_model = grid_search.best_estimator_

# Melakukan praproses dan melatih model terbaik
best_model.fit(X_train, y_train)

# Simpan model terbaik ke file
joblib.dump(best_model, 'model/best_model.pkl')

# Prediksi dengan data X_test menggunakan model yang telah disimpan
loaded_model = joblib.load('model/best_model.pkl')
y_pred = loaded_model.predict(X_test)

# Menyimpan hasil prediksi dalam DataFrame
results = pd.DataFrame({'ID': test_features['ID'], 'Price': y_pred})

# Mengubah tipe data kolom 'Price' menjadi integer
results['Price'] = results['Price'].astype(int)

# Menyimpan DataFrame ke file CSV
results.to_csv('predictions.csv', index=False)

# Menampilkan beberapa hasil prediksi
print("Hasil Prediksi:")
for i, pred in enumerate(y_pred[:5]):
    print("Data ke-{}: Prediction: Rp.{:.0f}".format(i+1, pred))

# Menghitung metrik evaluasi
y_train_pred = loaded_model.predict(X_train)
mae = mean_absolute_error(y_train, y_train_pred)
mse = mean_squared_error(y_train, y_train_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_train, y_train_pred)
explained_variance = explained_variance_score(y_train, y_train_pred)

# Menampilkan metrik evaluasi
print("Mean Absolute Error (MAE): Rp.{:.0f}".format(mae))
print("Mean Squared Error (MSE): Rp.{:.0f}".format(mse))
print("Root Mean Squared Error (RMSE): Rp.{:.0f}".format(rmse))
print("R^2 Score: {:.2f}%".format(r2 * 100))
print("Explained Variance Score: {:.2f}%".format(explained_variance * 100))

# Visualisasi distribusi harga (target variable)
plt.figure(figsize=(8, 6))
sns.histplot(train_data['Price'], kde=True)
plt.title('Distribution of Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Visualisasi hubungan antara rating dan harga
plt.figure(figsize=(8, 6))
sns.scatterplot(data=train_data, x='rating', y='Price')
plt.title('Rating vs Price')
plt.xlabel('Rating')
plt.ylabel('Price')
plt.show()

# Visualisasi harga sebenarnya vs harga yang diprediksi
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_train_pred)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()
