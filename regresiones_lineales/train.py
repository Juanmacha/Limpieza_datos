import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


df = pd.read_csv(r"C:\Users\juanm\Documents\Learning\precios_casa.csv", header=0)

# Separar manualmente si todo vino en una sola columna
if df.shape[1] == 1:
    df = df.iloc[:,0].str.split(",", expand=True)
    df.columns = ["size","bedrooms","age","price"]

# Convertir a numérico
df["size"] = pd.to_numeric(df["size"], errors="coerce")
df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce")
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")

print(df.head())
print(df.info())



#Limpieza de datos 
#Convertir errores de texto a NaN
df["size"] = pd.to_numeric(df['size'], errors='coerce')
df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')

#Rellenar valores faltantes
df['size'].fillna(df['size'].mean(), inplace=True)
df['bedrooms'].fillna(df['bedrooms'].mode()[0], inplace=True)
df['age'].fillna(df['age'].mean(), inplace=True)
df['price'].fillna(df['price'].mean(), inplace=True)

#Eliminar duplicados
#df.drop_duplicates(inplace=True)

#Eliminar outliers
#df = df[(df['size'] > 100) & (df['size'] < 10000)]
#df = df[(df['price'] > 10000) & (df['price'] < 1000000)]
#df = df[(df['bedrooms'] > 0) & (df['bedrooms'] < 20)]
#df = df[(df['age'] >= 0) & (df['age'] < 100)]

print(df.info())
print(df.describe())
print(df.isnull().sum())
df.to_csv('precios_casa_limpio.csv', index=False)

#Separar X e y
X = df[['size', 'bedrooms', 'age']]
y = df['price']
print(X.head())
print(y.head())

#Dividir en conjunto de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
print(X_train.head())
print(y_train.head())

# --- MODELADO (CRISP-DM) ---
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# EVALUACIÓN
print("MSE:", mean_squared_error(y_test, y_pred))
print("R^2:", r2_score(y_test, y_pred))

# Mostrar ecuación
print("Intercepto (b0):", model.intercept_)
print("Coeficientes (b1, b2, b3):", model.coef_)
print(f"Ecuación del modelo: price = {model.intercept_:.2f} "
      f"+ ({model.coef_[0]:.2f} * size) "
      f"+ ({model.coef_[1]:.2f} * bedrooms) "
      f"+ ({model.coef_[2]:.2f} * age)")

# Guardar modelo y columnas
joblib.dump(model, 'modelo_regresion_lineal.pkl')
joblib.dump(X.columns.tolist(), 'columnas_modelo.pkl')