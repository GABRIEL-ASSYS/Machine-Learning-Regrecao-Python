import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

data = pd.read_excel('expectativaVida.xlsx')
data = data.dropna()

print(data.head())
print(data.info())

X = data.drop('Life expectancy males', axis=1)
y = data['Life expectancy males']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelos = {
    'Regressão Linear': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'Árvore de Decisão': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor()
}

metricas = []

plt.figure(figsize=(12, 8))

for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)

    y_pred_treino = modelo.predict(X_train)
    y_pred_teste = modelo.predict(X_test)

    r2_treino = r2_score(y_train, y_pred_treino)
    r2_teste = r2_score(y_test, y_pred_teste)

    rmse_treino = np.sqrt(mean_squared_error(y_train, y_pred_treino))
    rmse_teste = np.sqrt(mean_squared_error(y_test, y_pred_teste))

    mae_treino = mean_absolute_error(y_train, y_pred_treino)
    mae_teste = mean_absolute_error(y_test, y_pred_teste)

    mape_treino = mean_absolute_percentage_error(y_train, y_pred_treino)
    mape_teste = mean_absolute_percentage_error(y_test, y_pred_teste)

    metricas.append({
        'Modelo': nome,
        'R2 Treino': r2_treino,
        'R2 Teste': r2_teste,
        'RMSE Treino': rmse_treino,
        'RMSE Teste': rmse_teste,
        'MAE Treino': mae_treino,
        'MAE Teste': mae_teste,
        'MAPE Treino': mape_treino,
        'MAPE Teste': mape_teste
    })

    plt.subplot(2, 3, len(metricas))
    plt.plot(y_train, label='Real - Treino', marker='o')
    plt.plot(y_pred_treino, label='Predito - Treino', marker='x')
    plt.plot(y_test, label='Real - Teste', marker='o')
    plt.plot(y_pred_teste, label='Predito - Teste', marker='x')
    plt.xlabel('Linhas da Base de Dados')
    plt.ylabel('Valores')
    plt.title(f'Comparação entre Valores Reais e Preditos por {nome}')
    plt.legend()

plt.tight_layout()
plt.show()

metricas_df = pd.DataFrame(metricas)
print(metricas_df)
