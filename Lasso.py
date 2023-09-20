import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso 
import matplotlib.pyplot as plt

data = pd.read_excel('expectativaVida.xlsx')
print(data.head())

x = data.drop('Life expectancy males', axis=1)
y = data['Life expectancy males']

lasso_model = Lasso(alpha=1.0)
lasso_model.fit(x, y)

y_pred = lasso_model.predict(x)

plt.figure(figsize=(10, 6))
plt.plot(y, label='Valores reais', color='blue')
plt.plot(y_pred, label='Valores preditos', color='orange')
plt.xlabel('Índice')
plt.ylabel('Valores')
plt.title('Comparação entre Valores Reais e Preditos (Regressao Lasso)')
plt.legend()
plt.show()