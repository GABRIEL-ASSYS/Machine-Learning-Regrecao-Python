import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt

data = pd.read_excel('expectativaVida.xlsx')
print(data.head())