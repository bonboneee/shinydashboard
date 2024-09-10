from matplotlib import pyplot as plt
import seaborn as sns
from palmerpenguins import load_penguins
import pandas as pd

df = load_penguins()

# !pip install scikit-learn
from sklearn.linear_model import LinearRegression

model = LinearRegression()
penguins=df.dropna()

penguins_dummies = pd.get_dummies(
    penguins, 
    columns=['species'],
    drop_first=True
    )

# x와 y 설정
x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]]
y = penguins_dummies["bill_depth_mm"]

# 모델 학습
model.fit(x, y)

model.coef_
model.intercept_

regline_y=model.predict(x)

x=pd.DataFrame({
    'bill_length_mm': 15.0,
    'species': pd.Categorical(['Adelie'], 
                                categories=['Adelie', 'Chinstrap', 'Gentoo'])
    })
x = pd.get_dummies(
    x, 
    columns=['species'],
    drop_first=True
    )
y_hat=model.predict(x)
y_hat