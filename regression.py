from palmerpenguins import load_penguins
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import patsy

df = load_penguins()
penguins=df.dropna()

model=LinearRegression()
#pasty를 사용하여 수식으로 상호작용 항 생성
#0+는 절편 제거함
#종속변수 ~ 독립변수1 + 독립변수2
# 아래식에서 ~0 을 없애버리면 인터셉트 1이 생긴다.
formula = 'bill_depth_mm ~ 0 + bill_length_mm * body_mass_g * flipper_length_mm * species'
y, x = patsy.dmatrices(formula, penguins, return_type = 'dataframe')

x = x.iloc[:,1:]
#"Adelie", 몸무게 450g, 날개길이 15cm, 부리길이 10.5cm
model.fit(x, y)

model.coef_
model.intercept_