import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
import matplotlib.pyplot as plt
penguins = load_penguins()
penguins.head()

# 펭귄 분류 문제
# y: 펭귄의 종류
# x1: bill_length_mm (부리 길이)
# x2: bill_depth_mm (부리 깊이)

df=penguins.dropna()
df=df[["species", "bill_length_mm", "bill_depth_mm"]]
df=df.rename(columns={
    'species': 'y',
    'bill_length_mm': 'x1',
    'bill_depth_mm': 'x2'})
df




#x1 기준으로 최적 기준값 찾기
# 기준값 x를 넣으면 엔트로피 값이 나오는 함수는?

def entropy(n,df):
    df_left=df.query(f"x1 < {n}")  # 1번 그룹
    df_right=df.query(f"x1 >= {n}") # 2번 그룹
    n_left=df_left.shape[0]  # 1번 그룹 n
    n_right=df_right.shape[0] # 2번 그룹 n
    p_left=df_left['y'].value_counts() / n_left
    entropy_left=-sum(p_left * np.log2(p_left))
    p_right=df_right['y'].value_counts() / n_right
    entropy_right=-sum(p_right * np.log2(p_right))
    return (n_left * entropy_left + n_right * entropy_right)/(n_left + n_right)

result = []
x1_values = np.arange(df["x1"].min(),df["x1"].max()+1,0.1)
for x in x1_values:
    result.append(entropy(x,df))
result
x1_values[np.argmin(result)]

plt.plot(x1_values, result)
plt.xlabel('x1')
plt.ylabel('entropy')
plt.show()


# x1, x2 산점도를 그리되, 점 색깔은 펭귄 종별 다르게 그리기!
import seaborn as sns

sns.scatterplot(data=df, x="x1", y="x2", hue='y')
plt.axvline(x=42.30)



