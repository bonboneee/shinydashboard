import pandas as pd
import numpy as np
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()


df=penguins.dropna()
df=df[["bill_length_mm", "bill_depth_mm"]]
df=df.rename(columns={'bill_length_mm': 'y',
                   'bill_depth_mm': 'x'})
df

# 원래 MSE는?
np.mean((df["y"] - df["y"].mean())**2)
29.81

# x=15 기준으로 나눴을때, 데이터포인트가 몇개 씩 나뉘나요?
# 57, 276
n1=df.query("x < 15").shape[0]  # 1번 그룹
n2=df.query("x >= 15").shape[0] # 2번 그룹

# 1번 그룹은 얼마로 예측하나요?
# 2번 그룹은 얼마로 예측하나요?
y_hat1=df.query("x < 15").mean()[0]
y_hat2=df.query("x >= 15").mean()[0]

# 각 그룹 MSE는 얼마 인가요?
mse1=np.mean((df.query("x < 15")["y"] - y_hat1)**2)
mse2=np.mean((df.query("x >= 15")["y"] - y_hat2)**2)

# x=15 의 MSE 가중평균은?
# (mse1 + mse2)*0.5 가 아닌
(mse1* n1 + mse2 * n2)/(n1+n2)
29.23

29.81 - 29.23

# x = 20일때 MSE 가중평균은?
n1=df.query("x < 20").shape[0]  # 1번 그룹
n2=df.query("x >= 20").shape[0] # 2번 그룹
y_hat1=df.query("x < 20").mean()[0]
y_hat2=df.query("x >= 20").mean()[0]
mse1=np.mean((df.query("x < 20")["y"] - y_hat1)**2)
mse2=np.mean((df.query("x >= 20")["y"] - y_hat2)**2)
(mse1* n1 + mse2 * n2)/(n1+n2)
29.73

29.81-29.73

# 기준값 x를 넣으면 MSE값이 나오는 함수는?
def my_mse(x):
    n1=df.query(f"x < {x}").shape[0]  # 1번 그룹
    n2=df.query(f"x >= {x}").shape[0] # 2번 그룹
    y_hat1=df.query(f"x < {x}").mean()[0]
    y_hat2=df.query(f"x >= {x}").mean()[0]
    mse1=np.mean((df.query(f"x < {x}")["y"] - y_hat1)**2)
    mse2=np.mean((df.query(f"x >= {x}")["y"] - y_hat2)**2)
    return float((mse1* n1 + mse2 * n2)/(n1+n2))

my_mse(20)

df["x"].min()
df["x"].max()

# 13~22 사이 값 중 0.01 간격으로 MSE 계산을 해서
# minimize 사용해서 가장 작은 MSE가 나오는 x 찾아보세요!
# depth1 
group1 = df.query("x < 16.4")# 1번 그룹
group2 = df.query("x >= 16.4")  # 2번 그룹 
n1 = df.query("x < 16.4").shape[0]  # 1번 그룹
n2 = df.query("x >= 16.4").shape[0]  # 2번 그룹 

# 1번 그룹, 2번 그룹 예측값 (mean)
y_hat1 = df.query("x < 16.4")['y'].mean() # 1번 그룹 예측값
y_hat2 = df.query("x >= 16.4")['y'].mean() # 2번 그룹 예측값

# 각 그룹의 MSE는 얼마인가요?
mse1 = np.mean((df.query("x < 16.4")['y'] - y_hat1)**2)
mse2 = np.mean((df.query("x >= 16.4")['y'] - y_hat2)**2)

# x = 15의 MSE 가중평균은? 
(mse1*n1 + mse2*n2) / (n1+n2) # 26.09
#----------------------------------트리나무 2세대

# 13~16 사이 값 중 0.01 간격으로 MSE 계산을 해서
# minimize 사용해서 가장 작은 MSE가 나오는 x 찾아보세요!
# depth2 
def my_mse(data,x):
    n1 = data.query(f"x < {x}").shape[0]  # 1번 그룹
    n2 = data.query(f"x >= {x}").shape[0]  # 2번 그룹 
    y_hat1 = data.query(f"x < {x}")['y'].mean() # 1번 그룹 예측값
    y_hat2 = data.query(f"x >= {x}")['y'].mean() # 2번 그룹 예측값

      # 각 그룹의 MSE는 얼마인가요?
    mse1 = np.mean((data.query(f"x < {x}")['y'] - y_hat1)**2)
    mse2 = np.mean((data.query(f"x >= {x}")['y'] - y_hat2)**2)

    return (mse1*n1 + mse2*n2) / (n1+n2) 
my_mse(group1, 14)

x_values = np.arange(group1['x'].min()+0.01, group1['x'].max(), 0.01)
result = np.repeat(0.0, len(x_values))
for i in range(0, len(x_values)):
    result[i] = my_mse(group1, x_values[i])

np.min(result)
x_values[np.argmin(result)]  
#np.float64(14.00999999999998)

x_values = np.arange(group2['x'].min() + 0.01, group2['x'].max(), 0.01)
result = np.repeat(0.0, len(x_values))
for i in range(0, len(x_values)):
    result[i] = my_mse(group2, x_values[i])

np.min(result) 
x_values[np.argmin(result)]
#np.float64(19.400000000000468)



# x, y 산점도를 그리고, 빨간 평행선 4개 그려주세요!
import matplotlib.pyplot as plt

df.plot(kind="scatter", x="x", y="y")
thresholds=[14.01, 16.42, 19.4]
df["group"]=np.digitize(df["x"], thresholds)
y_mean=df.groupby("group").mean()["y"]
k1=np.linspace(13, 14.01, 100)
k2=np.linspace(14.01, 16.42, 100)
k3=np.linspace(16.42, 19.4, 100)
k4=np.linspace(19.4, 22, 100)
plt.plot(k1, np.repeat(y_mean[0],100), color="red")
plt.plot(k2, np.repeat(y_mean[1],100), color="red")
plt.plot(k3, np.repeat(y_mean[2],100), color="red")
plt.plot(k4, np.repeat(y_mean[3],100), color="red")


#--------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 랜덤 시드 고정 (재현성을 위해)
np.random.seed(42)

# x 값 생성: -10에서 10 사이의 100개 값
x = np.linspace(-10, 10, 100)

# y 값 생성: y = x^2 + 정규분포 노이즈
y = x ** 2 + np.random.normal(0, 10, size=x.shape)

# 데이터프레임 생성
df = pd.DataFrame({'x': x, 'y': y})

# 데이터 시각화
plt.scatter(df['x'], df['y'], label='Noisy Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Noisy Quadratic Data')
plt.legend()
plt.show()

# 입력 변수와 출력 변수 분리
X = df[['x']]  # 독립 변수는 2차원 형태로 제공되어야 함
y = df['y']

# 학습 데이터와 테스트 데이터로 분할 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 디시전 트리 회귀 모델 생성 및 학습
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = model.predict(X_test)

# 모델 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# 테스트 데이터의 실제 값과 예측 값 시각화
plt.scatter(X_test, y_test, color='blue', label='Actual Values')
plt.scatter(X_test, y_pred, color='red', label='Predicted Values')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Tree Regression')
plt.legend()
plt.show()



#------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 랜덤 시드 고정 (재현성을 위해)
np.random.seed(42)

# x 값 생성: -10에서 10 사이의 100개 값
x = np.linspace(-10, 10, 100)

# y 값 생성: y = x^2 + 정규분포 노이즈
y = x ** 2 + np.random.normal(0, 10, size=x.shape)

# 데이터프레임 생성
df = pd.DataFrame({'x': x, 'y': y})

# 데이터 시각화
plt.scatter(df['x'], df['y'], label='Noisy Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Noisy Quadratic Data')
plt.legend()
plt.show()

# 입력 변수와 출력 변수 분리
X = df[['x']]  # 독립 변수는 2차원 형태로 제공되어야 함
y = df['y']

# 학습 데이터와 테스트 데이터로 분할 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# 디시전 트리 회귀 모델 생성 및 학습
model = DecisionTreeRegressor(random_state=42,
                              max_depth=4)
model.fit(X_train, y_train)


df_x=pd.DataFrame({"x": x})

# -10, 10까지 데이터에 대한 예측
y_pred = model.predict(df_x)


plt.scatter(df['x'], df['y'], label='Noisy Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Noisy Quadratic Data')
plt.legend()
plt.scatter(df_x['x'], y_pred, color="red")




# 모델 평가
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# 테스트 데이터의 실제 값과 예측 값 시각화
plt.scatter(X_test, y_test, color='blue', label='Actual Values')
plt.scatter(X_test, y_pred, color='red', label='Predicted Values')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Tree Regression')
plt.legend()
plt.show()