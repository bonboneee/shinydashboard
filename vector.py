import numpy as np
#벡터 * 벡터(내적)
a = np.arange(1, 4)
b = np.array([3, 6, 9])
a.dot(b)


#행렬 * 벡터(곱셈)
a = np.array([1, 2, 3, 4]).reshape((2, 2), order = 'F')
a

b = np.array([5, 6]).reshape(2, 1)
b

a.dot(b)
a @ b


#행렬 * 행렬
a = np.array([1, 2, 3, 4]).reshape((2, 2), order = 'F')
b = np.array([5, 6, 7, 8]).reshape((2, 2), order = 'F')
a
b
a @ b



# 연습문제1
a = np.array([1, 2, 1, 0, 2, 3]).reshape(2, 3)
b = np.array([1, 0, -1, 1, 2, 3]).reshape(3, 2)
a
b
a @ b


#연습문제2(단위행렬)
a = np.array([3, 5, 7, 2, 4, 9, 3, 1, 0]).reshape(3, 3)
np.eye(3)

a @ np.eye(3)


#transpose
a = np.array([3, 5, 7, 2, 4, 9, 3, 1, 0]).reshape(3, 3)
a.transpose()
b = a[:,0:2]
b
b.transpose()


#회귀분석 데이터행렬(y_hat 한번에 행렬로 데려오기)
x = np.array([13, 15, 12, 14, 10, 11, 5, 6]).reshape(4, 2)
x
vec1 = np.repeat(1, 4).reshape(4, 1)
matX = np.hstack((vec1, x))
matX

beta_vec = np.array([2, 3, 1]).reshape(3, 1)
beta_vec
matX @ beta_vec

y = np.array([20, 19, 20, 12]).reshape(4, 1)

(y - matX @ beta_vec).transpose() * (y - matX @ beta_vec) #beta_vec 값이 바뀌면 결과도 바뀜

# 2*2 역행렬
a = np.array([1, 5, 3, 4]).reshape(2, 2)
a_inv = (-1/11)*np.array([4, -5, -3, 1]).reshape(2, 2)
a @ a_inv

# 3*3 역행렬
a = np.array([-4, -6, 2, 5, -1, 3, -2, 4, -3]).reshape(3, 3)
a_inv = np.linalg.inv(a)

np.round(a @ a_inv, 3)
np.linalg.det(a) #np.float64(18.000000000000014)

# 역행렬 존재하지 않는 경우(선형 종속)
# 특이행렬(singular matrix) = 행렬식이 0이다.
b = np.array([1, 2, 3, 2, 4, 5, 3, 6, 7]).reshape(3, 3)
b_inv = np.linalg.inv(b)  # LinAlgError: Singular matrix
np.linalg.det(b) #np.float64(0.0)

# 베타 구하기
#<최적화 회귀직선 계수 구하기>_방법 1
#전치행렬, 단위행렬, 역행렬 개념이용해서 구하기
XtX_inv = np.linalg.inv(matX.transpose() @ matX)
Xty = matX.transpose() @ y
beta_hat = XtX_inv @ Xty
beta_hat

# <최적화 회귀직선 계수 구하기>_방법 2
# model.fit(x,y) 써서 베타 구하기
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(matX[:, 1:], y)

model.coef_
model.intercept_


# <최적화 회귀직선 계수 구하기>_방법 3
# minimize 사용하기
from scipy.optimize import minimize

def line_perform(beta):
    beta = np.array(beta).reshape(3, 1)
    a = (y - matX @ beta) #matX @ beta = y_hat
    return (a.transpose() @ a)

#line_perform([8.55, 5.96, -4.38])

## 초기 추정값
initial_guess = [0, 0, 0]

## 최소값 찾기
result = minimize(line_perform, initial_guess)

##결과출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)





# minimize로 lasso beta구하기(람다가 3일때)
from scipy.optimize import minimize

def line_perform_lasso(beta):
    beta = np.array(beta).reshape(3, 1)
    a = (y - matX @ beta)
    return (a.transpose() @ a) + 3*np.abs(beta[1:]).sum()

#line_perform_lasso([8.55, 5.96, -4.38])

## 초기 추정값
initial_guess = [0, 0, 0]

## 최소값 찾기
result = minimize(line_perform_lasso, initial_guess)

##결과출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)
##최소값: 27.64144762523693
##최소값을 갖는 x 값: [ 3.82322035e+00  1.35518652e+00 -1.10454967e-09]

[8.55, 5.96, -4.38] #람다 0
#일반 선형회귀랑 동일함

[8.14, 0.96, 0] #람다 3
#예측식: y_hat = 8.14 + 0.96 * X1 + 0 * X2

[17.74, 0, 0] #람다 500
# 예측식: y_hat = 17.74 + 0 * X1 + 0 * X2

#람다 값에 따라 변수가 선택된다.
# X 변수가 추가되면, train X에서는 성능 항상 좋아짐.
# X 변수가 추가되면, valid X에서는 좋아졌다가 나빠짐(오버피팅)
#어느순간 X 변수 추가하는 것을 멈춰야 함.
#람다 0부터 시작": 내가 가진 모든 변수를 넣겠다!
#점점 람다를 증가: 변수가 하나씩 빠지는 효과
#valid X에서 가장 성능이 좋은 람다를 선택!
#변수가 선택됨을 의미.


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

# 20차 모델 성능을 알아보자능
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
df = pd.DataFrame({
    "y" : y,
    "x" : x
})
df

train_df = df.loc[:19]
train_df

for i in range(2, 21):
    train_df[f"x{i}"] = train_df["x"] ** i
    
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
train_x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
train_y = train_df["y"]

valid_df = df.loc[20:]
valid_df

for i in range(2, 21):
    valid_df[f"x{i}"] = valid_df["x"] ** i

# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
valid_x = valid_df[["x"] + [f"x{i}" for i in range(2, 21)]]
valid_x
valid_y = valid_df["y"]
valid_y

from sklearn.linear_model import Lasso

val_result=np.repeat(0.0, 100)
tr_result=np.repeat(0.0, 100)

for i in np.arange(0, 100):
    model= Lasso(alpha=i*0.01)
    model.fit(train_x, train_y)

    # 모델 성능
    y_hat_train = model.predict(train_x)
    y_hat_val = model.predict(valid_x)

    perf_train=sum((train_df["y"] - y_hat_train)**2)
    perf_val=sum((valid_df["y"] - y_hat_val)**2)
    tr_result[i]=perf_train
    val_result[i]=perf_val

tr_result
val_result

import seaborn as sns

df = pd.DataFrame({
    'lambda': np.arange(0, 1, 0.01), 
    'tr': tr_result,
    'val': val_result
})

# seaborn을 사용하여 산점도 그리기
sns.scatterplot(data=df, x='lambda', y='tr')
sns.scatterplot(data=df, x='lambda', y='val', color='red')
plt.xlim(0, 0.4)

val_result[0]
val_result[1]
np.min(val_result)

# alpha를 0.03로 선택!
np.argmin(val_result)
np.arange(0, 1, 0.01)[np.argmin(val_result)]



##추정된 라쏘(lambda=0.03) 모델을 사용해서, -4,4까지 간격 0.01
## x에 대하여 예측값계산
##산점도에 valid set 그린 다음
## -4, 4까지 예측값을 빨간선으로 겹쳐서 그릴것
model= Lasso(alpha=0.03)
model.fit(train_x, train_y)
model.coef_
model.intercept_
#model.predict(test_x)
line_x = np.linspace(-4,4,801)

df_new = pd.DataFrame({
    "x" : line_x
})

df_new

for i in range(2, 21):
    df_new[f"x{i}"] = df_new["x"] ** i
line_new = model.predict(df_new)

plt.plot(df_new["x"], line_new, color="red")
plt.scatter(valid_df["x"], valid_df["y"], color="blue")




# 트레인셋 비율 80 
# 벨리드셋 비율 20
# 그런데 벨리드셋이 교차됨.
# 0.03 람다에서 총 5개의 점들을 얻을텐데 그 점들의 평균과 표준편차 구하기

#1. 편향적인 데이터 뭉침을 셔플로 풀어주기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

# 20차 모델 성능을 알아보자
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

df = pd.DataFrame({
    "y" : y,
    "x" : x
})
df
for i in range(2, 21):
    df[f"x{i}"] = df["x"] ** i

df

def make_tr_val(fold_num, df):
    np.random.seed(2024)
    myindex=np.random.choice(30, 30, replace=False)

    # valid index
    val_index=myindex[(10*fold_num):(10*fold_num+10)]

    # valid set, train set
    valid_set=df.loc[val_index]
    train_set=df.drop(val_index)

    train_X=train_set.iloc[:,1:]
    train_y=train_set.iloc[:,0]

    valid_X=valid_set.iloc[:,1:]
    valid_y=valid_set.iloc[:,0]

    return (train_X, train_y, valid_X, valid_y)

# -----------------------

from sklearn.linear_model import Lasso

val_result_total=np.repeat(0.0, 3000).reshape(3, -1)
tr_result_total=np.repeat(0.0, 3000).reshape(3, -1)

for j in np.arange(0, 3):
    train_X, train_y, valid_X, valid_y = make_tr_val(fold_num=j, df=df)

    # 결과 받기 위한 벡터 만들기
    val_result=np.repeat(0.0, 1000)
    tr_result=np.repeat(0.0, 1000)

    for i in np.arange(0, 1000):
        model= Lasso(alpha=i*0.01)
        model.fit(train_X, train_y)

        # 모델 성능
        y_hat_train = model.predict(train_X)
        y_hat_val = model.predict(valid_X)

        perf_train=sum((train_y - y_hat_train)**2)
        perf_val=sum((valid_y - y_hat_val)**2)
        tr_result[i]=perf_train
        val_result[i]=perf_val

    tr_result_total[j,:]=tr_result
    val_result_total[j,:]=val_result


import seaborn as sns

df = pd.DataFrame({
    'lambda': np.arange(0, 10, 0.01), 
    'tr': tr_result_total.mean(axis=0),
    'val': val_result_total.mean(axis=0)
})

df['tr']

# seaborn을 사용하여 산점도 그리기
# sns.scatterplot(data=df, x='lambda', y='tr')
sns.scatterplot(data=df, x='lambda', y='val', color='red')
plt.xlim(0, 10)

# alpha를 2.67로 선택!
np.argmin(val_result_total.mean(axis=0))
np.arange(0, 10, 0.01)[np.argmin(val_result_total.mean(axis=0))]



#-------------------------------------
## 각 행을 고정한 상태에서 셔플
## 각 행을 고정한 상태에서 셔플한 후 데이터프레임으로 변환
#2. 데이터를 총 5등분하고 이름을 지정하기
#3. 1~4가 트레인셋, 5번째가 벨리드셋 => 람다 0.03점의 Y값 구하기
#4. 1~3, 5가 트레인셋, 4번째가 벨리드셋=> 람다 0.03점의 Y값 구하기
#5. 1~2, 4~5가 트레인셋, 3번째가 벨리드셋=> 람다 0.03점의 Y값 구하기
#6. 1, 3~5가 트레인셋, 2번째가 벨리드셋=> 람다 0.03점의 Y값 구하기
#7. 2~5가 트레인셋, 1번째가 벨리드셋=> 람다 0.03점의 Y값 구하기
#8. 3~7까지 구한 5개의 Y값들의 평균과 표준편차 구하기

















# minimize로 ridge beta구하기(람다가 3일때)
from scipy.optimize import minimize

def line_perform_ridge(beta):
    beta = np.array(beta).reshape(3, 1)
    a = (y - matX @ beta)
    return (a.transpose() @ a) + 3*(beta**2).sum()

line_perform_ridge([8.55, 5.96, -4.38])

## 초기 추정값
initial_guess = [0, 0, 0]

## 최소값 찾기
result = minimize(line_perform_ridge, initial_guess)

##결과출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)
## 최소값: 30.552748885586993
## 최소값을 갖는 x 값: [0.86627049 0.91084704 0.61961358]