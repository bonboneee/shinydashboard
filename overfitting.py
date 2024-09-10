import numpy as np
import matplotlib.pyplot as plt
#y = a*x**2 +b*x + c 그래프 그리기
a = 2
b = 3
c = 5
x = np.linspace(-8, 8, 100)
y = a*x**2 +b*x + c

plt.plot(x, y, color='black')
plt.show()
plt.clf()



#y = a*x**3 + b*x**2 + c*x + d 그래프 그리기
a = 2
b = 3
c = 0
d = -1
x = np.linspace(-8, 8, 100)
y = a*x**3 + b*x**2 + c*x + d

plt.plot(x, y, color='black')
plt.show()
plt.clf()



#y = a*x**4 + b*x**3 + c*x**2 + d*x + e 그래프 그리기
a = 1
b = 0
c = -10
d = -0
e = 10
x = np.linspace(-4, 4, 100)
y = a*x**4 + b*x**3 + c*x**2 + d*x + e

plt.plot(x, y, color='black')
plt.show()
plt.clf()




# 데이터 만들기
from scipy.stats import norm
from scipy.stats import uniform
norm.rvs(size=1, loc=0, scale=3)

# 검정곡선
k = np.linspace(-4, 4, 20)
sin_y = np.sin(k)
#파란 점들
x = uniform.rvs(size = 20, loc = -4, scale = 8)
y = np.sin(x) + norm.rvs(size=20, loc=0, scale=0.3)

plt.plot(k,sin_y, color = 'black')
plt.scatter(x, y, color = 'blue')




#train, test 데이터 만들기
np.random.seed(42)
x = uniform.rvs(size = 30, loc = -4, scale = 8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
df = pd.DataFrame({
    "x":x, "y":y
})
df

train_df = df.loc[:19]
train_df

test_df = df.loc[20:]
test_df

plt.scatter(train_df["x"], train_df["y"], color = "blue")




#train셋으로 회귀직선 만들어보기
from sklearn.linear_model import LinearRegression
model=LinearRegression()

# x와 y 설정
x = train_df[["x"]]
y = train_df["y"]

# 모델 학습
model.fit(x, y)

model.coef_
model.intercept_

regline =model.predict(x)

plt.plot(x, regline, color = "red")
plt.scatter(train_df["x"], train_df["y"], color = "blue")


#2차곡선회귀- 회귀곡선이 이상한 경우
train_df
train_df["x2"] = train_df["x"]**2
x = train_df[["x", "x2"]]
y = train_df["y"]

model.fit(x,y)

model.coef_
model.intercept_


regline =model.predict(x)

plt.plot(x, regline, color = "red")
plt.scatter(train_df["x"], train_df["y"], color = "blue")





#2차곡선회귀 시작!
train_df["x2"] = train_df["x"]**2
train_df

x = train_df[["x", "x2"]]
y = train_df["y"]

model.fit(x,y)

model.coef_
model.intercept_

k = np.linspace(-4, 4, 200)
df_k = pd.DataFrame({
    "x":k, "x2": k**2
})
df_k
regline =model.predict(df_k)

plt.plot(k, regline, color = "red")
plt.scatter(train_df["x"], train_df["y"], color = "blue")






#3차곡선회귀 시작!
train_df["x3"] = train_df["x"]**3
train_df

x = train_df[["x", "x2", "x3"]]
y = train_df["y"]

model.fit(x,y)

model.coef_
model.intercept_

k = np.linspace(-4, 4, 200)
df_k = pd.DataFrame({
    "x":k, "x2": k**2, "x3": k**3
})
df_k
regline =model.predict(df_k)

plt.plot(k, regline, color = "red")
plt.scatter(train_df["x"], train_df["y"], color = "blue")




#4차곡선회귀 시작!
train_df["x4"] = train_df["x"]**4
train_df

x = train_df[["x", "x2", "x3", "x4"]]
y = train_df["y"]

model.fit(x,y)

model.coef_
model.intercept_

k = np.linspace(-4, 4, 200)
df_k = pd.DataFrame({
    "x":k, "x2": k**2, "x3": k**3, "x4": k**4
})
df_k
regline =model.predict(df_k)

plt.plot(k, regline, color = "red")
plt.scatter(train_df["x"], train_df["y"], color = "blue")


#9차곡선회귀 시작!
train_df["x5"] = train_df["x"]**5
train_df["x6"] = train_df["x"]**6
train_df["x7"] = train_df["x"]**7
train_df["x8"] = train_df["x"]**8
train_df["x9"] = train_df["x"]**9
train_df

x = train_df[["x", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]]
y = train_df["y"]

model.fit(x,y)

model.coef_
model.intercept_

k = np.linspace(-4, 4, 200)
df_k = pd.DataFrame({
    "x":k, "x2": k**2, "x3": k**3, "x4": k**4,
    "x5": k**5, "x6": k**6, "x7": k**7, "x8": k**8, "x9": k**9
})
df_k
regline =model.predict(df_k)

plt.plot(k, regline, color = "red")
plt.scatter(train_df["x"], train_df["y"], color = "blue")




#15차곡선회귀 시작!
train_df["x10"] = train_df["x"]**10
train_df["x11"] = train_df["x"]**11
train_df["x12"] = train_df["x"]**12
train_df["x13"] = train_df["x"]**13
train_df["x14"] = train_df["x"]**14
train_df["x15"] = train_df["x"]**15
train_df

x = train_df[["x", "x2", "x3", "x4", "x5", "x6", "x7",
              "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15"]]
y = train_df["y"]

model.fit(x,y)

model.coef_
model.intercept_

k = np.linspace(-4, 4, 200)
df_k = pd.DataFrame({
    "x":k, "x2": k**2, "x3": k**3, "x4": k**4,
    "x5": k**5, "x6": k**6, "x7": k**7, "x8": k**8, "x9": k**9,
    "x10": k**10, "x11": k**11, "x12": k**12, "x13": k**13, "x14": k**14, "x15": k**15
})
df_k
regline =model.predict(df_k)

plt.plot(k, regline, color = "red")
plt.scatter(train_df["x"], train_df["y"], color = "blue")





#<테스트셋을 과대적합 (9차곡선)그래프에 넣어보기>
train_df["x5"] = train_df["x"]**5
train_df["x6"] = train_df["x"]**6
train_df["x7"] = train_df["x"]**7
train_df["x8"] = train_df["x"]**8
train_df["x9"] = train_df["x"]**9
train_df

x = train_df[["x", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]]
y = train_df["y"]

model.fit(x,y)

model.coef_
model.intercept_

k = np.linspace(-4, 4, 200)
df_k = pd.DataFrame({
    "x":k, "x2": k**2, "x3": k**3, "x4": k**4,
    "x5": k**5, "x6": k**6, "x7": k**7, "x8": k**8, "x9": k**9
})
df_k
regline =model.predict(df_k)

plt.plot(k, regline, color = "red")
plt.scatter(train_df["x"], train_df["y"], color = "blue")

## 테스트셋 x를 9차까지 만들어주기
test_df["x"] = test_df["x"]
test_df["x2"] = test_df["x"]**2
test_df["x3"] = test_df["x"]**3
test_df["x4"] = test_df["x"]**4
test_df["x5"] = test_df["x"]**5
test_df["x6"] = test_df["x"]**6
test_df["x7"] = test_df["x"]**7
test_df["x8"] = test_df["x"]**8
test_df["x9"] = test_df["x"]**9
test_df

x = test_df[["x", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]]
y_hat = model.predict(x)

sum((test_df["y"] - y_hat)**2) #9차 모델 성능: 0.8949117687299492




#20차 트레인셋 만들고, 20차 회귀곡선을 테스트셋으로 성능 평가해보기
train_df["x10"] = train_df["x"]**10
train_df["x11"] = train_df["x"]**11
train_df["x12"] = train_df["x"]**12
train_df["x13"] = train_df["x"]**13
train_df["x14"] = train_df["x"]**14
train_df["x15"] = train_df["x"]**15
train_df["x16"] = train_df["x"]**16
train_df["x17"] = train_df["x"]**17
train_df["x18"] = train_df["x"]**18
train_df["x19"] = train_df["x"]**19
train_df["x20"] = train_df["x"]**20
train_df

x = train_df[["x", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
              "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18",
              "x19", "x20"]]
y = train_df["y"]

model.fit(x,y)

model.coef_
model.intercept_

k = np.linspace(-4, 4, 200)
df_k = pd.DataFrame({
    "x":k, "x2": k**2, "x3": k**3, "x4": k**4,
    "x5": k**5, "x6": k**6, "x7": k**7, "x8": k**8, "x9": k**9,
    "x10": k**10, "x11": k**11, "x12": k**12, "x13": k**13, "x14": k**14,
    "x15": k**15, "x16": k**16, "x17": k**17, "x18": k**18, "x19": k**19, "x20": k**20
})
df_k
regline =model.predict(df_k)

plt.plot(k, regline, color = "red")
plt.scatter(train_df["x"], train_df["y"], color = "blue")

## 테스트셋 x를 20차까지 만들어주기
test_df["x"] = test_df["x"]
test_df["x2"] = test_df["x"]**2
test_df["x3"] = test_df["x"]**3
test_df["x4"] = test_df["x"]**4
test_df["x5"] = test_df["x"]**5
test_df["x6"] = test_df["x"]**6
test_df["x7"] = test_df["x"]**7
test_df["x8"] = test_df["x"]**8
test_df["x9"] = test_df["x"]**9
test_df["x10"] = test_df["x"]**10
test_df["x11"] = test_df["x"]**11
test_df["x12"] = test_df["x"]**12
test_df["x13"] = test_df["x"]**13
test_df["x14"] = test_df["x"]**14
test_df["x15"] = test_df["x"]**15
test_df["x16"] = test_df["x"]**16
test_df["x17"] = test_df["x"]**17
test_df["x18"] = test_df["x"]**18
test_df["x19"] = test_df["x"]**19
test_df["x20"] = test_df["x"]**20
test_df

x = test_df[["x", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
             "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18",
              "x19", "x20"]]
y_hat = model.predict(x)

sum((test_df["y"] - y_hat)**2) #20차 성능 검사: 278823.27881988714

## 20차 모델 성능을 알아보자능(한결언니: for 문써서 하는 법)
np.random.seed(42)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
df = pd.DataFrame({
    "x" : x , "y" : y
})
df

train_df = df.loc[:19]
train_df

for i in range(2, 21):
    train_df[f"x{i}"] = train_df["x"] ** i
    
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
y = train_df["y"]

model.fit(x,y)

test_df = df.loc[20:]
test_df

for i in range(2, 21):
    test_df[f"x{i}"] = test_df["x"] ** i

# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
x = test_df[["x"] + [f"x{i}" for i in range(2, 21)]]

y_hat = model.predict(x)

# 모델 성능
sum((test_df["y"] - y_hat)**2)







