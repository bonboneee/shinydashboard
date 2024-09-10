#y = (x-2)**2 +1 의 그래프를 그리세요.
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-4, 8, 100)
y = (x-2)**2 + 1

plt.plot(x, y, color= 'black')
plt.xlim(-4, 8)
plt.ylim(0, 15)

#y = 4x-11
line_y = 4*x -11
plt.plot(x, line_y, color = 'red')

# ---------------------------------------

# k값이 바뀜에 따라 직선의 기울기가 달라짐.
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-4, 8, 100)
y = (x-2)**2 + 1

plt.plot(x, y, color= 'black')
plt.xlim(-4, 8)
plt.ylim(0, 15)

k = 1

#f'(x) = 2x-4
#k=4에서의 기울기
l_slope=2*k -4
f_k = (k-2)**2 + 1
l_intercept = f_k - (l_slope * k)

#y = slope*x+intercept 그래프
line_y = l_slope*x + l_intercept
plt.plot(x, line_y, color = 'red')

# ---------------------------------------
#f'(x) = 2x이고 초기값이 10이고 델타가 0.9일때 x(100)구하는 경사하강법
x = 10
lstep = 0.9
for i in range(100):
    x = x-lstep*(2*x)

print(x)
# ---------------------------------------
#0자리로 갈수 있도록 조정하는 단계
#f'(x) = 2x이고 초기값이 10이고 델타가 0.9일때 x(100)구하는 경사하강법
x = 10
lstep = 0.00009
for i in range(100000):
    x = x-lstep*(2*x)

print(x)
# ---------------------------------------
#learning step을 조정하는 단계
#f'(x) = 2x이고 초기값이 10이고 델타가 0.9일때 x(100)구하는 경사하강법
x = 10
lstep = np.arange(100, 0, -1)*0.01
for i in range(100):
    x = x-lstep[i]*(2*x)

print(x)

# -----------------------------------------
#세계관 확장
# f(x, y) = (x-3)**2 + (y-4)**2 + 3
# 시작값:(9,2), 델타 = 0.1

#----------------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# x, y의 값을 정의합니다 (-1에서 7까지)
x = np.linspace(-1, 7, 400)
y = np.linspace(-1, 7, 400)
x, y = np.meshgrid(x, y)

# 함수 f(x, y)를 계산합니다.
z = (x - 3)**2 + (y - 4)**2 + 3

# 그래프를 그리기 위한 설정
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 표면 그래프를 그립니다.
ax.plot_surface(x, y, z, cmap='viridis')

# 레이블 및 타이틀 설정
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(x, y)')
ax.set_title('Graph of f(x, y) = (x-3)^2 + (y-4)^2 + 3')

# 그래프 표시
plt.show()

# ==========================
# 등고선 그래프
import numpy as np
import matplotlib.pyplot as plt

# x, y의 값을 정의합니다 (-1에서 7까지)
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
x, y = np.meshgrid(x, y)

# 함수 f(x, y)를 계산합니다.
z = (x - 3)**2 + (y - 4)**2 + 3

# 등고선 그래프를 그립니다.
plt.figure()
cp = plt.contour(x, y, z, levels=20)  # levels는 등고선의 개수를 조절합니다.
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가합니다.

# 특정 점 (9, 2)에 파란색 점을 표시
plt.scatter(9, 2, color='red', s=50)

x=9; y=2
lstep=0.1
for i in range(100):
    x, y = np.array([x, y]) - lstep * np.array([2*x-6, 2*y-8])
    plt.scatter(float(x), float(y), color='red', s=25)

print(x, y)

# 축 레이블 및 타이틀 설정
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Contour Plot of f(x, y) = (x-3)^2 + (y-4)^2 + 3')
plt.xlim(-10, 10)
plt.ylim(-10, 10)

# 그래프 표시
plt.show()

#---------------------------------조별 최소 베타 벡터
# beta0, beta1의 값을 정의합니다 
beta0 = np.linspace(-200, 200, 100)
beta1 = np.linspace(-200, 200, 100)
beta0, beta1 = np.meshgrid(beta0, beta1)

# 함수 f(beta0, beta1)를 계산합니다.
z = 4*beta0**2 + 20*beta0*beta1 - 23*beta0 + 30*beta1**2 - 67*beta1 + 44.25

# 등고선 그래프를 그립니다.
plt.figure()
cp = plt.contour(beta0, beta1, z, levels=500)  # levels는 등고선의 개수를 조절합니다.
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가합니다.

# 특정 점 (10, 10)에 파란색 점을 표시
plt.scatter(10, 10, color='red', s=20)

beta0=10; beta1=10
lstep=0.01
for i in range(100):
    beta0, beta1 = np.array([beta0, beta1]) - lstep * np.array([8*beta0+20*beta1-23, 60*beta1+20*beta0-67])
    plt.scatter(float(beta0), float(beta1), color='red', s=20)

print(beta0, beta1)

# 축 레이블 및 타이틀 설정
plt.xlabel('beta0')
plt.ylabel('beta1')
plt.xlim(-30, 30)
plt.ylim(-30, 30)

# 그래프 표시
plt.show()

#---------------------------
# 모델 fit으로 베타 구하기
import pandas as pd
from sklearn.linear_model import LinearRegression

df=pd.DataFrame({
    'x': np.array([1, 2, 3, 4]),
    'y': np.array([1, 4, 1.5, 5])
})
model = LinearRegression()
model.fit(df[['x']], df['y'])

model.intercept_
model.coef_






# -----------------------------
#블루베리 데이터 셋에 라쏘,릿지, KNN회귀분석 코드 적용해서
#하이퍼 파라미터 결정 후 각 모델의 예측값을 계산, bagging 적용해서 sumbit
# lasso
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

blueberry_train=pd.read_csv("data/BLUEBERRY/train.csv")
blueberry_test=pd.read_csv("data/BLUEBERRY/test.csv")
sub_df=pd.read_csv("data/BLUEBERRY/sample_submission.csv")

# 데이터 생성
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

# 데이터를 DataFrame으로 변환하고 다항 특징 추가
x_vars = np.char.add('x', np.arange(1, 21).astype(str))
X = pd.DataFrame(x, columns=['x'])
poly = PolynomialFeatures(degree=20, include_bias=False)
X_poly = poly.fit_transform(X)
X_poly=pd.DataFrame(
    data=X_poly,
    columns=x_vars
)

# 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, X_poly, y, cv = kf,
                                     scoring = "neg_mean_squared_error").mean())
    return(score)

lasso = Lasso(alpha=0.01)
rmse(lasso)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0, 10, 0.01)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

df

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)

#---------------Ridge
from sklearn.linear_model import Ridge
# 데이터 생성
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

# 데이터를 DataFrame으로 변환하고 다항 특징 추가
x_vars = np.char.add('x', np.arange(1, 21).astype(str))
X = pd.DataFrame(x, columns=['x'])
poly = PolynomialFeatures(degree=20, include_bias=False)
X_poly = poly.fit_transform(X)
X_poly=pd.DataFrame(
    data=X_poly,
    columns=x_vars
)

# 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, X_poly, y, cv = kf,
                                     scoring = "neg_mean_squared_error").mean())
    return(score)

ridge = Ridge(alpha=0.01)
rmse(ridge)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0, 10, 0.01)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    ridge = Ridge(alpha=alpha)
    mean_scores[k] = rmse(ridge)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

df

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('ridge Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)

import pandas as pd
