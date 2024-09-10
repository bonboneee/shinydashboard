# 01. Lasso로 분석하기
## 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

blueberry_train=pd.read_csv("data/BLUEBERRY/train.csv")
blueberry_test=pd.read_csv("data/BLUEBERRY/test.csv")
sub_df=pd.read_csv("data/BLUEBERRY/sample_submission.csv")

# 데이터 전처리
# IQR (Interquartile Range) 계산을 통한 이상치 탐지 및 제거
def remove_outliers(blueberry_train):
    # IQR 범위를 설정할 열 목록
    cols = blueberry_train.columns.drop(['id'])

    # IQR 방식으로 이상치 제거
    for col in cols:
        Q1 = blueberry_train[col].quantile(0.25)
        Q3 = blueberry_trainf[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        blueberry_train = blueberry_train[(blueberry_train[col] >= lower_bound) & (blueberry_train[col] <= upper_bound)]

    return blueberry_train


# train_X, train_y 구분
blueberry_train_X = blueberry_train.drop(columns=["id", "yield"])
blueberry_train_y = blueberry_train[["yield"]]


# 교차 검증 설정
kf = KFold(n_splits= 6, shuffle=True, random_state=2024)

# n_jobs = -1, cpu에 작업을 각각 할당
def rmse(model):
    score = np.sqrt(-cross_val_score(model, X, y, cv=kf,
                                     n_jobs=-1, 
                                     scoring="neg_mean_squared_error").mean())
    return score

# Lasso와 Ridge 모델에 대한 검증 오류 계산
alpha_values = np.arange(0, 100, 0.1)
lasso_scores = np.zeros(len(alpha_values))
ridge_scores = np.zeros(len(alpha_values))

for i, alpha in enumerate(alpha_values):
    lasso = Lasso(alpha=alpha)
    ridge = Ridge(alpha=alpha)
    lasso_scores[i] = rmse(lasso)
    ridge_scores[i] = rmse(ridge)

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'lasso_validation_error': lasso_scores,
    'ridge_validation_error': ridge_scores
})

# 결과 시각화
plt.plot(df['lambda'], df['lasso_validation_error'], label='Lasso Validation Error', color='red')
plt.plot(df['lambda'], df['ridge_validation_error'], label='Ridge Validation Error', color='blue')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso vs Ridge Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기
optimal_lasso_alpha = df['lambda'][np.argmin(df['lasso_validation_error'])]
optimal_ridge_alpha = df['lambda'][np.argmin(df['ridge_validation_error'])]

print("Optimal Lasso lambda:", optimal_lasso_alpha)
print("Optimal Ridge lambda:", optimal_ridge_alpha)



# 라쏘 회귀 모델 생성
model = Lasso(alpha = 0.0)
model_1 = Ridge(alpha = 1.1)

# 모델 학습
model.fit(X, y)
model_1.fit(X, y)


# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/berry_sample_submission.csv", index=False)


# -----------------------------------------------------------------------

# 회귀직선으로 분석해보기-------------------------------------------------
# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

## 필요한 데이터 불러오기
berry_train=pd.read_csv("data/BLUEBERRY/train.csv")
berry_test=pd.read_csv("data/BLUEBERRY/test.csv")
sub_df=pd.read_csv("data/BLUEBERRY/sample_submission.csv")

##train, test 데이터 만들기
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