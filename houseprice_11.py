# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


## 필요한 데이터 불러오기
house_train=pd.read_csv("C:/Users/USER/Documents/LS빅데이터/shinydashboard/data/train.csv")
house_test=pd.read_csv("C:/Users/USER/Documents/LS빅데이터/shinydashboard/data/test.csv")
sub_df=pd.read_csv("C:/Users/USER/Documents/LS빅데이터/shinydashboard/data/sample_submission.csv")

## NaN 채우기
# 각 숫치형 변수는 평균 채우기
# 각 범주형 변수는 Unknown 채우기
house_train.isna().sum()
house_test.isna().sum()

## 숫자형 채우기
quantitative = house_train.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)
house_train[quant_selected].isna().sum()

## 범주형 채우기
qualitative = house_train.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_train[col].fillna("unknown", inplace=True)
house_train[qual_selected].isna().sum()


# test 데이터 채우기
## 숫자형 채우기
quantitative = house_test.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_test[col].fillna(house_train[col].mean(), inplace=True)
house_test[quant_selected].isna().sum()

## 범주형 채우기
qualitative = house_test.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_test[col].fillna("unknown", inplace=True)
house_test[qual_selected].isna().sum()


house_train.shape
house_test.shape
train_n=len(house_train)

# 통합 df 만들기 + 더미코딩
# house_test.select_dtypes(include=[int, float])

df = pd.concat([house_train, house_test], ignore_index=True)
# df.info()
df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )
df

# train / test 데이터셋
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

## 이상치 탐색
#train_df=train_df.query("GrLivArea <= 4500")


## train
train_x=train_df.drop("SalePrice", axis=1)
train_y=train_df["SalePrice"]

## test
test_x=test_df.drop("SalePrice", axis=1)

# 모델 정의 및 그리드 서치
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# ElasticNet
eln_model = ElasticNet()
param_grid_eln = {
    'alpha': [0.1, 1.0, 10.0, 100.0],
    'l1_ratio': [0, 0.1, 0.5, 1.0]
}
grid_search_eln = GridSearchCV(
    estimator=eln_model,
    param_grid=param_grid_eln,
    scoring='neg_mean_squared_error',
    cv=3, verbose=1
)
grid_search_eln.fit(train_x, train_y)
best_eln_model = grid_search_eln.best_estimator_

# RandomForest
rf_model = RandomForestRegressor(n_estimators=500, max_features="sqrt")
param_grid_rf = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [5, 10, 20, 30],
    'min_samples_leaf': [5, 10, 20, 30]
}
grid_search_rf = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid_rf,
    scoring='neg_mean_squared_error',
    cv=3, verbose=1
)
grid_search_rf.fit(train_x, train_y)
best_rf_model = grid_search_rf.best_estimator_

# XGBoost 추가
xgb_model = xgb.XGBRegressor()
param_grid_xgb = {
    'n_estimators': [100, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7]
}
grid_search_xgb = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid_xgb,
    scoring='neg_mean_squared_error',
    cv=3, verbose=1
)
grid_search_xgb.fit(train_x, train_y)
best_xgb_model = grid_search_xgb.best_estimator_

# 스택킹
y1_hat = best_eln_model.predict(train_x)
y2_hat = best_rf_model.predict(train_x)
y3_hat = best_xgb_model.predict(train_x)

train_x_stack = pd.DataFrame({
    'y1': y1_hat,
    'y2': y2_hat,
    'y3': y3_hat
})

# Randomforest 메타 모델로 스택킹
from sklearn.linear_model import Ridge
param_grid_ridge = {
    'alpha': np.arange(0, 10, 0.01)
}
rg_model = Ridge()
grid_search_ridge = GridSearchCV(
    estimator=rg_model,
    param_grid=param_grid_ridge,
    scoring='neg_mean_squared_error',
    cv=3, verbose=1
)
grid_search_ridge.fit(train_x_stack, train_y)
blander_model = grid_search_ridge.best_estimator_

# 테스트 데이터에 대한 예측
pred_y_eln = best_eln_model.predict(test_x)
pred_y_rf = best_rf_model.predict(test_x)
pred_y_xgb = best_xgb_model.predict(test_x)

test_x_stack = pd.DataFrame({
    'y1': pred_y_eln,
    'y2': pred_y_rf,
    'y3': pred_y_xgb
})

pred_y = blander_model.predict(test_x_stack)


# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# # csv 파일로 내보내기
sub_df.to_csv("C:/Users/USER/Documents/LS빅데이터/shinydashboard/data/submit/sample_submission3.csv", index=False)