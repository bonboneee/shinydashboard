# 원하는 변수를 사용해서 회귀모델을 만들고, 제출할것!
# 원하는 변수 2개
# 회귀모델을 통한 집값 예측

# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## 필요한 데이터 불러오기
house_train=pd.read_csv("data/train.csv")
house_test=pd.read_csv("data/test.csv")
sub_df=pd.read_csv("data/sample_submission.csv")

house_train.shape
house_test.shape

df = pd.concat([house_train, house_test], ignore_index=True)
df

neighborhood_dummies = pd.get_dummies(
    df["Neighborhood"],
    drop_first= True
)
neighborhood_dummies

x = pd.concat([df[["GrLivArea", "GarageArea"]],
               neighborhood_dummies], axis = 1)
y = df["SalePrice"]

train_x = x.iloc[:1460,]
test_x = x.iloc[1460:,]

train_y = y[:1460]


#validation 셋(모의고사 셋) 만들기
np.random.seed(42)
val_index = np.random.choice(np.arange(1460), size = 438, replace = True)
val_index

valid_x = train_x.loc[val_index]  #30%
train_x = train_x.drop(val_index)  #70%
valid_y = train_y[val_index]
train_y = train_y.drop(val_index)


# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(train_x, train_y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b


#성능 측정
y_hat = model.predict(valid_x)
np.sqrt(np.mean((valid_y-y_hat)**2) )

# 성능:np.float64(35991.36537405705)




pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission10.csv", index=False)




#______________이상치 제거해준 버전
# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

house_train.shape
house_test.shape
train_n=len(house_train)

# 통합 df 만들기 + 더미코딩
### house_price 데이터의 모든 범주형 칼럼들을 더미코딩해볼까?
#하지만, 변수들 중에서 good 이나 very good처럼 순서가 있는 아이들은 숫자로 바꿔줘야하고,
#숫자로 되어있음에도 불구하고 범주형인 데이터도 있을 것이다.
#이런 친구들도 더미코딩을 해 줘야한다. 이런 경우 우리들이 변수를 보고 수정을 해야하지만,
#시간이 없으니까 object 타입 열만 가져와서 해보자.
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

# Validation 셋(모의고사 셋) 만들기
np.random.seed(42)
val_index=np.random.choice(np.arange(train_n), size=438, replace=False)
val_index

# train => valid / train 데이터셋
valid_df=train_df.loc[val_index]  # 30%
train_df=train_df.drop(val_index) # 70%

## 이상치 탐색
train_df=train_df.query("GrLivArea <= 4500")

# x, y 나누기
# regex (Regular Expression, 정규방정식)
# ^ 시작을 의미, $ 끝남을 의미, | or를 의미
# selected_columns=train_df.filter(regex='^GrLivArea$|^GarageArea$|^Neighborhood_').columns

## train
train_x=train_df.drop("SalePrice", axis=1)
train_y=train_df["SalePrice"]

## valid
valid_x=valid_df.drop("SalePrice", axis=1)
valid_y=valid_df["SalePrice"]

## test
test_x=test_df.drop("SalePrice", axis=1)

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(train_x, train_y)  # 자동으로 기울기, 절편 값을 구해줌

# 성능 측정 ()
y_hat=model.predict(valid_x)
np.sqrt(np.mean((valid_y-y_hat)**2))

pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission13.csv", index=False)
#------------------------------------------------일준오빠 코드
######### 하우스 프라이스 선생님이 하려던거 완성 시켜 보기
## 필요한 데이터 불러오기
house_train=pd.read_csv("data/train.csv")
house_test=pd.read_csv("data/test.csv")
sub_df=pd.read_csv("data/sample_submission.csv")

# 트레인, 테스트 합치기(더미변수 만드는거 한 번에 처리하기 위해서 더하는거.)
combine_df = pd.concat([house_train, house_test], ignore_index = True) # ignore_index 옵션이 있음.

# 더미변수 만들기
neighborhood_dummies = pd.get_dummies(
    combine_df["Neighborhood"],
    drop_first=True
)

# 더미데이터를 train, test로 데이터 나누기
train_dummies = neighborhood_dummies.iloc[:1460,]

test_dummies = neighborhood_dummies.iloc[1460:,]
test_dummies = test_dummies.reset_index(drop=True) # 인덱스를 초기화(house_test 원본 데이터와 맞춰야) 잘 합쳐짐

# 원래 데이터에서 필요한 변수들만 골라서 더미데이터를 합치기.
my_train = pd.concat([house_train[["SalePrice", "GrLivArea", "GarageArea"]],
               train_dummies], axis=1)

my_test_x = pd.concat([house_test[["GrLivArea", "GarageArea"]],
               test_dummies], axis=1)

# train 데이터의 길이 구하기
train_n = len(my_train) # 1460

## Validation 셋(모의고사 셋) 만들기
np.random.seed(42)
val_index = np.random.choice(np.arange(train_n), size = 438,
                 replace = False) #30% 정도의 갯수를 랜덤으로 고르기.
val_index

new_valid = my_train.loc[val_index]  # 30% 438개
new_train = my_train.drop(val_index) # 70% 1022개

######## 이상치 탐색 및 없애기
new_train = new_train.query("GrLivArea <= 4500") # 나중에 실행하지 말고도 구해보기.

# train 데이터의 길이 구하기
len(new_train) # 1020

# train 데이터 가격 분리하기.
train_x = new_train.iloc[:,1:]
train_y = new_train[["SalePrice"]]

valid_x = new_valid.iloc[:,1:]
valid_y = new_valid[["SalePrice"]]

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(train_x, train_y)

# 성능 측정
y_hat = model.predict(valid_x)
np.mean(np.sqrt((valid_y-y_hat)**2)) #26265

# 위에서 이상치 없애기를 하지 않았다면?
# 25820
# 더 낮은데?