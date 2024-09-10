#종속변수: 백혈병 세포 관측 불가 여부 (REMISS), 1이면 관측 안됨을 의미
#
#독립변수:
#
#골수의 세포성 (CELL)
#골수편의 백혈구 비율 (SMEAR)
#골수의 백혈병 세포 침투 비율 (INFIL)
#골수 백혈병 세포의 라벨링 인덱스 (LI)
#말초혈액의 백혈병 세포 수 (BLAST)
#치료 시작 전 최고 체온 (TEMP)
#문제 1.데이터를 로드하고, 로지스틱 회귀모델을 적합하고, 회귀 표를 작성하세요.
import pandas as pd
df = pd.read_csv("./data/leukemia_remission.csv")

import statsmodels.api as sm
import pandas as pd
import numpy as np
model = sm.formula.logit("REMISS ~ CELL + SMEAR + INFIL + LI + BLAST + TEMP",
                         data=df).fit()
print(model.summary())
#문제 2. 해당 모델은 통계적으로 유의한가요? 그 이유를 검정통계량를 사용해서 설명하시오.
#유의수준 0.05하에서 LLR p-value가 0.0467로 유의수준보다 작으므로 통계적으로 유의하다.
llr = -2 * (-17.186 + 10.797) # 검정 통계량  12.78
1 - chi2.cdf(llr, 6) # p-value인 0.0467

#문제 3. 유의수준이 0.2를 기준으로 통계적으로 유의한 변수는 몇개이며, 어느 변수 인가요?
#P>|z|가 유의수준 0.2보다 작은 변수는 LI, TEMP다.

#문제 4. 다음 환자에 대한 오즈는 얼마인가요?

#CELL (골수의 세포성): 65%

#SMEAR (골수편의 백혈구 비율): 45%

#INFIL (골수의 백혈병 세포 침투 비율): 55%

#LI (골수 백혈병 세포의 라벨링 인덱스): 1.2

#BLAST (말초혈액의 백혈병 세포 수): 1.1세포/μL

#TEMP (치료 시작 전 최고 체온): 0.9

#오즈값구하기 0.0381748712694388
my_odds = np.exp(64.2581 +30.8301*0.65 + 24.686316*0.45 -24.9745*0.55 +4.3605*1.2 -0.0115*1.1 -100.1734*0.9)


#문제 5. 위 환자의 혈액에서 백혈병 세포가 관측되지 않은 확률은 얼마인가요?
my_odds / (my_odds+1) # 백혈병 관측되지 않을 확률: 0.036771137816849764


#문제 6. TEMP 변수의 계수는 얼마이며, 해당 계수를 사용해서 TEMP 변수가 백혈병 치료에 대한 영향을 설명하시오.
#TEMP 변수의 계수는 -100.1734이다.
#체온이 1 올라가면 로그 오즈는 100.1734만큼 감소하며, 백혈병 상태에 도달할 가능성이 크게 줄어든다


#문제 7. CELL 변수의 99% 오즈비에 대한 신뢰구간을 구하시오.

cell_beta = 30.830
z = 2.58
std_err = 52.135

upper = cell_beta + z * std_err
lower = cell_beta - z * std_err

upper #165.3383
lower #-103.6783


#문제 8. 주어진 데이터에 대하여 로지스틱 회귀 모델의 예측 확률을 구한 후, 50% 이상인 경우 1로 처리하여, 혼동 행렬를 구하시오.
from sklearn.metrics import confusion_matrix

# 1. 모델을 사용하여 예측 확률을 계산
pred_probs = model.predict(df)

# 2. 50% 기준으로 이진화 (0 또는 1로 변환)
predictions = [1 if prob >= 0.5 else 0 for prob in pred_probs]

# 3. 실제 값 (df['REMISS'])과 예측 값 (predictions) 비교하여 혼동 행렬 계산
conf_matrix = confusion_matrix(df['REMISS'], predictions)

# 혼동 행렬 출력
conf_matrix


#문제 9. 해당 모델의 Accuracy는 얼마인가요?
(15+5)/(15+3+4+5)


#문제 10. 해당 모델의 F1 Score를 구하세요.
precision = 5/(5+3)
recall = 5/(5+4)
F1_score = 2* (precision*recall/(precision + recall)) #0.5882352941176471