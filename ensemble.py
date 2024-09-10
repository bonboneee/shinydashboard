from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bagging_model = BaggingClassifier(DecisionTreeClassifier(),
                                  n_estimators=100,
                                  max_samples=100,
                                  n_jobs=-1,
                                  random_state=42)

# n_estimator: Bagging에 사용될 모델 개수
# max_sample: 데이터셋 만들 때 뽑을 표본크기
# bagging_model.fit(X_train, y_train)