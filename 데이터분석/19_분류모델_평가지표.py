import pandas as pd
df = pd.read_csv('http://javaspecialist.co.kr/pds/382')
df.head()
df.info()
#    CUST_ID  y_true  y_pred
# 0       37       0       0
# 1       51       0       0
# 2       60       0       0
# 3       65       0       0
# 4       73       0       0
#1) 정확도
from sklearn.metrics import accuracy_score
accuracy_score(df['y_true'], df['y_pred']) #0.94255
#(TP+TN)/(TP+TN+FP+FN)
#TP : True-Positive 실제 참인것을 참이라고 예측
#TN : True-Negative 실제 거짓인것을 거짓이라고 예측
#FP : False-Positive 실제 거짓인것을 참이라고 예측
#FN : False-Negative 실제 참인것을 거짓이라고 예측
#(정리 : T 맞춘것, F 못맞춤, P 참, N 거짓)

#2) 정밀도
from sklearn.metrics import precision_score
precision_score(df['y_true'], df['y_pred']) #0.7777
#TP/(TP+FP)
#예측한것 중에 실제로 참인것을 참이라고 예측한 비율

#3) 민감도(재현율, recall)
from sklearn.metrics import recall_score
recall_score(df['y_true'],df['y_pred']) #0.48734
#실제 참인것중에 참이라고 예측한 비율

#4) F-score
from sklearn.metrics import f1_score
f1_score(df['y_true'],df['y_pred']) #0.59922
#민감도와 정밀도의 조화평균값 (두 지표를 결합해서 산출한 점수)

#특정한 타겟만 더 잘 분류하고 싶은 경우
#정확도 지표를 사용하지 않고 주로 F1스코어 활용
#모델을 최적할때도 f1스코어값이 높을 수 있도록 조정

#알고리즘 종류는 많아도 역할은 같다.
#성능지표가 좋은 알고리즘을 사용하면 됨.
#분류, 예측(회귀)

#딥러닝은 인간의 신경망을 모방한 머신러닝 알고리즘 중 한 분야이다.
#비정형 데이터 분류 혹은 예측 사용
#CNN : 이미지 분류
#이미지의 패턴학습
#--------------------------------------------------