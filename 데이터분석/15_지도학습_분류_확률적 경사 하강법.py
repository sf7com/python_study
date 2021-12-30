#15_지도학습_분류_확률적 경사 하강법
#점진적인 학습
#전체 데이터를 한꺼번에 모두 불러와서 학습하면 컴퓨터 메모리 소모 크고, 느리다.
#전체 데이터를 한꺼번에 못 불러올 수도 있음

#확률적 경사하강법 (딥러닝 학습 방법)
#경사하강법 : 최적의 해를 찾기 위해(손실함수의 오차를 최소화하기 위해)
#            손실 함수의 기울기를 이용하여 가중치를 업데이트 하는 방법
#오차 함수의 미분값이 0인 지점을 찾는게 목적
#기울기를 구하고 가중치를 계속 업데이트 하여 미분값이 가장 작은 지점에 이를 때 까지 반복

#손실함수 : 머신러닝 알고리즘의 오차, 얼마나 부정확한지 척도
#평균제곱오차와 교차엔트로피 오차등을 사용
#평균제곱오차(예측값과 실제값 사이의 차이제곱의 평균값)

#다항회귀
#Y = a1X1 + a2X2 + a3X3 ... +b
#머신러닝 모델이 학습해서 찾는 것은 (a1, a2, a3,...b) 오차가 가장작은 가중치를 찾는게 목적
#오차 = 정답 - 예측

#확률적 경사하강법
#전체 데이터 중 하나의 랜덤한 샘플 데이터를 통해 경사하강법 진행

#에포크(epoch) -훈련 횟수
#훈련세트에 샘플 데이터를 채워 넣고 훈련 세트를 한번 모두 사용하는 과정을 에포크
#일반적으로 경사 하강법은 수십, 수백번 이상의 에포크를 수행
#ex) epoch 100이면 10000개의 데이터중 일부를 추출해서 훈련 진행(100번).

#미니배치 - 한번 가중치 업데이트시 필요한 데이터 개수
#1개의 샘플데이터가 아니라 무작위로 몇 개의 샘플 데이터를 선택해 경사하강법 진행
# 1000개 데이터 중 미니배치값 10, 훈련횟수 100, 총 10번의 epoch 

#배치 경사하강법
#모든 데이터를 사용해 경사하강법 훈련
#가장 안정적인 방법이 될 수 있지만, 컴퓨터 자원 소모가 크다.

#1) 데이터 로드
import pandas as pd
df = pd.read_csv('./데이터분석/data/Fish.csv')
df.head()
df.info()

#2) 데이터 전처리
#2-1) 훈련/테스트 데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:, 1:], df.iloc[:,0], test_size=0.2, random_state=42)
#2-2) 데이터 표준화
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_train)
X_train_scaled = ss.transform(X_train)
X_test_scaled = ss.transform(X_test)

#3) 분석모델 구축 및 훈련
#확률적경사하강법(SGD) - 훈련세트에서 1개씩 랜덤으로 샘플 데이터를 꺼내어 경사하강법
#                       진행 후 가중치 업데이트
#손실함수 log(로지스틱 손실함수), 기본값 hinge(서포트 벡터머신의 손실함수)
#max_iter : 에포크 횟수 10, 전체 훈련 세트를 10회 반복
from sklearn.linear_model import SGDClassifier
model = SGDClassifier(loss='log',max_iter=10, random_state=42)
model.fit(X_train_scaled, Y_train)

model.score(X_train_scaled,Y_train) #0.74803
model.score(X_test_scaled,Y_test) #0.84375
#과소적합 - 훈련이 덜 되었음.

#경사하강법 - 모델 재학습 할 수 있다.
#점진적인 학습
#기존 훈련 모델에서 1에포크씩 이어서 훈련
model.partial_fit(X_train_scaled, Y_train)
model.score(X_train_scaled,Y_train) #0.74803 =>0.74803
model.score(X_test_scaled,Y_test) #0.84375 =>0.8125
#score 올라갈수도 내려갈수도 있다.

#최적의 에포크 찾기
import numpy as np
model = SGDClassifier(loss='log', random_state=42)
train_scores = []
test_scores = []
classes = np.unique(Y_train)
classes
# array(['Bream', 'Parkki', 'Perch', 'Pike', 'Roach', 'Smelt', 'Whitefish'],dtype=object)

#partial fit만으로 훈련
for _ in range(0,300) :
    model.partial_fit(X_train_scaled, Y_train, classes=classes)
    train_scores.append(model.score(X_train_scaled,Y_train))
    test_scores.append(model.score(X_test_scaled,Y_test))

import matplotlib.pyplot as plt
plt.plot(np.arange(1,301), train_scores, label='train')
plt.plot(np.arange(1,301), test_scores, label='test')
plt.xlabel("epoch")
plt.ylabel("score")
plt.legend()
plt.grid(True)
plt.show()

#최적의 에포크 값 100 이상
from sklearn.linear_model import SGDClassifier
model = SGDClassifier(loss='log',max_iter=200, random_state=42, tol=None)
#tol 파라미터 : 오차값이 tol 이하가 되면 더이상 학습진행 안함.
model.fit(X_train_scaled, Y_train)
model.score(X_train_scaled,Y_train) #0.74803 =>0.95275
model.score(X_test_scaled,Y_test) #0.84375 =>0.90625
