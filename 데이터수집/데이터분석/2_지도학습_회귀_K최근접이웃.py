#2_지도학습_회귀_K최근접이웃.py
#회귀는 데이터를 예측
#K최근접이웃회귀 : 이웃한 샘플 데이터들의 평균값을 내어 새로운 데이터의
#                타겟값을 도출한다.

#단점 : 가장 가까운 데이터들의 평균값을 구하므로, 훈련세트의 범위를
#      벗어나는 데이터는 예측 성능이 떨어짐
#---------------------------------------------------------------
#0) 농어의 길이를 통해 농어의 무게를 예측
#1) 데이터 수집
import numpy as np
perch_length = np.array(
    [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 
     21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 
     22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 
     27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 
     36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 
     40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
     )
perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 
     1000.0, 1000.0]
     )

#Regression 회귀(예측)
#회귀 -> 평균값으로 회귀한다.
#예측모델 -> 예측모델의 값으로 회귀한다. 예측(오차)

#2) 데이터 탐색 및 전처리
#2-1) 시각화
import matplotlib.pyplot as plt
plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.grid(True)
plt.show()

#2-2) 훈련/테스트 데이터 분활
#훈련데이터/테스트 데이터 비율 7:3, 8:2 (일반적인 비율)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = \
        train_test_split(perch_length, perch_weight, test_size=0.3,
                random_state=42)
X_train.shape #(39,) #1차원 데이터
#머신러닝 모델의 훈련데이터 행렬이 되어야한다.
X_train = X_train.reshape(39, 1)
X_train.shape #(39, 1) #행렬 2차원 데이터
X_train
X_test = X_test.reshape(-1,1) #배열크기에 -1하면 원소의 개수로 대입
X_test.shape
#훈련데이터를 2차원 데이터로 표현하는 이유는
#일반적으로 Feature의 개수는 1개 이상

#2-3) 데이터 표준화
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_train) 
X_train_scaled = ss.transform(X_train)
X_test_scaled = ss.transform(X_test)

#3) 분석모델 구축 및 훈련
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()
model.fit(X_train_scaled, Y_train)

#4) 모델 평가 및 결과분석
#정확도 : 테스트 세트에 있는 타겟을 정확하게 맞춘 비율
#회귀모델에서 정확도를 구할 수가 없다.(정확한 수치값을 딱 맞추는것이 불가능)
#결정계수(R2)값을 통해 모델성능 평가한다.
#결정계수값은 0~1사이의 값을 갖는다. 1에 가까울술고 예측모델과 데이터가
#잘 들어맞는다. (예측한 데이터와 실제데이터간의 오차가 적다)
model.score(X_test_scaled, Y_test) #0.9930487320507357
test_pred = model.predict(X_test_scaled)
test_pred #테스트 데이터 예측 무게값
import pandas as pd
df = pd.DataFrame(columns=['예측값', '실제값'])
df['예측값'] = test_pred
df['실제값'] = Y_test
df

#오차의 평균값
from sklearn.metrics import mean_absolute_error, mean_squared_error
#데이터 산포도 지표와 비슷
#예측모델로부터 얼마나 오차가 있는지 정도를 나타냄
#mean_absolute_error : (편차 절대값의 평균값)
#예측한 데이터와 실제 데이터간의 차이의 데이터들의 절댓값을 취한후 평균

#mean_squared_error : (분산과 개념이 비슷)
#mean((오차)*(오차))
#예측한 데이터와 실제 데이터간의 차이의 제곱한 데이터들의 평균값 
mean_absolute_error(Y_test, test_pred) #17.564705
mean_squared_error(Y_test, test_pred) # 627.49294
np.sqrt(mean_squared_error(Y_test, test_pred)) #25.049
#평균 오차는 대략 17~25kg정도 됨을 나타냄
#어떤 데이터를 예측했는데 오차가 평균적으로 17~25kg 됨
#평균 오차이기 때문에 실제로는 오차가 더 클수도 있다.
#------------------------------------------------------------------------
model.score(X_train_scaled, Y_train) #0.968014147
model.score(X_test_scaled, Y_test) #0.9930487
#훈련데이터로 모델을 만들었기 때문에 일반적으로
#훈련데이터의 점수가 더 높아야 한다.

#과소적합과 과대적합
#과소적합 : 훈련 세트의 점수보다 테스트 세트의 점수가 높게 나온 경우,
#          혹은 두 점수 모두 낮은 경우 해당
#          발생원인 - 훈련데이터가 부족한 경우, 모델이 너무 단순한 경우
#          해결책 : 훈련 데이터 늘려주거나, 모델을 복잡하게 만든다.
#                  훈련양을 늘려준다.

#과대적합 : 훈련세트에서 점수가 좋게 나왔는데, 테스트 데이터에서 점수가
#          나쁜 경우
#          발생원인 : 모델이 너무 복잡하거나 모델이 훈련데이터에 지나치게
#                    적합되어있어서 일반적인 예측 성능이 떨어지는 경우
#          해결책 :  규제, 드롭아웃, 하이퍼파라미터 변경등

#과소적합 해결
model.n_neighbors = 3 #전 모델은 디폴트 값이 5였음
model.fit(X_train_scaled, Y_train)
model.score(X_train_scaled, Y_train)  #0.97939
model.score(X_test_scaled, Y_test)  #0.9766
#이상적으로 학습됨

#n_neighbors 값에 따라 예측모델 성능 시각화
#예측 데이터 
x = np.arange(5, 45).reshape(-1,1) #예측하려는 데이터(농어의 길이 데이터)
x_scaled = ss.transform(x) 
x_scaled
#n=1,3,5,10 일때 예측 결과를 시각화
for i,n in enumerate([1,3,5,10]) :
    model.n_neighbors = n
    model.fit(X_train_scaled, Y_train)
    prediction = model.predict(x_scaled)
    plt.scatter(X_train, Y_train, label='train_data', c='b')
    plt.plot(x, prediction, label='pred_data', c='r')
    plt.title(f'n_neighbors = {n}')
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.legend()
    plt.show()

#2X2 그래프 그리기
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10),
            sharex=True, sharey=True)
for i,n in enumerate([1,3,5,10]) :
    model.n_neighbors = n
    model.fit(X_train_scaled, Y_train)
    prediction = model.predict(x_scaled)
    ax = axes[i//2, i%2]
    ax.scatter(X_train, Y_train, label='train_data', color='b')
    ax.plot(x, prediction, label='pred_data', color='r')
    ax.set_title(f'n_neighbors = {n}')
    ax.legend()
plt.tight_layout()
plt.show()
#---------------------------------------------------------------------
#K최근접이웃 알고리즘 한계
np.max(X_train) #44.0cm
x = [50] #50cm 농어의 무게를 예측
x_scaled = ss.transform([x])
y_pred = model.predict([x])
plt.scatter(X_train, Y_train, label='train')
plt.scatter(x, y_pred, label='pred', marker="^")
plt.xlabel('length')
plt.ylabel('weight')
plt.legend()
plt.grid(True)
plt.show()
