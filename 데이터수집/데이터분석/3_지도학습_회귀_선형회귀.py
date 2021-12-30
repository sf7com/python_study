#3_지도학습_회귀_선형회귀.py
#선형회귀 : 대표적인 회귀 알고리즘
#모델이 비교적 간단하면서 모델 성능이 뛰어남
#변수(Feature)들의 중요성 정도도 파악 가능

#K최근접이웃의 단점을 해결
#(훈련세트를 벗어난 데이터도 예측을 잘함)
#----------------------------------------------
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
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_scaled, Y_train)

#4) 모델평가 및 결과분석
model.score(X_train_scaled, Y_train) #0.9371
model.score(X_test_scaled, Y_test) #0.8324
#과대적합
#=> 규제(나중에 배움)

#선형식의 회귀계수값(기울기값), 절편값
model.coef_, model.intercept_
#(array([39.27726005]), -718.4390718914794)
#Y=39.3X-718.4
#무게 = 39.3*길이-718.4
#선형식은 직선
#직선 Y=aX+b
#길이 30인 농어의 무게
#무게 = 39.3*30 -718.4

#(array([342.69318293]), 424.02564102564094)
#무게 = 342.7*길이 + 424.0 
#길이 10인 농어의 무게를 예측
#무게 = 342.7*10 + 424.0
#무게 = 342.7*(길이의 표준화점수) + 424.0 
#훈련데이터를 표준화된 데이터로 훈련했기때문에 Feature데이터가 표준화된값

#5) 모델활용
#새로운 데이터 예측
x = [50]
x_scaled = ss.transform([x]) #x가 1차원데이터여서 [x] 2차원데이터
y_pred = model.predict(x_scaled)  
y_pred #array([1245.42393074])

xs = np.arange(15,50).reshape(-1,1)
xs_scaled = ss.transform(xs)
ys = model.predict(xs_scaled)
plt.plot(xs,ys, label='reg line', c='g')
plt.scatter(X_train, Y_train, label='train')
plt.scatter(x, y_pred, label='50cm fish', marker="^")
plt.xlabel('length')
plt.ylabel('weight')
plt.legend()
plt.grid(True)
plt.show()
#_--------------------------------------------------------------
#다항회귀
#길이의 제곱 데이터를 추가
#              np.column_stack() 데이터를 열방향으로 결합
X_train_poly = np.column_stack((X_train**2, X_train))
X_train_poly

X_test_poly = np.column_stack((X_test**2, X_test))

X_train_poly.shape #(39, 2)
X_test_poly.shape #(17, 2)

#표준화
ss = StandardScaler()
ss.fit(X_train_poly)
X_train_poly_scaled = ss.transform(X_train_poly)
X_test_poly_scaled = ss.transform(X_test_poly)

#모델훈련
model2 = LinearRegression()
model2.fit(X_train_poly_scaled, Y_train)

#모델성능평가
model2.score(X_train_poly_scaled, Y_train) #0.96893, 기존 0.93
model2.score(X_test_poly_scaled, Y_test) #0.97930, 기존 0.83

#회귀계수값 및 절편
model2.coef_, model2.intercept_
#(array([ 532.25217837, -185.80632042]), 424.0256410256409)
#무게 = 532.3*(길이제곱의 표준화값)-185.8*(길이의 표준화값) + 424.0
#ex) 무게 = 532.3*(10*10)-185*10+424
#결과 시각화
#2차식의 곡선 데이터
xs = np.arange(15,50).reshape(-1,1)
xs_poly = np.column_stack((xs**2, xs))
xs_poly_scaled = ss.transform(xs_poly)
ys = model2.predict(xs_poly_scaled)

plt.plot(xs,ys, label='reg line', c='g')
plt.scatter(X_train, Y_train, label='train')

plt.xlabel('length')
plt.ylabel('weight')
plt.legend()
plt.grid(True)
plt.show()
