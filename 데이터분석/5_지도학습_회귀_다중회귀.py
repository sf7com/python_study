#다중회귀
#여러개의 특성(독립변수)을 사용한 선형회귀를 다중회귀라 부름

#특성이 1개일 때 모델이 학습하는 것은 직선 (시각화)
#특성이 2개일 때 모델이 학습하는 것은 평면 (시각화)
#특성이 3개 이상일 때 모델이 학습하는 하는 것은 고차원

#특성공학 : 기존의 특성(독립변수)를 사용해서 새로운 특성을 뽑아내는 작업
#Ex) 길이-> 길이제곱,  (길이,높이)->길이*높이
#---------------------------------------------------------------------------
import pandas as pd
import numpy as np
#0) 문제정의 : 길이,높이,너비 데이터로 농어의 무게를 예측
#1) 데이터 수집
df = pd.read_csv('./데이터분석/data/perch_full.csv')
df.head()
df.info()  #농어의 길이, 높이, 너비

perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 
     1000.0, 1000.0]
     )
df['weight'] = perch_weight
df.head()

#2) 데이터 탐색 및 전처리
#2-1) 훈련/테스트 데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:, :-1],
    df.iloc[:,-1], random_state=32)

#2-2) 표준화
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_train)
X_train_scaled = ss.transform(X_train)
X_test_scaled = ss.transform(X_test)

#3) 분석모델 구축 및 훈련
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_scaled, Y_train) 

#4) 모델 평가 및 결과분석
model.score(X_train_scaled, Y_train) #0.93767
model.score(X_test_scaled, Y_test) #0.9347

#-------------------------------------------------------------------------
#특성공학으로 데이터 추가
#사이킷런 변환기
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures()
poly.fit([(2,3)])
poly.transform([(2,3)])
#각 특성을 제곱한 항을 추가하고 서로 곱한 항을 추가한
#array([[1., 2., 3., 4., 6., 9.]])
poly = PolynomialFeatures(degree=3, include_bias=False) #최대 3제곱, 절편은 없앰
poly.fit([(2,3)])
poly.transform([(2,3)])
#각 특성을 제곱한 항을 추가하고 서로 곱한 항을 추가한
#각 특성의 제곱*다른특성, 각 특성의 3제곱 추가
#array([[ 2.,  3.,  4.,  6.,  9.,  8., 12., 18., 27.]])

#2) 데이터 탐색 및 전처리
#2-1) 훈련/테스트 데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:, :-1],
    df.iloc[:,-1], random_state=32)

#2-2) 특성공학 degree 2
poly = PolynomialFeatures(include_bias=False)
poly.fit(X_train) #sklearn -> 데이터의 훈련 fit함수
#                           -> 변환 transform
X_train_poly = poly.transform(X_train)
X_test_poly = poly.transform(X_test)
X_train_poly.shape #(42, 9) 
poly.get_feature_names() 
#['x0', 'x1', 'x2', 'x0^2', 'x0 x1', 'x0 x2', 'x1^2', 'x1 x2', 'x2^2']
#[길이, 높이, 너비,길이제곱,길이*높이,길이*너비,높이제곱, 높이*너비, 너비제곱]

#2-3) 표준화
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_train_poly) #평균 0, 표준편차 1 표준화 점수
#각 데이터의 Z 점수 = (X-평균)/표준편차
X_train_poly_scaled = ss.transform(X_train_poly)

#train 데이터의 평균, 표준편차 값으로 표준화 
#모델 -> train 훈련을 함
X_test_poly_scaled = ss.transform(X_test_poly)

#3) 분석모델 구축 및 훈련
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_poly_scaled, Y_train) 

#4) 모델 평가 및 결과분석
model.score(X_train_poly_scaled, Y_train) # #0.99146751, 이전 0.93767
model.score(X_test_poly_scaled, Y_test) #0.974054, 이전 0.9347
model.coef_, model.intercept_
# (array([ -279.46330089,   208.68905494,  -120.52114167,  1949.18826398,
#        -3430.45431611,   246.53332938,  1263.46137517,   861.33534735, 
#         -373.68701167]), 350.5571428571427)
#--------------------------------------------------------------------
#과제 적합 해결 방법 : 규제
#릿지와 라쏘 : 선형회귀 모델에서 규제를 추가한 모델
#회귀계수에 대한 제약조건을 추가함으로써 모형이 과도하게 최적화되는(과적합) 막는 방법
#ex) 계수의 크기가 커지면 과적합 되는 경우 -> 계수의 크기를 제한한다.

#(1) 릿지 : 계수를 제곱한 값을 기준으로 규제를 적용
#(2) 랏쏘 : 계수의 절댓값을 기준으로 규제를 적용
#(3) 엘라스틱넷 : 추정계수의 절댓값 합과 제곱합을 동시에 최소로 하는 것(큰 데이터셋에 효과적)
#일반적으로 릿지 모델 많이 사용

#릿지회귀
from sklearn.linear_model import Ridge
ridge = Ridge() #alpha 하이퍼파라미터 : 규제양 (기본값 1)
ridge.fit(X_train_poly_scaled, Y_train)
ridge.score(X_train_poly_scaled, Y_train) #0.98875, 규제전 0.9914
ridge.score(X_test_poly_scaled, Y_test) #0.977222, 규제전 0.9740
ridge.coef_, ridge.intercept_
# (array([-27.67389849, -29.5685773 , -22.6185933 ,  69.96338106,
#         65.96525319,  69.1741496 ,  61.66123147,  65.27106656,
#         66.85084493]), 350.55714285714276)

#규제양 - alpha
#alpha 크면 : 규제 강도가 커짐 -> 회귀계수값 더 줄이게됨 -> 과소적합되도록 유도
#alpha 작으면 : 규제 작아짐 -> 회귀계수값 커짐 -> 과대적합도리 가능성이 커짐
#적절한 규제양을 찾아야한다. (데이터 엔지니어 역할)

#최적의 alpha값 찾기
import matplotlib.pyplot as plt
alphaList = [0.001, 0.01, 0.1, 1, 10, 100] #이 중 최적의 알파 찾기
train_score = []
test_score = []
for alpha in alphaList :
    ridge = Ridge(alpha=alpha) #alpha 하이퍼파라미터 : 규제양 (기본값 1)
    ridge.fit(X_train_poly_scaled, Y_train)
    train_s = ridge.score(X_train_poly_scaled, Y_train) 
    test_s = ridge.score(X_test_poly_scaled, Y_test) 
    train_score.append(train_s)
    test_score.append(test_s)
train_score
test_score

#alphaList =      [0.001, 0.01, 0.1, 1, 10, 100]
#밑이 10인 로그함수  -3,    -2,   -1,  0, 1, 2
#x = log10(0.001) = -3
#0.001= 10^-3

#2^x = 4
#x = log2(2^2) = 2

plt.plot(np.log10(alphaList), train_score, label='train')
plt.plot(np.log10(alphaList), test_score, label='test')
plt.xticks(np.log10(alphaList), labels=alphaList)
plt.xlabel("alpha")
plt.ylabel(r"$R^2$") #수학 기호 표시 가능
plt.grid(True)
plt.legend()
plt.show()

#---------------------------------------------
#최적의 alpha 0.1
ridge = Ridge(alpha=0.1) #alpha 하이퍼파라미터 : 규제양 (기본값 1)
ridge.fit(X_train_poly_scaled, Y_train)
ridge.score(X_train_poly_scaled, Y_train)  #0.99078
ridge.score(X_test_poly_scaled, Y_test)  #0.9795

#오차의 평균값
from sklearn.metrics import mean_absolute_error
y_pred = ridge.predict(X_test_poly_scaled)
mean_absolute_error(Y_test, y_pred) #36.30336g 

#모델활용 
new_length = 44
new_height = 14
new_width = 5
x = [new_length, new_height, new_width]
#모델 훈련시 - 데이터 Poly화 및 스케일 함
#예측 데이터도 똑같이 Poly화 및 스케일 해야함
x_poly = poly.transform([x])
x_poly_scaled = ss.transform(x_poly)
ridge.predict(x_poly_scaled) #array([884.48749695]) 884.5g

ridge.coef_
# array([-49.82170897, -47.98938447, -64.3612914 ,  96.10114077,
#         75.20818441,  83.4197923 ,  63.83916967,  78.22554439,
#         84.87040491])
#회귀계수값이 크면 결과에 더 영향을 많이 주는 feature임
#식 자체를 만들 수 있어 많이 사용함