#------------------------------------------------------
#K최근접 이웃 다중분류
#원리 : 샘플 주위에 있는 각각 클래스(타겟)별 데이터 개수로
#샘플 타겟값이 무엇인지 확률로 파악
#Ex) 예측하고 싶은 데이터 주위에 Bream 3마리, Perch 2마리, Smelt 1마리
#예측 데이터가 Bream인 확률 3/6=1/2, Perch일 확률 2/6, Smelt 1/6
#가장 높은 확률값을 타겟으롯 선정

import pandas as pd
import numpy as np
#1) 데이터 수집
df = pd.read_csv('./데이터분석/data/Fish.csv')
df.head()
df.info() #Null 값 없음

#2) 데이터 탐색 및 전처리
#타겟 값 확인
df['Species'].unique()
#array(['Bream', 'Roach', 'Whitefish', 'Parkki', 'Perch', 'Pike', 'Smelt']
#총 7개 종 분류
#2-1) 훈련/테스트 데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:, 1:],
                                df.iloc[:, 0], random_state=32)
#2-2) 데이터 정규화
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_train)
X_train_scaled = ss.transform(X_train)
X_test_scaled = ss.transform(X_test)

#3) 분석모델 구축 및 훈련
from sklearn.neighbors import KNeighborsClassifier

#n_neighbors=3 #주위 3개의 데이터를 통해서 새로운 데이터를 예측
model = KNeighborsClassifier(n_neighbors=3) 
model.fit(X_train_scaled, Y_train)
#4) 모델 평가 및 결과분석
model.score(X_train_scaled, Y_train) #0.873
model.score(X_test_scaled, Y_test) #0.9
#과소적합

#5) 모델 활용
model.predict(X_test_scaled[:5])
#array(['Bream', 'Perch', 'Bream', 'Pike', 'Pike'], dtype=object)
Y_test[:5]
#      Bream  Perch  Bream  Pike  Pike
#각 클래스(타겟)별 확률
model.classes_
proba = model.predict_proba(X_test_scaled[:5])
proba
# array(['Bream', 'Parkki', 'Perch', 'Pike', 'Roach', 'Smelt', 'Whitefish'],
# array([[1.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
#        [0.        , 0.        , 0.66666667, 0.        , 0.        0.        , 0.33333333],
#        [1.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
#        [0.        , 0.        , 0.        , 1.        , 0.        ,0.        , 0.        ],
#        [0.        , 0.        , 0.        , 1.        , 0.        ,0.        , 0.        ]])
#------------------------------------------------------------------------------------------
#로지스틱회귀 (분류모델)
#=> 회귀식을 통해서 분류를 한다.
#선형회귀와 동일하게 선형방정식을 학습한다.
#ex) Y = a*Weight+ b*length + c*diagnal + d*height + e*width + f
#Y는 타겟값이아니라 수치값이다.
#어떠한 값이든 다 나올 수 있다.
#나온 수치값을 토대로 각 클래스별 확률로 바꿔야함

#이진분류 (2개의 타겟만 분류 0~1 값으로 나오면 됨)
#0에 가까우면 A타겟, 1에 가까우면 B타겟 이런식으로 해석함
#위의 수치데이터를 시그모이드함수값으로 취한다.
#시그모이드함수(로지스틱 함수) =  1 / (1+e^(-Y))
# 0~1 사이의 값을 갖는다.

#다중분류
#선형회귀식은 클래스(타겟) 개수 만큼 학습
#ex)
#농어 = a1*Weight+ b1*length + c1*diagnal + d1*height + e1*width + f1
#빙어 = a2*Weight+ b2*length + c2*diagnal + d2*height + e2*width + f2
#화이트피쉬 = a3*Weight+ b3*length + c3*diagnal + d3*height + e3*width + f3
#....
#새로운 데이터의 타겟을 예측할 때 소프트맥스라는 함수를 통해서 확률로 표현
#농어의 확률 = exp(농어값) / exp(농어값)+exp(빙어값)+exp(화이트피쉬값)+....
#빙어의 확률 = exp(빙어값) / exp(농어값)+exp(빙어값)+exp(화이트피쉬값)+....
#모든 탓겟들의 합은 1
#농어의 확률 = (농어값) / (농어값)+(빙어값)+(화이트피쉬값)+....
#빙어의 확률 = (빙어값) / (농어값)+(빙어값)+(화이트피쉬값)+....
#지수함수 취하는 이유 : 값이 조금만 달라져오 차이가 커진다.
#특징이 부각되고 오차를 계산하기 좋아진다.
#확률값도 명확해진다.
#머신러닝이 잘 학습하기 위해서 소프트맥스함수를 통해 확률값으로 사용
#-----------------------------------------------------------------------
#이진분류 - 로지스틱회귀모델
#ex) 도미(Bream)와 빙어(Smelt)분류

#인덱스 마스킹 활용해서 원하는 데이터만 추출
bream_smelt = df[(df['Species']=='Bream') | (df['Species']=='Smelt')]
bream_smelt.head()
bream_smelt['Species'].unique()

#2-1) 훈련/테스트 데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(bream_smelt.iloc[:, 1:],
                                bream_smelt.iloc[:, 0], random_state=32)
#2-2) 데이터 정규화
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_train)
X_train_scaled = ss.transform(X_train)
X_test_scaled = ss.transform(X_test)

#3) 모델구축 및 훈련
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_scaled, Y_train)

#4) 모델 평가 및 결과분석
lr.score(X_train_scaled, Y_train) #1.0
lr.score(X_test_scaled, Y_test) #1.0
lr.coef_, lr.intercept_
# (array([[-0.5445108 , -0.7560181 , -0.78309416, -0.81065788, -0.77822997]]), 
# array([-2.95878514]))
#Y=-0.54*(무게 값)-0.76*(길이)-0.78*(대각선길이)-0.81*(높이)-0.77(너비)+b

lr.predict(X_test_scaled[:5])
#array(['Smelt', 'Bream', 'Smelt', 'Bream', 'Smelt']
Y_test[:5]
lr.predict_proba(X_test_scaled[:5])
lr.classes_
#array(['Bream', 'Smelt'],
# array([[1.05450939e-01, 8.94549061e-01],
#        [9.93971801e-01, 6.02819937e-03],
#        [3.68970474e-02, 9.63102953e-01],
#        [9.99492460e-01, 5.07540197e-04],
#        [1.00417450e-01, 8.99582550e-01]])
#각 데이터별 선형회귀값 
lr.decision_function(X_test_scaled[:5])
##Y=-0.54*(무게 값)-0.76*(길이)-0.78*(대각선길이)-0.81*(높이)-0.77(너비)+b
#array([ 2.13807394, -5.10526048,  3.26202878, -7.58542697,  2.19259482])
y = lr.decision_function(X_test_scaled[:5])
from scipy.special import expit #시그모이드함수
expit(y)
# array([8.94549061e-01, 6.02819937e-03, 9.63102953e-01, 5.07540197e-04,
#        8.99582550e-01])
#lr.predict_proba(X_test_scaled[:5]) 나온값과 동일
#-----------------------------------------------------------------------
#로지스틱회귀 다중분류
#각 클래스마다 선형방정식 생성
#각 클래스마다 y값을 얻어 소프트맥스 함수를 통해 확률 변환
#e_sum= e^y1+e^y2+.....+e^zn
#s1 = e^z1/e_sum (첫번째 클래스의 확률)
#s2 = e^z2/e_sum (두번째 클래스의 확률)

#로지스틱 회귀는 학습하기 위해 반복적인 알고리즘 수행
#대표적 하이퍼파라미터
#max_iter : 학습횟수 지정 (기본값 100)
#L2 규제 : 릿지회귀와 같이 계수의 제곱을 규제함
#C : 규제제어 (기본값 1), C값은 작을수록 규제가 커진다(릿지회귀와 반대)
#--------------------------------------------------------------------
import pandas as pd
import numpy as np
#1) 데이터 수집
df = pd.read_csv('./데이터분석/data/Fish.csv')
df.head()
df.info() #Null 값 없음

#2) 데이터 탐색 및 전처리
#타겟 값 확인
df['Species'].unique()
#array(['Bream', 'Roach', 'Whitefish', 'Parkki', 'Perch', 'Pike', 'Smelt']
#총 7개 종 분류
#2-1) 훈련/테스트 데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:, 1:],
                                df.iloc[:, 0], random_state=32)
#2-2) 데이터 정규화
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_train)
X_train_scaled = ss.transform(X_train)
X_test_scaled = ss.transform(X_test)

#3) 분석모델 구축 및 훈련
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=200, max_iter=1000) 
model.fit(X_train_scaled, Y_train)
#4) 모델 평가 및 결과분석
model.score(X_train_scaled, Y_train) #0.9831
model.score(X_test_scaled, Y_test) #0.975
#과소적합

#5) 모델 활용
model.predict(X_test_scaled[:5])
#array(['Bream', 'Perch', 'Bream', 'Pike', 'Pike']
Y_test[:5]
#      Bream  Perch  Bream  Pike  Pike
#각 클래스(타겟)별 확률
model.classes_
proba = model.predict_proba(X_test_scaled[:5])
proba

model.coef_ #타겟별 회귀계수값들 리스트로 저장됨
model.intercept_ #절편값, Y=ax+b
model.classes_

#           무게          길이           대각선길이  높이 너비
# array([[ -5.42325672,  -1.65649176,   8.00904021,  14.50512089,
#          -2.62667157], //Bream
#        [  2.4890931 ,  -2.63301881,  -9.39544082,  11.84643184,
#          -1.94508293], //Parkki
#        [  5.7495599 ,  21.52347485, -25.75048642, -11.44668748,
#           7.39979135], //Perch
#        [  0.07275894,   5.15051397,   5.28152068,  -4.68013495,
#          -2.15856712],
#        [ -3.16407231, -21.14521158,  17.71796211,  -3.62711458,
#           6.30648472],
#        [ -2.73888316,   4.505892  ,   4.08679623,  -9.4689469 ,
#         -10.04843373],
#        [  3.01480025,  -5.74515866,  -0.030392  ,   2.87133119,
#           3.07247929]])