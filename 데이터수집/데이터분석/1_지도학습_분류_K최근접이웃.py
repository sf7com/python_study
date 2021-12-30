#1_지도학습_분류_K최근접이웃.py
#지도학습 : 정답이 있는 데이터를 통해서 알고리즘을 스스로 학습하는 시스템
#학습데이터 : Feature ex) 물고기의 길이, 너비, 무게, 등
#정답데이터 : Target ex) 물고기 종

#K최근접이웃 알고리즘
#어떤 데이터에 대한 답을 구할 때 주위의 다른 데이터를 보고
#다수를 차지하는 것을 정답으로 사용
#새로운 데이터에 대해 예측(분류)할 때는 가장 가까운 거리의 데이터를 
#보고 판단

#장점 : 단순하고 효율적이다
#단점 : 데이터가 많아지면 많은 메모리가 필요, 직선계산에 많은 시간이 든다.

#머신러닝 프로세스
#(0) 문제 정의(분류 주체, 예측하려는 대상)
#(1) 데이터 수집
#(2) 데이터 전처리(결측치 처리 등) 및 탐색(데이터 시각화 등)
#(3) 분석모델 구축 및 훈련
#(4) 모델 평가 및 결과분석 -> 성능 최적화
#(5) 모델 활용
#----------------------------
#머신러닝 유명 사이트 : 캐글(머신러닝 학습 데이터 다수 있고, 알고리즘)
#                     경진대회 진행

#-------------------------------------------------------
#도미와 빙어를 길이와 무게에 따른 분류

#1) 데이터 수집 후 데이터 로드
#도미 데이터 35개
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
len(bream_length)

#빙어 데이터 14개
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
len(smelt_length)

#2) 데이터 전처리 및 탐색
#2-1) 탐색 - 데이터 시각화
import matplotlib.pyplot as plt
#2차원 데이터 시각화 - 산점도 
plt.scatter(bream_length, bream_weight, label='bream')
plt.scatter(smelt_length, smelt_weight, label='smelt')
plt.xlabel("length")
plt.ylabel("weight")
plt.legend()
plt.grid(True)
plt.show()

#2-2) 전처리 - 데이터 병합
#훈련데이터(Features)
length = bream_length + smelt_length
weight = bream_weight + smelt_weight
fishData = [(l,w) for l,w in zip(length, weight)]
fishData
#타겟 데이터 생성(Targets)
fish_target = ['bream']*len(bream_length)+['smelt']*len(smelt_length)
fish_target
#---훈련데이터와 타겟데이터의 인덱스 위치는 같아야한다.

#훈련데이터/테스트 데이터 분할
#머신러닝 관련 모듈 : sklearn
#pip install sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = \
        train_test_split(fishData, fish_target, test_size=0.3,
            random_state=52)
#훈련 데이터 : 머신러닝 모델이 학습할때 쓰는 데이터
#테스트 데이터 : 머신러닝 모델의 성능 측정할 때 쓰는 데이터
#test_size : 테스터 데이터 비율 
# (ex) 0.3인경우 훈련데이터 70%, 테스트 데이터 30%)
X_train
X_test
Y_train
Y_test

#3) 분석모델 구축 및 훈련
from sklearn.neighbors import KNeighborsClassifier #분류모델
model = KNeighborsClassifier()
#모델훈련
model.fit(X_train, Y_train)

#4) 모델 평가 및 결과분석
#정확도
model.score(X_train, Y_train) #1.0 => 100%
model.score(X_test, Y_test) #1.0 => 100%

#5) 모델활용 - 새로운 데이터 예측
new_len = 30
new_weight = 600
model.predict([(new_len, new_weight)]) #array(['bream'], dtype='<U5')
plt.scatter(new_len, new_weight, label='new_fish', marker='^')
plt.scatter(bream_length, bream_weight, label='bream')
plt.scatter(smelt_length, smelt_weight, label='smelt')
plt.xlabel("length")
plt.ylabel("weight")
plt.legend()
plt.grid(True)
plt.show()
#----------------------------------------------------------------
#KNN 모델의 하이퍼파라미터
#하이퍼파라미터 : 개발자가 직접 값을 넣어줘야하는 파라미터
#n_neighbors : 이웃의 개수 지정 (기본값 : 5)
#p : 거리재는 방법(1:맨허튼거리, 2:유클리디안거리, 기본값 : 2)
#n_jobs : CPU코어 지정(-1:모든 코어 사용, 기본값 1)
#----------------------------------------------------------------
#하이퍼파라미터 값에 따라 모델의 성능이 달라지므로 최적화 필요
#일반적으로 디폴트값 쓰면 성능이 괜찮게 나온다.

#이웃 10개로 새로운 데이터판단
model10 = KNeighborsClassifier(n_neighbors=10)
model10.fit(X_train, Y_train)
model10.score(X_test, Y_test) #1.0
#---------------------------------------------------------------------
#데이터의 정규화(표준화) 필요성
#Z표준화(평균 0, 표준편차 1), 편찻값(평균 50, 표준편차 10)
#=> 데이터의 단위가 사라지고, 서로 다른 데이터의 상대적 위치를 파악하기 좋음

#데이터 표준화를 안한 경우
new_len = 25
new_weight = 150
model.predict([(new_len, new_weight)]) #array(['smelt'], dtype='<U5')
plt.scatter(new_len, new_weight, label='new_fish', marker='^')
plt.scatter(bream_length, bream_weight, label='bream')
plt.scatter(smelt_length, smelt_weight, label='smelt')
plt.xlabel("length")
plt.ylabel("weight")
plt.legend()
plt.grid(True)
plt.show()

#X,Y축 스케일 똑같이 표현
plt.scatter(new_len, new_weight, label='new_fish', marker='^')
plt.scatter(bream_length, bream_weight, label='bream')
plt.scatter(smelt_length, smelt_weight, label='smelt')
plt.xlabel("length")
plt.ylabel("weight")
plt.legend()
plt.xlim(0,1000)
plt.grid(True)
plt.show()

#표준화를 해주는 클래스
from sklearn.preprocessing import StandardScaler
ss = StandardScaler() #평균 0, 표준편차1
#ss.fit() 훈련에 맞춰 표준화 작업
#Z = (x-평균값)/표준편차 
#평균값과 표준편차값을 훈련데이터의 평균값과 표준편차값으로 활용
ss.fit(X_train) 
X_train_scaled = ss.transform(X_train)
import numpy as np
np.mean(X_train_scaled), np.std(X_train_scaled)
X_train_scaled = (X_train - np.mean(X_train))/np.std(X_train_scaled)
#(2.938825653419532e-17, 0.9999999999999999)

X_test_scaled = ss.transform(X_test)
model.fit(X_train_scaled, Y_train)
model.score(X_test_scaled, Y_test) #1.0

#새로운 데이터 예측시에도 표준화 작업을 해줘야한다.
#모델훈련을 표준화된 데이터로 했기때문에
new_len, new_weight 
predict_data = ss.transform([(new_len, new_weight)])
predict_data
model.predict(predict_data) #array(['bream'], dtype='<U5')

predict_data #array([[-0.0849956 , -0.78184947]])
plt.scatter(predict_data[0][0], predict_data[0][1], 
            label='new_fish', marker='^')
plt.scatter([x[0] for x in X_train_scaled], 
                [x[1] for x in X_train_scaled], label='train_data')
plt.xlabel("length")
plt.ylabel("weight")
plt.legend()
plt.grid(True)
plt.show()




#일반알고리즘 VS 머신러닝 VS AI
#일반알고리즘 : 개발자(연구자) 알고리즘을 만들어야한다.
#             ex) 도미와 빙어를 분류하는 기준을 우리가
#                직접 생각하고 고안해야한다.
#             길이가 30cm이상, 이면서 무게가 500kg이상 도미
            
#머신러닝 :  데이터를 통해서 학습해서 기준을 만들어준다.
#            분류, 예측을 알고리즘을 만들어준다.

#AI      : 가장 큰 범위, AI개발하는데 있어서 머신러닝 알고리즘
