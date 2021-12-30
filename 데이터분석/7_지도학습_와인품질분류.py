#와인속서을 이용해서 품질등급 예측(다중분류)

#1) 데이터 수집
from os import sep
import pandas as pd
red_df = pd.read_csv('./데이터분석/data/winequality-red.csv', sep=';')
white_df = pd.read_csv('./데이터분석/data/winequality-white.csv', sep=';')
red_df.head()
white_df.head()

red_df.info() #결측치 없음, 1599
white_df.info() #결측치 없음, 4898

#통계 : t검정 -  두 그룹간의 평균에 차이가 있는지 비교하는 분석방법
#레드와인, 화이트와인의 품질 평균값간의 차이가 있는지 비교
red_df['quality'].mean() #5.63
red_df['quality'].std() #0.807 (표준편차 : 평균으로부터 평균오차)
white_df['quality'].mean() #5.877
white_df['quality'].std() #0.8856 (표준편차 : 평균으로부터 평균오차)

from scipy import stats #대표적인 통계 라이브러리
#               두집단간의 평균값의 차이가 있는지 유무를 판단하는 함수
tResult = stats.ttest_ind(red_df['quality'], white_df['quality'], 
                    equal_var=False) #equal_var 두집단의 분산이 같은지 여부
tResult
# Ttest_indResult(statistic=-10.149363059143164, pvalue=8.168348870049682e-24)
#통계학에서 일반적으로 p값이 0.05 미만 대립가설이 유의하다.
#대립가설 : 두집단간의 차이가 있다. 
#귀무가설 : 두집간간의 차이가 없다.
#-----------------------------------------------------------------
#와인 정보 통합
red_df.insert(0, column='type', value='red') #열 추가하는 메서드 
white_df.insert(0, column='type', value='white') #열 추가하는 메서드 
red_df.head()
white_df.head()

wind_df = pd.concat([red_df, white_df])
wind_df.head()
#-------------------------------------------------------------
#머신러닝 프로세스에 맞게 로지스틱회귀 모델 와인품질 예측
#type 열은 제외하고 나머지 속성값 활용 quality 다중분류
#어떤 피처가 중요한 변수인지 판단
#11시 50분까지 완성

#2) 데이터 탐색 및 전처리
#타겟 값 확인
wind_df['quality'].unique()
wind_df.info()
#array([5, 6, 7, 4, 8, 3, 9] #총 7개 종 분류
#2-1) 훈련/테스트 데이터 분할
#데이터 프레임 데이터를 가져오는 메서드
#loc[인덱스 이름:열이름]
#iloc[인덱스 번호:열번호]
#wind_df.iloc[:, 1:-1] => [인덱스 처음부터 끝까지, 열번호 1~끝전까지]
#wind_df.iloc[:, -1] =>  [인덱스 처음부터 끝까지, 마지막 열]
wind_df.info()
wind_df.head()
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(wind_df.iloc[:, 1:-1],
                                wind_df.iloc[:, -1], random_state=32, 
                                test_size=0.3)
#train_test_split(피쳐X, 타겟Y,...)

#원본데이터 [1,2,3,4,5,6,7,8,9]
#랜덤하지 않는 방식 훈련 [1,2,3,4,5,6,7], 테스트 [8,9]
#랜덤한 방식 훈련 [1,8,2,5,7,9,6], 테스트 [3,4]

#2-2) 데이터 정규화
from sklearn.preprocessing import StandardScaler
mean = X_train['fixed acidity'].mean()
std = X_train['fixed acidity'].std()
temp = (X_train['fixed acidity']-mean)/std
temp.mean() #4.6879881400490446e-17
#4.68*10^-17 =0.00000000000000468
temp.std() #1.0

ss = StandardScaler() #데이터 표준화, z표준화 = (x-평균값)/표준편차
#z표준화 데이터 평균이0, 표준편차 1
ss.fit(X_train)
X_train_scaled = ss.transform(X_train)
X_test_scaled = ss.transform(X_test)
X_train_scaled

#3) 분석모델 구축 및 훈련
from sklearn.linear_model import LogisticRegression
model = LogisticRegression() 
model.fit(X_train_scaled, Y_train)
#4) 모델 평가 및 결과분석
model.score(X_train_scaled, Y_train) #0.5472
model.score(X_test_scaled, Y_test) #0.536
#낮은 이유는 등급이 비슷한 것끼리 오차 있을 수 있다.
model.predict(X_test_scaled[:5])
#array([7, 6, 7, 5, 5], dtype=int64)
#  6, 8, 7, 4, 6
Y_test[:5]


#다중회귀모델로 등급을 예측, 분류(X)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_scaled, Y_train)
#4) 모델 평가 및 결과분석
model.score(X_train_scaled, Y_train) #0.5472
model.score(X_test_scaled, Y_test) #0.536

model.predict(X_test_scaled[:5])
#array([6.61427768, 5.98149805, 6.84649311, 5.58000209, 4.63381947])
#  6, 8, 7, 4, 6
Y_test[:5]

wind_df.info()
wind_df.head()
#type열 -> 질적변수, red, white-> 바로 분석데이터 쓸수는 없다.
#=> 더미변수 바꾼다. (0 red, 1:white)
pd.get_dummies(wind_df['type']) #red, white 칼럼이 생긴다.
dummpy_df = pd.get_dummies(wind_df)
dummpy_df.info() #12, 13열 type_red, type_white, 기존type열은 사라짐
dummpy_df.head()
X = dummpy_df.drop('quality', axis=1) #열삭제
X.info()
Y = dummpy_df['quality']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=32, 
                                test_size=0.3)
#2-2) 데이터 정규화
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_train)
X_train_scaled = ss.transform(X_train)
X_test_scaled = ss.transform(X_test)

#3) 분석모델 구축 및 훈련
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=20, max_iter=1000) 
model.fit(X_train_scaled, Y_train)
#4) 모델 평가 및 결과분석
model.score(X_train_scaled, Y_train) #0.55091
model.score(X_test_scaled, Y_test) #0.5394

model.classes_
#array([3, 4, 5, 6, 7, 8, 9], dtype=int64)
model.coef_
X.columns
#3등급 = 0.25*fixed acidity+0.99*volatile acidit+....-0.454*type_red+0.454*type_white
# [ 0.25182149,  0.99122202, -0.10282871,  0.20678959,  1.2757581 ,
#          0.87142528, -1.14286484,  0.15362122,  0.37447421, -0.540647  ,
#         -0.91492799, -0.45499205,  0.45499205]
#3등급 분류하는데 영향을 주는 변수 volatile acidity, alcohol
pd.DataFrame([model.coef_[0]], columns=X.columns).T.sort_values(0)

#                              0
# total sulfur dioxide -1.142865
# alcohol              -0.914928
# sulphates            -0.540647
# type_red             -0.454992
# citric acid          -0.102829
# density               0.153621
# residual sugar        0.206790
# fixed acidity         0.251821
# pH                    0.374474
# type_white            0.454992
# free sulfur dioxide   0.871425
# volatile acidity      0.991222
# chlorides             1.275758

#type :     red, blue, white
#레드와인    1, 0,  0
#블루와인    0, 1 , 0
#화이트와인  0, 0, 1

#type :     red, blue
#레드와인    1, 0, 
#블루와인    0, 1 
#화이트와인  0, 0

#더미칼럼 개수: 질적변수의 값의 유니크한 개수 -1 
##더미칼럼 개수: 질적변수의 값의 유니크한 개수(이렇게 분석해도 지장은 없다.)

#type :     red, white
#레드와인    1,  0
#화이트와인  0,  1

#type :     red
#레드와인    1
#화이트와인  0