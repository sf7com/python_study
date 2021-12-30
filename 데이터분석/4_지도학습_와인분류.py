import pandas as pd
#1) 데이터 수집
df = pd.read_csv('./데이터분석/data/winequality-red.csv',sep=';')
df.head()
df.info()

#2) 데이터 탐색 및 전처리
#2-1) 기술통계
#와인품질 예측
#와인품질 분포
df.quality.unique() #array([5, 6, 7, 4, 8, 3], dtype=int64)
sorted(df.quality.unique())
df.groupby('quality').describe()['fixed acidity']

#df.info()
#df.iloc[인덱스 번호, 열번호]
#        행,열
# df.iloc[0:10, 0:5]
# df.loc[0:10, ['fixed acidity', 'volatile acidity']]
df.info()
#훈련/테스트 분할
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = \
    train_test_split(df.iloc[:, :-1], df.iloc[:, -1],
            test_size=0.3, random_state=13)
#훈련/테스트 데이터 비율 일반적으로 7:3, 8:2

#표준화
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_train)
X_train_scaled = ss.transform(X_train)
X_test_scaled = ss.transform(X_test)

#3) 분석모델 구축 및 훈련
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train_scaled, Y_train)

#4) 모델평가 및 결과분석
model.score(X_train_scaled, Y_train) #0.698
model.score(X_test_scaled, Y_test) #0.5520
#과대적합, 과소적합? 과대적합
#과대적합 : 훈련데이터의 점수는 높은데 테스트 데이터가 낮은경우
#모델 복잡

#과소적합 : 훈련데이터의 점수보다 테스트 데이터의 점수가 높은 경우
#모델 단순

#오차값
from sklearn.metrics import mean_absolute_error
pred = model.predict(X_test_scaled)
mean_absolute_error(Y_test, pred) #0.4875
#3,4,5,6,7,8

#5) 모델활용
#새로운 와인 데이터의 등급 예측
data = {'fixed acidity':[8.5,8.1], 
       "volatile acidity":[0.8, 0.5], 
       "citric acid" :[0.3,0.4],
       "residual sugar":[6.1,5.8],
       "chlorides" : [0.055,0.04],
       "free sulfur dioxide":[30.0,31.0],
       "total sulfur dioxide":[98.0,99],
       "density":[0.996,0.91],
       "pH":[3.25,3.01],
       "sulphates":[0.4,0.35],
       "alcohol":[9.0,0.88]}
data_df = pd.DataFrame(data)
data_df.head()

#표준화
data_scaled = ss.transform(data_df)
data_scaled
pred = model.predict(data_scaled)
pred #array([5, 6], dtype=int64)