#18_지도학습_앙상블_배추가격
#------------------------------------------------------------
#머신러닝 프로세스
#(0) 분석 목표 확립(어떤 데이터로 무엇을 예측/분류?)
#(1) 데이터 수집 및 로드
#(2) 데이터 탐색 및 전처리
#   (2-1) 데이터 탐색 - 데이터 시각화, 데이터 구조 파악, 데이터 통계
#                      데이터 탐색의 목적 : 데이터 인사이트(정보) 얻기
#   (2-2) 결측치 제거 - null인 데이터를 삭제) 또는 결측치 데이터를
#                      다른 데이터로 채우기(평균값, 최빈값)
#   (2-3) 중복 데이터 제거
#   (2-4) 텍스트 데이터 전처리 : 정제, 정규화, 토큰화, 불용어제거,표제어추출
#   (2-5) 훈련/테스트 데이터 분할
#   (2-6) 데이터 정규화
#(3) 분석모델구축 및 훈련
#   (3-1) 모델 선정 및 하이퍼파라미터 선정
#(4) 모델평가 및 결과분석
#   (4-1) 모델 성능 측정을 통해 모델 최적화(모델 또는 파라미터 변경)
#         교차검증, 그리드서치, 랜덤서치
#(5) 모델활용 (새로운 데이터 분류/예측)
#-------------------------------------------------------------
#(0) 배추가격 예측 모델
#(1) 데이터 수집 및 로드
import pandas as pd
from scipy.sparse.construct import random
df = pd.read_csv('./데이터분석/data/price data.csv')
df.head()
df.info()
#(2) 데이터 탐색 및 전처리
#2-1) 데이터 전처리 - 데이터 분석에 활용할 수 있는 데이터로 변경
#Year를 분리 -> 년도,월,일
#year가 int이므로 문자열 타입으로 바꾸기
df['year'] = df['year'].astype(str)
df.info()
df['Year'] = df['year'].str[0:4].astype(int)
df['Mon'] = df['year'].str[4:6].astype(int)
df['Day'] = df['year'].str[6:8].astype(int)
df.head()
#int -> 범주형 변수로 바꿀예정
#2-2) 데이터 탐색 - year,mon 기준으로 평균온도와 가격 추세 시각화
yearMonGroup = df.groupby(['Year','Mon']).mean()
yearMonGroup
import matplotlib.pyplot as plt
#한 창에 x축 공유, y축 다르게 두개의 그래프 시각화
#x축 년월데이터, y축(평균기온, 평균가격)
fig,ax1 = plt.subplots()
ax1.set_title("Avg Temp And Price By Year-Mon")
xlabels = [f'{str(y)}-{str(m):>02}' for y,m in yearMonGroup.index]
xlabels
ax1.plot(xlabels, yearMonGroup['rainFall'].values, c='r', label='rainFall')
#한창에 x축 공유 및 y축 다르게
ax2 = ax1.twinx()
ax2.plot(xlabels, yearMonGroup['avgPrice'].values, c='g',label='AvgPrice')
ax1.grid(True)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax1.set_xticklabels(xlabels, rotation=75)
plt.show()
#------------------------------------------------------------------------
#2-3) 달 데이터 범주형 변수로 바꾸고 더미변수화
df.columns.difference(['year','Day','avgPrice'])
# Index(['Mon', 'Year', 'avgTemp', 'maxTemp', 'minTemp', 'rainFall'], dtype='object')
X=df[df.columns.difference(['year','Day','avgPrice'])]
X.info()
X = X[['Year','Mon']+list(X.columns[2:])]
X.head()
Y=df['avgPrice']

#Mon 더미 변수화
X['Mon'] = X['Mon'].astype('category')
X = pd.get_dummies(X, ['Mon'])
X.head()
X.info()
#      Mon_1  Mon_2  Mon_3  Mon_4  Mon_5  Mon_6  Mon_7  Mon_8  Mon_9  Mon_10  Mon_11  Mon_12
#1월달   1      0      0      0      0 ...
#2월달   0      1      0      0      0 ...
#3월달   0      0      1      0      0 ...

#2-4) 훈련/테스트 분할
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=32)

#3) 분석 모델 구축 및 훈련
from sklearn.ensemble import RandomForestRegressor  #예측
model = RandomForestRegressor(oob_score=True)
model.fit(X_train, Y_train)
model.oob_score_ #0.89084

#4) 모델평가 및 결과분석
#최적화
from sklearn.model_selection import GridSearchCV
params = {
    'n_estimators' : range(200,300,20), #300
    'max_depth' : range(5,10,2) #27
}
# 300*27 = 8100개 모델 형성
from sklearn.model_selection import StratifiedKFold
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=32)
gs = GridSearchCV(model, params, n_jobs=-1, cv=splitter)
gs.fit(X_train, Y_train)
gs.best_params_ #{'max_depth': 9, 'n_estimators': 200}
import numpy as np
np.max(gs.cv_results_['mean_test_score']) #0.81215

#최상의 모델얻기
X.info()
# 0   Year      2922 non-null   int32
#  1   avgTemp   2922 non-null   float64
#  2   maxTemp   2922 non-null   float64
#  3   minTemp   2922 non-null   float64
#  4   rainFall  2922 non-null   float64
#  5   Mon_1     2922 non-null   uint8
#  6   Mon_2     2922 non-null   uint8
#  7   Mon_3     2922 non-null   uint8
#  8   Mon_4     2922 non-null   uint8
#  9   Mon_5     2922 non-null   uint8
#  10  Mon_6     2922 non-null   uint8
#  11  Mon_7     2922 non-null   uint8
#  12  Mon_8     2922 non-null   uint8
#  13  Mon_9     2922 non-null   uint8
#  14  Mon_10    2922 non-null   uint8
#  15  Mon_11    2922 non-null   uint8
#  16  Mon_12    2922 non-null   uint8
# dtypes: float64(4), int32(1), uint8(12)
#unit 8bit => 0~255 양수만 표현
#int 8bit => -128~127

best_model = gs.best_estimator_
best_model.score(X_train,Y_train) #0.8839
best_model.score(X_test,Y_test) #0.8095

#피쳐 중요도 확인
pd.DataFrame(best_model.feature_importances_, index=X_train.columns).sort_values(by=0,ascending=False)
#                 0
# Year      0.387570
# Mon_9     0.165345
# Mon_10    0.079529
# avgTemp   0.077874
# minTemp   0.058804
# Mon_8     0.052678
# Mon_4     0.044585
# maxTemp   0.029924
# Mon_6     0.022902
# Mon_3     0.021883
# Mon_1     0.013843
# rainFall  0.010667
# Mon_11    0.009802
# Mon_12    0.008824
# Mon_5     0.007433
# Mon_2     0.006045
# Mon_7     0.002292

#5) 모델활용 (가격예측)
monList = [0]*12
monList[11] = 1
pred_x = [2021, 3.9,7.0,-3.0,0] + monList
pred_x
# [2021, 3.9, 7.0, -3.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
best_model.predict([pred_x]) #array([3285.02129911])