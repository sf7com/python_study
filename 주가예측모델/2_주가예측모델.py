
#1) 데이터 로드
import pandas as pd
df = pd.read_csv('./주가예측모델/data/삼성전자 2021-12-22~2009-10-29.csv')
df.info()
df.head()
#2) 데이터 탐색 및 전처리
#2-1) 분석용 데이터 구축
cols = df.columns[1:]
cols #Index(['종가', '등락율', '시작가', '최고가', '최저가', '거래량'], dtype='object')
for colName in cols :
    for i in range(1,5) :
        df[colName+f"_{i}"] = df[colName].iloc[i:].reset_index()[colName]

df.head(20)
df.info()
df.to_csv('data.csv')
#2-2) 결측치 제거 
df.dropna(inplace=True)
df.info()

#2-3) 피쳐/타겟 데이터 생성
df.head()
df['등락율']
Y = df['등락율'].iloc[:len(df)-1]
#Y2 = df['종가'].iloc[:len(df)-1]

#다음날 오를건지/내릴지 예측 모델(분류 모델)
#분류모델 : 로지스틱회귀분석, 랜덤포레스트분류
#타겟 데이터 생성
#등락률>0 라벨 1, 등락률<=0 라벨 0
Y2 = Y.apply(lambda x : 1 if x>=1 else 0) #x 는 등락율
Y2
Y2.value_counts()
# 0    1503
# 1    1492
#
# 0    2672 2%이상 안 오른날
# 1     323 2%이상 오른날


X = df[df.columns.difference(['날짜'])].iloc[1:].reset_index(drop=True)
X.head()
X.info()
len(X), len(Y) #(2995, 2995)

#2-4) 훈련/테스트 데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y2, 
            test_size=0.3, random_state=32)

#2-5) 데이터 정규화
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_train)
X_train_scaled = ss.transform(X_train)
X_test_scaled = ss.transform(X_test)

#3) 분석모델구축 및 훈련
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_scaled, Y_train)
model.score(X_train_scaled, Y_train) #0.569 => 0.888835 (2%오른날 기준)
model.score(X_test_scaled, Y_test) #0.5155 => 0.899888

from sklearn.metrics import precision_score, recall_score, f1_score
y_pred = model.predict(X_test_scaled)
y_pred
y_pred[y_pred==1]
Y_test[Y_test==1]
precision_score(Y_test, y_pred) #0.0 => 0.25 (1%오른날 기준)
recall_score(Y_test, y_pred) #0.0 => 0.0043
f1_score(Y_test, y_pred) #0.0 => 0.0086
#2% 오른날 기준으로 예측이 불가 함.


# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# model.fit(X_train_scaled, Y_train)
# model.score(X_train_scaled, Y_train) #0.9939510
# model.score(X_test_scaled, Y_test) #0.5216

import pickle
with open('./주가예측모델/data/삼전_주가모델.pickle', 'wb') as f :
    pickle.dump(model, f)
with open('./주가예측모델/data/삼전_정규화객체.pickle', 'wb') as f :
    pickle.dump(ss, f)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(Y_test, model.predict(X_test_scaled))
# 977.677
model.predict(X_test_scaled)[:5]
Y_test[:5]
#4) 모델 평가 및 결과분석
#특성 중요도 파악
pd.DataFrame(model.coef_, index=X_train.columns).sort_values(by=0, 
                ascending=False)
#다음날 주가 예측
#                   0   
# 종가     13492.450474  (예측날 1일전 종가)
# 종가_2    6413.645158  (예측날 3일전 종가)
# 최저가     4428.602190 (예측날 1일전 최저가 )
# 종가_4    3377.874058 
# 시작가_3   2618.368139
# 최고가_4   2037.291362
# 최고가_1   2005.426208
# 최저가_4   1586.381984
# 시작가_1   1461.561413
# 최고가_2  -2552.887965
# 최저가_1  -2676.166466
# 최저가_3  -2686.088859
# 시작가    -3533.924998
# 시작가_4  -3559.267231
# 종가_3   -5588.183818 (예측 4일전 종가)

from sklearn.ensemble import GradientBoostingRegressor
model2 = GradientBoostingRegressor(random_state=32, n_estimators=300)
model2.fit(X_train, Y_train)
model2.score(X_train, Y_train) #0.9989
model2.score(X_test, Y_test) #0.9927

pd.DataFrame(model2.feature_importances_, 
            index=X_train.columns).sort_values(by=0, ascending=False)
# 종가     0.707752
# 최고가    0.103892
# 최저가    0.049938

#5) 모델활용
#빙그레 다음날 12/23 주가 등락률 예측
X_train.info()
#API를 통해서 전 5일치 데이터를 가져와서 예측

