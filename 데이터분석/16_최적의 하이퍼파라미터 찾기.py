#하이퍼 파라미터 : 머신러닝 모델의 세팅값(데이터 분석가가 모델에 설정해줘야 하는 값)
#                 ex) epoch(훈련횟수), C(규제값), max_depth(의사결정나무)

#최적의 하이퍼파라미터 찾을 때 테스트 데이터의 점수로 맞춰서 하이퍼파라미터
#조정시 문제점 ? 
#모델이 테스트 데이터 맞춰서 훈련이 된다.
#테스트 데이터는 모델의 성능 측정이 목적이다. 그래서 올바른 성능 측정을 위해서
#테스트 데이터는 모델을 만들때 사용하면 안된다.

#데이터를 다음처럼 나눈다.
#ex) 훈련세트 : 60%, 검증세트 : 20%, 테스트 세트 : 20%
#(1) 훈련세트에서 모델 훈련
#(2) 검증세트로 모델 평가(하이퍼 파라미터 바꾸며 최적화)
#(3) 최적의 하이퍼파라미터로 훈련세트와 검증세트 합쳐 다시 모델 훈련
#(4) 테스트 세트를 통해 최종 점수 평가

import pandas as pd
from scipy.sparse.construct import random
df = pd.read_csv('./데이터분석/data/wine.csv')
df.info()

#훈련/테스트 데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:,:-1],df.iloc[:,-1],random_state=42,test_size=0.2)

#훈련/ 검증 데이터 분할
X_sub, X_val, Y_sub, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=13)

X_train.shape #(5197, 3) #전체 훈련데이터
X_sub.shape #(4157, 3) #임시훈련데이터
X_val.shape #(1040, 3) #검증데이터

#모델 훈련
from sklearn.tree import DecisionTreeClassifier
depthList = [i for i in range(3,100)]
trainScores = []
valScores = []
for depth in depthList :
    model = DecisionTreeClassifier(random_state=32, max_depth=depth)
    model.fit(X_sub,Y_sub)
    trainScores.append(model.score(X_sub,Y_sub))
    valScores.append(model.score(X_val, Y_val))

import matplotlib.pyplot as plt
import numpy as np
plt.plot(np.arange(3,100), trainScores, label='train')
plt.plot(np.arange(3,100), valScores, label='test')
plt.xlabel("depth")
plt.ylabel("score")
plt.legend()
plt.grid(True)
plt.show()

#최적의 depth 30
model = DecisionTreeClassifier(random_state=32, max_depth=30)
model.fit(X_train, Y_train)
model.score(X_train, Y_train) #0.9969
model.score(X_test,Y_test) #0.8623
#--------------------------------------------------------------------
#교차 검증 : 훈련 데이터에서 검증세트를 떼어 내어 평가하는 과정을 여러 번 반복
#점수 평균을 활용해서 최종 점수를 얻음

#k-폴드 교차검증 : 훈련세트를 k부분으로 나눠서 교차검증 수행
#ex) 3-폴드 교차검증
#훈련데이터를 3세트로 나눔 : 훈련세트1(33%), 훈련세트2(33%), 훈련세트3(33%)
#(1) 훈련세트1 + 훈련세트2 합쳐 모델 훈련 -> 훈련세트3 모델평가 (평가점수 1)
#(1) 훈련세트1 + 훈련세트3 합쳐 모델 훈련 -> 훈련세트2 모델평가 (평가점수 2)
#(1) 훈련세트2 + 훈련세트3 합쳐 모델 훈련 -> 훈련세트1 모델평가 (평가점수 3)
#검증 점수 평균 = mean(평가점수1,평가점수2,평가점수3)

#보통 5-폴드 교차검증 또는 10-폴드 교차검증 많이 사용함
#훈련데이터의 80~90%까지 훈련데이터로 활용가능
#검증데이터 개수는 줄어들지만 검증 점수의 평균값을 통해서 최종 검증 점수를 얻을 수
#있어서 안정된 점수로 파악이 가능

from sklearn.model_selection import cross_validate
#기본적으로 5-폴드 교차검증 수행, cv 파라미터로 바꿀 수 있다.
scores = cross_validate(model, X_train, Y_train)
scores
# {'fit_time': array([0.00698113, 0.00600791, 0.00595999, 0.00598431, 0.00598383]), 
# 'score_time': array([0.0009973 , 0.00099778, 0.00199485, 0.00099707, 0.00102472]), 
# 'test_score': array([0.86634615, 0.85288462, 0.87487969, 0.84985563, 0.84023099])}
#fit_time : 훈련시간
#score_time : 성능평가 시간
#test_score : 각 검증별 점수
import numpy as np
#검증점수의 평균값
np.mean(scores['test_score']) #0.8568

#훈련세트를 섞으면서 다른 K-폴드 교차검증 진행하기 위해서 분할기 설정
from sklearn.model_selection import StratifiedKFold
# StratifiedKFold : 데이터를 랜덤하게 나눠주면서 타겟들의 분포도 균등하게 나눠줌
#10-폴드 교차검증, 훈련데이터 섞는다.
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=32)
scores = cross_validate(model, X_train, Y_train, cv=splitter)
np.mean(scores['test_score']) #0.8568 => 0.8624
#교차검증을 수행하면 입력한 모델에서 얻을 수 있는 테스트 셋의 성능 지표를 가늠해 볼 수 있다.
#-----------------------------------------------------------------------
#하이퍼파라미터 튜닝
#모델마다 적게 1~2, 많게는 5개 이상의 하이퍼파라미터가 있다.
#(1) 탐색할 하이퍼파라미터 지정
#(2) 그리드 서치를 수행하여 최상의 평균 검증 점수가 나오는 매개변수 조합 찾기
#(3) 그리드 서치는 최상의 매개변수에서 전체 훈련세트를 사용해 최종 모델을 훈련
#    (자동으로 해줌, 그리드 서치 객체에 저장되어있다.)

#의사결정나무 하이퍼파라미터
#max_depth = [3~20] (18가지 경우)
#min_impurity_decrease = [0.0001, 0.0002,0.0003,0.0004,0.0005]
#(1) max_depth 먼저 최적의 검증점수가 나오는 값을 구하고 나서
# min_impurity_decrease의 최적의 검증점수가 나오는 값을 구하면?
# max_depth의 최적값은 min_impurity_decrease의 값에 따라서 달라질수가 있다.
#두 매개변수를 동시에 바꿔가면서 최적의 값을 구해야 한다.
#즉 모든 경우의 수에 대해서 각각 검증 점수를 얻어야 한다.
#총 90번의 검사를 해야한다. 18*5=90

#********************GridSearchCV 가 많이 사용 됨 ***********************

from sklearn.model_selection import GridSearchCV #(CV : 교차검정의 약자)
params = {'min_impurity_decrease' : [0.0001, 0.0002,0.0003,0.0004,0.0005],
            'max_depth' : range(5, 20, 1)}
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=32)
gs = GridSearchCV(DecisionTreeClassifier(random_state=32), params, n_jobs=-1, cv=splitter) #n_jobs -1은 cpu 모든 코어 사용
#75개의 모든 경우의 수에 대해서 모델을 만드는데, 5폴드 교차검증
#15번*5번의 모델을 만들어서 평가함
gs.fit(X_train, Y_train)
gs.best_params_ #최적의 하이퍼파라미터 값
# {'max_depth': 17, 'min_impurity_decrease': 0.0001}
gs.cv_results_['mean_test_score'].shape #(75,)
np.max(gs.cv_results_['mean_test_score']) #0.8726

#최상의 모델얻기
best_model = gs.best_estimator_
best_model.score(X_train, Y_train) #0.9611
best_model.score(X_test, Y_test) #0.87
#---------------------------------------------------------------
#GridSearch 단점 : 매개변수의 값의 범위를 설정 (경험)

#최적의 매개변수 찾기 - 랜덤값으로 찾기
#랜덤서치 사용
#랜덤서치 : 매개변수 값의 목록을 전달하는 것이 아니라 매개변수를 샘플링 할 수 있는
#          확률분포 객체를 전달
#균등분포 샘플링 : uniform(실수),randint(정수) 주어진 범위에서 고르게 값을 뽑는다.
from scipy.stats import uniform, randint
rInt = randint(0,10) #0~9까지 랜덤한 정수 확률분포
rInt.rvs(10) #10개의 샘플 추출
# array([1, 0, 7, 2, 3, 2, 4, 7, 5, 2])

#1000개 추출 후 각 갯수 파악
np.unique(rInt.rvs(1000),return_counts=True)
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 
# array([109,  99,  98,  95, 108, 107,  93,  86, 105, 100], dtype=int64))

rFloat = uniform(0,1) #0과 1사이의 랜덤한 실수 확률분포
rFloat.rvs(10)
# array([0.37133495, 0.94293961, 0.33839337, 0.29359713, 0.76304787,
#        0.92593068, 0.22409884, 0.20687327, 0.41211136, 0.98860591])
params = {'min_impurity_decrease' : uniform(0.0001,0.01),
            'max_depth' : randint(5,100)}
from sklearn.model_selection import RandomizedSearchCV
#n_iter 만큼 샘플링하여 교차 검증 수행하고 최적의 매개변수 조합 찾는다
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=32), params,
        n_iter=100, n_jobs=-1, random_state=32, cv=splitter)
gs.fit(X_train, Y_train)
gs.best_params_
# {'max_depth': 84, 'min_impurity_decrease': 0.0001692848044486716}
np.max(gs.cv_results_['mean_test_score']) #0.87031
best_model = gs.best_estimator_
best_model.score(X_train, Y_train) #0.9340
best_model.score(X_test, Y_test) #0.8676
#-------------------------------------------------------------------------
