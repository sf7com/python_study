#17_최적의 알고리즘_앙상블학습.py
#---------------------------------------------------------
#지금까지 배운 머신러닝 알고리즘 정형 데이터 학습에 잘 맞다.
#정형 데이터 : 데이터베이스에 저장된 데이터, 잘 정돈된 데이터, 바로 사용가능한 데이터
#반정형 데이터 : 데이터의 스키마와, 데이터가 같이 있는 구조 ex)html, xml
#비정형 데이터 : 사진, 동영상, 소리파일 등

#비정형 데이터 학습은 딥러닝 알고리즘 사용

#정형 데이터에서 뛰어난 성적을 내는 알고리즘 앙상블 학습
#앙상블학습 : 여러 모델의 결합으로 최종 예측, 분류를 한다.
#대부분 의사결정나무의 모델들 활용

#대표적인 앙상블학습
#랜덤 포레스트
#의사결정나무를 랜덤하게 훈련시켜 각 트리마다 다르게 예측하고, 각 트리들의
#예측을 활용해 최종 예측을 만든다.

#각 트리마다 다르게 학습하는 방법
#부트스트랩 샘플 방식 활용 : 입력한 훈련데이터에서 랜덤하게 샘플 데이터 추출(복원 추출)
#하여 각 트리를 훈련시킨다.

#랜덤 포레스트는 기본적으로 100개의 의사결정나무 모델을 활용함 이 수치는 바꿀 수 있다.
#분류모델 : 각 트리의 클래스별 확률을 평균하여 가장 높은 확률을 가진 클래스를 예측
#회귀모델 : 각 트리의 예측값을 평균

#100개의 데이터에서 부트스트랩 샘플링 60% 추출
#60개 -> 1개의 훈련시킴
#100개 중에 60개를 뽑아서 다른 훈련시킴
#1트리, 2트리 => 훈련 데이터 달라짐
#남는 40개의 데이터 성능 평가
#---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.sparse.construct import rand
from sklearn.model_selection import train_test_split

df = pd.read_csv('./데이터분석/data/wine.csv')
X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], random_state=32, test_size=0.2)

#성능 테스트 - 교차검증 수행
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_jobs=-1, random_state=32)
scores = cross_validate(model, X_train, Y_train, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']),np.mean(scores['test_score']))
# 0.997931535939715 0.888012

#피쳐 중요도
model.fit(X_train, Y_train)
model.feature_importances_
pd.DataFrame(model.feature_importances_, index=X_train.columns)
#                 0
# alcohol  0.231837
# sugar    0.499715
# pH       0.268448

#OOB Score (out of bag score) : 부트스트랩 샘플에서 하나의 의사결정나무 훈련시 남는 샘플로 성능 평가
#교차 검증을 대신할 수 있어 결과적으로 훈련세트에 더 많은 샘플 사용 가능
model = RandomForestClassifier(n_jobs=-1, random_state=32, oob_score=True)
model.fit(X_train, Y_train)
model.oob_score_ #0.893207
model.score(X_test, Y_test) #0.895384
#-----------------------------------------------------------------------------
#그래디언스 부스팅 - 깊이가 얕은 의사결정나무를 사용해서 이전 트리의 오차를
#보완하는 방식, 기본적으로 깊이가 3인 의사결정나무 100개를 사용
#특징 : 과대적합에 강하고 일반적으로 높은 성능을 기대할 수 있다.
#일반적으로 랜덤 포레스트 학습보다 조금 더 높은 성능을 얻을 수 있다.
#단점 : 훈련속도가 느리다. 트리를 순차적으로 학습한다.
#n_jobs 매개변수가 없음

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(random_state=42)
scores = cross_validate(model, X_train, Y_train, return_train_score=True)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
#0.8845969228697157 0.8699239
#과대적합이 되지 않음

#하이퍼파라미터
#n_estimators : 트리개수, learning_rate : 학습률 기본값 0.1
model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2,
            random_state=32)
scores = cross_validate(model, X_train, Y_train, return_train_score=True)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
#0.9412643335424187 0.871655067742652

model.fit(X_train, Y_train)
model.score(X_train, Y_train) #0.93553
model.score(X_test, Y_test) #0.87692

model.feature_importances_
pd.DataFrame(model.feature_importances_, index=X_train.columns)
#                 0
# alcohol  0.172475
# sugar    0.665125
# pH       0.162400
#랜덤포레스트보다 일부 특성(당도)에 더 집중
#------------------------------------------------------------------
#히스토그램 기반 그레디언트 부스팅
#정형데이터를 다루는 머신러닝 알고리즘 중 가장 인기가 높은 알고리즘
#그레디언트 부스팅의 속도와 성능을 더욱 개선한 알고리즘

#입력한 특성을 256개의 구간으로 나눠 노드 분할시 최적의 분할을 빠르게 찾는다.
#입력한 데이터에 누락된 특성값이 있더라도 따로 전처리 할 필요 없다.

from sklearn.ensemble import HistGradientBoostingClassifier
model = HistGradientBoostingClassifier(random_state=32)
scores = cross_validate(model, X_train, Y_train, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
#0.9412643335424187 0.871655067742652

#트리 개수 지정시 max_iter 매개변수 사용
model = HistGradientBoostingClassifier(random_state=32, max_iter=300)
model.fit(X_train, Y_train)
model.score(X_train, Y_train) #0.93553 =>0.96825
model.score(X_test, Y_test) #0.87692 =>0.88538

#특성 중요도 계산
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_train, Y_train, n_repeats=10, random_state=32, n_jobs=-1)
result #중요도 평균, 중요도 표준편차, 중요도
result['importances_mean']
# array([0.13888782, 0.27791033, 0.13109486])
pd.DataFrame(result['importances_mean'], index=X_train.columns)
#----------------------------------------------------------------
#앙상블학습 위의 알고리즘 말고도 다양, 지금도 연구원 좋은 앙상블 학습 모델 개선중
#(1) 랜덤포레스트 : 가장 대표적인 앙상블 학습 알고리즘, 첫번째로 시도하기 좋음, 성능이 좋고 , 안정적
#(2) 그레디언트부스팅 : 깊이가 얕은 트리를 연속적으로 추가하여 손실함수를 최소화
#                      성능이 뛰어나지만 병렬로 훈련할 수 없어 훈련속도가 느림
#(3) 히스토그램 기반 그레디언트 부스팅 : 가장 뛰어난 앙상블 학습으로 평가받고 있음.
#훈련 데이터를 256개의 구간으로 변환하여 노드 분할소도가 빠르다.
#----------------------------------------------------------------
#분류 : RandomForestClassifier
#회귀 : RandomForestRegressor
#분류 : GradientBoostingClassifier
#회귀 : GradientBoostingRegressor
#분류 : HistGradientBoostingClassifier(분류)
#회귀 : HistGradientBoostingRegressor
#-----------------------------------------------------------------------------------
#12/22 복습
#1.비지도학습 : 데이터의 정답이 없는 상태로 학습
# 1-1)군집(K-means) : 비슷한 데이터끼리 클러스터(군집) 형성
#   ex)과일 사진 분류, 소비자 유형 분석
# 1-2) 차원축소(PCA) : 데이터의 차원(특성) 줄여준다.
#      차원은 특성의 개수이다.
#      주성분은 데이터의 정보량이 많은(분산이 큰)방향의 벡터

#2.최적의 하이퍼파라미터 찾기
#  교차검증을 통해서 교차검증의 점수가 높은 하이퍼파라미터 찾기(K-폴드 교차검증)
#  (훈련세트에서 검증세트를 뽑아내서 테스트점수를 가늠할 수 있음)
# 2-1) GridSearch : 파라미터 값의 나열을 통해서 최적의 성능을 내는 파라미터를 찾는다.
# 2-2) RandomizedSearch : 파라미터 값을 확률분포에서 샘플링을 통해서 최적의 파라미터를 찾는방법

#3. 최적의 알고리즘 앙상블 학습 : 여러개의 모델을 결합해서 최종 예측 및 분류하는 모델
# 3-1) 랜덤포레스트 : 의사결정나무 여러개를 부트스트랩 데이터 샘플링을 통해서 학습
# 3-2) 그레디언트 부스팅 : 의사결정나무의 깊이값을 제한하고 순차적 전 나무의 오차에 따라 학습시킨다.
# 3-3) 히스토그램 그레디언트 부스팅 : 가장 뛰어난 앙상블 학습으로 평가받고 성능과 학습속도가 좋다.

