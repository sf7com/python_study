#의사결정 나무
#데이터 분류 기준을 여러개를 세워서 분류 정확 높임
#데이터 분류에 대한 명확한 기준과 이유를 제시하는 설명 가능한 알고리즘

#0) 문제정의 : 와인분류(레드와인, 화이트와인)
#분류 기준에 대한 명확한 이유까지 제시

#1) 데이터 로드
from numpy import dot
import pandas as pd
df = pd.read_csv('./데이터분석/data/wine.csv')
df.head()
df.info()
#class 0:레드와인, 1:화이트와인

#2) 데이터 탐색 및 전처리
#그룹별로 차이가 있는지 확인
df.groupby('class').describe().T
# class                  0.0          1.0
# alcohol count  1599.000000  4898.000000
#         mean     10.422983    10.514267
# sugar   count  1599.000000  4898.000000
#         mean      2.538806     6.391415
# pH      count  1599.000000  4898.000000
#         mean      3.311113     3.188267

#두 집단의 평균값 차이가 있는지 여부를 알아내는 통계적 기법?
#t검정
from scipy import stats
white_df = df[df['class']==1]
red_df = df[df['class']==0]
stats.ttest_ind(red_df['alcohol'], white_df['alcohol'],
        equal_var=False)
#Ttest_indResult(pvalue=0.004277779864993429)
#P값이 0.05미만이면 통계적으로 평균값 차이가 있다.
stats.ttest_ind(red_df['sugar'], white_df['sugar'],
        equal_var=False)
#Ttest_indResult(pvalue=0.0)
stats.ttest_ind(red_df['pH'], white_df['pH'],
        equal_var=False)
# Ttest_indResult(pvalue=2.3422645384856303e-149)
#대립가설 : 주장하고자 하는 가설
#귀무가설 : 대립가설의 반대
#-----------------------------------------------------
#훈련/테스트 분할
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    df.iloc[:, 0:-1], df.iloc[:, -1], 
    test_size=0.2, random_state=42, stratify=df.iloc[:, -1]
) #stratify 라벨들을 균등하게 훈련/테스트 데이터에 나눠준다.
#stratify=Target값 해주면 된다.
#target class0 20개, class1 5개 총 25개의 데이터 중에서
#랜덤하게 훈련/테스트 분할한다.
#타겟 분포가 불균형할때 해줘야하는 파라미터
df['class'].value_counts()
# 1.0    4898
# 0.0    1599

X_train.shape#(5197, 3)
X_test.shape#(1300, 3)
#데이터 표준화
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_train)
X_train_scaled = ss.transform(X_train)
X_test_scaled = ss.transform(X_test)

#3) 분석모델 - 로지스틱회귀모델
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_scaled, Y_train)
lr.score(X_train_scaled, Y_train) #0.7808
lr.score(X_test_scaled, Y_test) #0.7776
#회귀 계수값
lr.coef_, lr.intercept_
#(array([[ 0.51270274,  1.6733911 , -0.68767781]]), array([1.81777902]))
#z = 0.51*알콜표준화값+1.67*당도표준화값-0.69산도표준화값+1.8
#z값이 클수록 화이트와인(1), z값이 작을수록 레드와인 (0)
#당도의 회귀계값이 제일 크므로 분류에 있어 가장 큰 기여도
#PH(산도)값이 클수록 레드와인
#-------------------------------------------------------------
#의사결정 나무 모델화
#피쳐 데이터의 표준화를 안해도 되는 모델
#표준화를 안해야 설명이 명확해짐
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, Y_train)
dt.score(X_train, Y_train) #0.9969
dt.score(X_test, Y_test) #0.8584

#의사결정 나무 시각화
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10,7))
plot_tree(dt)
plt.show()


plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True, 
        feature_names=df.columns[:-1])
plt.show()

#-------------------------------------------------------------------
#의사결정나무 시각화 변수
#테스트 조건 (분류기준 ex) sugar<=0.432)
#불순도 (gini 혹은 엔트로피 계수)
#총샘플수 (samples)
#클래스별 샘플수(value)

#의사결정나무의 매개변수중 criterion(분할기준)의 기본값 gini
#지니 계수값이 작을 수록 한 클래스의 비율이 높음(정보의 순도가 높음)

#정보이득 : 부모와 자식 노드의 불순도 차이
#의사결정 나무는 부모노드와 자식노드의 불순도 차이가 가장 크도록 
#나무를 성장시킴
#정보이득이 최대가 되도록 데이터를 나눈다.

#트리의 깊이를 제한을 안하면 나무의 깊이아 매우 커져
#과대적합이 된다. 
dt = DecisionTreeClassifier(max_depth=3, random_state=32)
dt.fit(X_train, Y_train)
dt.score(X_train, Y_train) #0.8458
dt.score(X_test, Y_test) #0.8407

plt.figure(figsize=(10,7))
plot_tree(dt, filled=True, 
        feature_names=df.columns[:-1])
plt.show()
#레드와인 기준 
# 1.25 <= 당도 < 4.325
# 알콜 <= 11.025
#화이트와인은 나머지 경우

#특성 중요도 : 어떤 특성이 분류모델에서 기여도가 큰지를 계산
dt.feature_importances_
# array([0.14086835, 0.85360568, 0.00552597])
pd.DataFrame({'feature':df.columns[:-1], 
                'importance':dt.feature_importances_})
# 0  alcohol    0.140868
# 1    sugar    0.853606
# 2       pH    0.005526
#당도의 기여도가 85%, 알콜 14%,..
#데이터 탐색, 두집단의 당도의 평균값 차이가 가장 컸다.
#----------------------------------------------------------------
#결정트리의 깊이값 주면 대칭트리 형성
#min_impurity_decrease(최소불순도) : 어떤 노드의 불순도 값이 설정값 보다
#작으면 더 이상 분할하지 않음
#비대칭 트리 생성

dt = DecisionTreeClassifier(min_impurity_decrease=0.005)
dt.fit(X_train, Y_train)
dt.score(X_train, Y_train) #0.8458
dt.score(X_test, Y_test) #0.8407

plt.figure(figsize=(10,7))
plot_tree(dt, filled=True, 
        feature_names=df.columns[:-1])
plt.savefig('rf_5trees.png')
plt.show()
#--------------------------------------------------
#앙상블 학습(모델들을 결합해서 성능을 최적하는 기법)
#주로 의사결정 나무를 활용
#ex) 의사결정 모델 100개를 써서 앙상블 모델 -> 랜덤포레스트

import pydotplus
from sklearn.tree import export_graphviz
import os
from io import StringIO
path = './데이터분석/data'
dot_data = StringIO()
export_graphviz(dt, out_file='dt.dot',
                                  feature_names = X_train.columns,
                                  class_names = ['red','white'],
                                  filled = True, rounded = True,
                                  special_characters = True)
dot_data

dt_graph = pydotplus.graph_from_dot_data(dot_data)
dt_graph.create_png('aa.png')









graph = pydotplus.graph_from_dot_file('./tree.dot')
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.render(filename='tree.png')

import pydot
(graph, ) = pydot.graph_from_dot_file('./tree.dot', encoding='utf-8')
graph.write_png('de.png')