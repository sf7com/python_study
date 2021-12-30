# 1_분류_체격을통한학생분류.py

#(0) 문제정의 : 키, 몸무게, 성별 => 초등학생, 중학생, 고등학생

#1.데이터 로드
from numpy import tracemalloc_domain
import pandas as pd
df = pd.read_csv('./딥러닝/data/학생건강검사 결과분석 rawdata_서울_2015.csv',
        encoding='euc-kr')
df.head()
df.info()

#2.데이터 전처리
#2-1) 피쳐/타겟 데이터 생성
df['성별'] #남/여
df['성별'] = df['성별'].apply(lambda x:1 if x=='남' else 0) #남자1,여자0
df['성별']
#타겟데이터 라벨링 : 초등학생 0, 중학생 1, 고등학생 2
df['학교명'] = df['학교명'].\
    apply(lambda x: 0 if str(x).endswith("초등학교") 
        else 1 if str(x).endswith("중학교") else 2)
df['학교명'].value_counts()

X = df[['키','몸무게','성별']]
Y = df['학교명']

#2-2) 훈련/테스트 데이터 분할
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.3,
                                random_state=32)

#3.모델생성
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# model = GradientBoostingClassifier()
# model.fit(X_train, Y_train)
# model.score(X_train, Y_train) #0.73321
# model.score(X_test, Y_test) #0.736316
# pd.DataFrame(model.feature_importances_, index=X_train.columns) #키

# #모델활용
# model.predict([[180,80,1]]) #array([2]) => 고등학생
# model.predict([[150,50,1]]) #array([0]) => 초등학생

#12시에 진행하겠습니다.


#pip install tensorflow 

# # 021-12-24 11:40:16.282136: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
# # 2021-12-24 11:40:16.282447: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do 
# # not have a GPU set up on your machine.
from tensorflow import keras
from keras.layers import Dense
model = keras.Sequential()
model.add(Dense(100, activation='relu', input_shape=(3,))) #입력층 및 은닉층1
model.add(Dense(50, activation='relu')) #은닉층 2
model.add(Dense(3, activation='softmax')) #출력층
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
hist = model.fit(X_train, Y_train, epochs=30, batch_size=8,
                validation_split=0.2)

len(X_train)