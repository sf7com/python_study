#감성분석 : 텍스트에서 사용자의 긍정 또는 부정의 감정을 결정
#과거 감성분석 모델 : 감성 사전 기반으로 감정 단어를 검색해서
#점수 계산 ex) 좋다+1, 나쁘다-1, 기쁘다+1

#최근에는 감성분석을 머신러닝 기반으로 지도학습을 통해 모델구현

#감성분석 목적 : 상품 댓글들의 긍정 혹은 부정 비율을 통해
#               경영 의사결정 보조를 위한 수치화
#---------------------------------------------------------
#1) 데이터 로드
import pandas as pd
train_df = pd.read_csv('./텍스트분석/data/ratings_train.txt',
        encoding='utf-8', sep='\t')
train_df.head()
train_df.info()

#2) 데이터 탐색 및 전처리
#2-1) 결측치 제거
train_df = train_df[train_df['document'].notnull()]
train_df.info()
#2-2) 탐색 : 라벨 분포 확인
train_df['label'].value_counts()
# 0    75170
# 1    74825
#분류 모델에 있어서 Target의 분포가 비슷해야 좋다.
#라벨의 비율이 다른 경우 
#1) 작은 라벨의 데이터를 부풀리는 방법 (큰 라벨의 데이터 개수와 맞춰준다)
#2) 큰 라벨의 데이터를 작은 데이터의 개수로 맞춰주는 방법 (데이터를 버리는 작업)

#2-3) 텍스트 전처리 
#정제 
import re
train_df['document'] = train_df['document'].str.replace(r'[^ㄱ-ㅎ|가-힣|ㅏ-ㅣ]', " ")
train_df['document'].head()

#------------------------------------------------------------------
test_df = pd.read_csv('./텍스트분석/data/ratings_test.txt',
        encoding='utf-8', sep='\t')
test_df.head()
test_df.info()

#2) 데이터 탐색 및 전처리
#2-1) 결측치 제거
test_df = test_df[test_df['document'].notnull()]
test_df.info()
#2-2) 탐색 : 라벨 분포 확인
test_df['label'].value_counts()

#2-3) 텍스트 전처리 
#정제 
import re
test_df['document'] = test_df['document'].str.replace(r'[^ㄱ-ㅎ|가-힣|ㅏ-ㅣ]', " ")
test_df['document'].head()

#2-4) 데이터 개수 줄임(벡터화 시간 오래 걸려서)
len(train_df) #149995
len(test_df) #49997
train_df  = train_df.head(50000)
test_df  = test_df.head(15000)

#2-5) TF-IDF 기반 벡터화
from konlpy.tag import Okt
okt = Okt()
def okt_tokenizer(text) :
    tokens = okt.morphs(text, stem=True)
    return tokens
from sklearn.feature_extraction.text import TfidfVectorizer
tfVec = TfidfVectorizer(tokenizer=okt_tokenizer, ngram_range=(1,2), min_df=3, max_df=0.9)
tfVec.fit(train_df['document'])
train_vec = tfVec.transform(train_df['document'])
test_vec = tfVec.transform(test_df['document'])

#tfidf 모델 벡터화 단어들
tfVec.vocabulary_.items()
train_vec.toarray()[0] #array([0., 0., 0., ..., 0., 0., 0.])

#3) 분석모델 구축 및 훈련
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(train_vec, train_df['label'])
model.score(train_vec, train_df['label']) #0.90074
model.score(test_vec, test_df['label']) #0.8447333
#과대 적합
#규제

#4) 모델 평가 및 결과 분석
#최적의 C값(규제값) 찾기
cList = [0.001, 0.01, 0.1, 1, 10, 100] #값이 작을 수록 규제가 커진다
train_score = []
test_score = []
for c in cList :
    model = LogisticRegression(C=c)
    model.fit(train_vec, train_df['label'])
    train_score.append(model.score(train_vec, train_df['label'])) 
    test_score.append(model.score(test_vec, test_df['label']))  
train_score
test_score

#시각화
import matplotlib.pyplot as plt
import numpy as np
plt.plot(np.log10(cList), train_score, label='train')
plt.plot(np.log10(cList), test_score, label='test')
plt.xticks(np.log10(cList), labels=cList)
plt.xlabel("C")
plt.ylabel("score")
plt.grid(True)
plt.legend()
plt.show()

#최적의 C값은 1
model = LogisticRegression(C=1)
model.fit(train_vec, train_df['label'])
model.score(train_vec, train_df['label']) #0.90074
model.score(test_vec, test_df['label']) #0.84473

#5) 모델 활용
text = "ㅋㅋㅋㅋㅋㅋ 웃기다"
#텍스트 전처리
text = re.sub(r'[^ㄱ-ㅎ|가-힣|ㅏ-ㅣ]', " ", text)
#텍스트 벡터화
text_vec = tfVec.transform([text])
#예측
pred = model.predict(text_vec)
model.predict_proba(text_vec) #array([[0.88707606, 0.11292394]])
if(pred==0) :
    print(text, "---> 부정감성")
else :
    print(text, "---> 긍정감성")

#모델 저장
import pickle #객체를 저장할때 사용하는 모듈
with open('./텍스트분석/data/감정분석모델_tfidf.pickle', 'wb') as f :
    pickle.dump(tfVec, f)
with open('./텍스트분석/data/감정분석모델_model.pickle', 'wb') as f :
    pickle.dump(model, f)
