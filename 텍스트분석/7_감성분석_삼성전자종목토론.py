#7_감성분석_삼성전자종목토론.py

import pickle
import re
from konlpy.tag import Okt

#감성분석 모델 불러오기
with open('./텍스트분석/data/감정분석모델_model.pickle','rb') as f:
    model = pickle.load(f)
with open('./텍스트분석/data/감정분석모델_tfidf.pickle','rb') as f:
    tfVec = pickle.load(f)

text = "ㅋㅋㅋㅋㅋㅋ 웃기다"
#텍스트 전처리
text = re.sub(r'[^ㄱ-ㅎ|가-힣|ㅏ-ㅣ]', " ", text)
#텍스트 벡터화

okt = Okt()
def okt_tokenizer(text) :
    tokens = okt.morphs(text, stem=True)
    return tokens
text_vec = tfVec.transform([text])
#예측
pred = model.predict(text_vec)
model.predict_proba(text_vec) #array([[0.88707606, 0.11292394]])
if(pred==0) :
    print(text, "---> 부정감성")
else :
    print(text, "---> 긍정감성")


#1) 데이터 로드
import pandas as pd
df = pd.read_json('./텍스트분석/data/삼성전자종목토론실.json')
df.head()
df.info()

df = df.head(30000)
#2) 데이터 전처리
#정제
df['title'] = df['title'].str.replace(r'[^ㄱ-ㅎ|가-힣|ㅏ-ㅣ]+', " ")
df['title'].head()
#벡터화
title_vec = tfVec.transform(df['title'])
#(3) 모델활용
emo_pred = model.predict(title_vec)
df['label']= emo_pred
df.head(10)

#데이터 탐색
#긍부정 댓글에서 명사만 추출 시각화
df['nouns'] = df['title'].apply(lambda x:okt.nouns(x))
df['nouns'].head()
#단어의 길이가 2이상 추출
df['nouns'] = df['nouns'].apply(lambda x:[w for w in x if len(w)>1])
df['nouns'].head()

pos_nouns = []
neg_nouns = []
df.info()

for i, val in df.iterrows() : #한행씩
    if val['label']==1 :
        pos_nouns += val['nouns']
    else :
        neg_nouns += val['nouns']
pos_nouns
neg_nouns

#빈도수 얻기
from collections import Counter
posCounter = Counter(pos_nouns) 
negCounter = Counter(neg_nouns) 
del posCounter['삼성'] 
del posCounter['전자']
del negCounter['삼성'] 
del negCounter['전자'] 
#워드 클라우드 시각화
from wordcloud import WordCloud
import matplotlib.pyplot as plt
font_path = 'c:/Windows/Fonts/malgun.ttf'
wc = WordCloud(font_path, background_color='ivory', width=800, height=600)
cloud = wc.generate_from_frequencies(negCounter)
plt.imshow(cloud)
plt.axis('off')
plt.show()
#------------------------------------------------------------------
#날짜별 긍정 댓글 비율
df_group = df.groupby(['date', 'label']).size()
df_group.head()
df_group[('2021-05-25', 0)] #부정댓글 개수
df_group[('2021-05-25', 1)] #긍정댓글 개수

ratio_df = pd.DataFrame(columns=['date', 'ratio'])
df['date'].unique()
for date in df['date'].unique() :
    try :
        total = df_group[(date,0)] + df_group[(date,1)]
        ratio = df_group[(date,1)]/total
    except :
        #total 값이 0인 경우
        ratio = 0
    ratio_df = ratio_df.append({'date':date, 'ratio':ratio}, ignore_index=True)
ratio_df

#만약 정렬이 안되어있는 경우
ratio_df.sort_values(by=['date'], axis=0, inplace=True, ascending=False)
ratio_df.reset_index(inplace=True, drop=True)
ratio_df.head()

#데이터 저장
ratio_df.to_csv('./텍스트분석/data/삼성전자종목토론실_긍정댓글비율.csv',
        index=False)

#12시 진행하겠습니다.
