#문장유사도 : 문장간의 유사한 정도를 알아내는 기법
#줄거리가 비슷한 영화를 추천 모델

#문장 유사도 방법
#코사인 유사도 : 문장을 벡터화하고 두 벡터간의 각도값이 0에 
#               가까우면 두 문장은 유사하다고 판단
#               cos(0) = 1
#               두벡터간의 각도의 코사인값 => 1에 가까우면 
#               두문장이 유사함

#문장 벡터화 : 텍스트를 구성하는 단어를 추출하고 이를 숫자형인
#              값인 벡터로 표현한다.

#대표적인 벡터화 방법
#Bow(Bag of Words), Word Embedding 방식

#1) Bow는 문서가 가지고 있는 모든 단어에 대해 순서는 
#무시한 채 빈도만 고려하여 단어가 얼마나 자주 등장하는가로
#벡터를 만든다.
#Bow 알고리즘 종류 : 카운트 기반 벡터화 , TF-IDF 벡터화
#1-1) 카운트 기반 벡터화 : 단어가 전체 문서에 등장하는 
#횟수(빈도수)를 기반으로 벡터화하는 방식(정수값으로 할당)
#특징 : 단어 빈도수가 높을 수록 중요한 단어로 다루어진다.
text = ['I go to my home my home is very large', 
    'I went out my home I go to the market', 
    'I bought a yellow lemon I go back to home']
from re import I
from sklearn.feature_extraction.text \
    import CountVectorizer, TfidfVectorizer
countVec = CountVectorizer()
countVec.fit(text)
vecText = countVec.transform(text)
vecText.toarray()

# array([[0, 0, 1, 2, 1, 1, 0, 0, 2, 0, 0, 1, 1, 0, 0],
#        [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0],
#        [1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1]], dtype=int64)
countVec.vocabulary_.items()
sorted(countVec.vocabulary_.items())
# [('back', 0), ('bought', 1), ('go', 2), ('home', 3), ('is', 4), ('large', 5), ('lemon', 6), ('market', 7), ('my', 8), ('out', 9), ('the', 10), ('to', 11), ('very', 12), ('went', 13), ('yellow', 14)]

#1-2) TF-IDF 기반 벡터화 : 빈도가 높은 단어가 문서에서 많이 사용된
#중요한 단어일 수도 있지만, 단지 문서 구성상 많이 사용하는
#단어(관사, 접속사 등) 단어 일 수 있음.
#이러한 문제점을 보완하기 위해 특정 문서에서 많이 나타나는
#단어는 해당 문서의 단어 벡터에 가중치를 높이고
#모든 문서에서 많이 나타나는 단어는 문장을 구성하는 범용적인 단어로
#취급하여 가중치를 낮추는 방식. (벡터의 값이 실수가 됨)
tfVec = TfidfVectorizer()
tfVec.fit(text)
textVec = tfVec.transform(text)
textVec.toarray()
text = ['I go to my home my home is very large', 
    'I went out my home I go to the market', 
    'I bought a yellow lemon I go back to home']
# array([[0.        , 0.        , 0.2170186 , 0.4340372(home) , 0.36744443,
#         0.36744443, 0.        , 0.        , 0.55890191(my), 0.        ,
#         0.        , 0.2170186 , 0.36744443, 0.        , 0.        ],
sorted(tfVec.vocabulary_.items())
# [('back', 0), ('bought', 1), ('go', 2), ('home', 3), ('is', 4), ('large', 5), ('lemon', 6), ('market', 7), ('my', 8), ('out', 9), ('the', 10), ('to', 11), ('very', 12), ('went', 13), ('yellow', 14)]
#------------------------------------------------
#TF-IDF 벡터화의 파라미터
#(1) min_df (min최솟값, df:document frequency) :
#    특정 단어가 나타나는 문서 수의 최솟값 지정
#ex) 'home' 3개의 문장에서 모두 나왔기때문에 df값은 3이된다.
# 특징 : 모든 문장에 너무 적게 나오는 단어는 제외시키는 목적
tfVec = TfidfVectorizer(min_df=3)
tfVec.fit(text)
textVec = tfVec.transform(text)
textVec.toarray()
sorted(tfVec.vocabulary_.items())
#[('go', 0), ('home', 1), ('to', 2)]
#(2) max_df : 최대 몇개의 문서에 걸쳐 포함된 단어까지 학습 단어로
#사용할 것인지 [0.0 1.0] 실수 범위로 지정할 수 있음
#(문서의 비율을 설정)
#특징 : 모든 문장에 많이 나오는 단어를 제외시키는 목적
#전체 문장의 90%이하로 나오는 단어만 벡터화 단어로 선정
tfVec = TfidfVectorizer(max_df=0.9)
tfVec.fit(text)
textVec = tfVec.transform(text)
textVec.toarray()
sorted(tfVec.vocabulary_.items())
#(3) ngram_range : 단어의 묶음 범위 설정
#ngram_range=(1,2) 한단어와 한단어와 옆에있는 단어까지(총 2개)
#ngram_range=(1,1)
text = ['I go to my home my home is very large', 
    'I went out my home I go to the market', 
    'I bought a yellow lemon I go back to home']
tfVec = TfidfVectorizer(ngram_range=(1,2))
tfVec.fit(text)
textVec = tfVec.transform(text)
textVec.toarray()
sorted(tfVec.vocabulary_.items())
#(4) max_feature : tf-idf 벡터의 피쳐 개수(단어 최대 개수) 설정
#ex) max_feature=100, 상위 100개의 단어만 벡터의 피처로 생성
tfVec = TfidfVectorizer(max_features=5)
tfVec.fit(text)
textVec = tfVec.transform(text)
textVec.toarray()
sorted(tfVec.vocabulary_.items())
#[('back', 0), ('go', 1), ('home', 2), ('my', 3), ('to', 4)]

#(5) token_pattern : 문장을 토큰화하는 기준, 정규표현식으로 작성
#                         \b단어의 경계,\w(숫자,문자)
#단어의경계(스페이스, 특수기호,)단어들(단어의 경계)
#(?u)유니코드
tfVec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
tfVec.fit(text)
textVec = tfVec.transform(text)
textVec.toarray()
sorted(tfVec.vocabulary_.items())
#(6) stop_words : 불용어 제거
stop_words = ['a', 'is']
tfVec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", 
            stop_words=stop_words)
tfVec.fit(text)
textVec = tfVec.transform(text)
textVec.toarray()
sorted(tfVec.vocabulary_.items())
#----------------------------------------------------------
#코사인 유사도 : 벡터간의 각도를 계산해서 코사인 값을 얻음
#벡터간의 각도가 0에 가까우면 두 벡터는 비슷한 벡터라고 판단
#cos(0) = 1, 
#코사인 값이 1에 가까우면 두 문장이 비슷하다고 판단
#두 벡터의 코사인값 구하기
#cos() = (두 벡터의 내적) / (두 벡터의 크기의 곱)
from numpy import dot #벡터의 내적
from numpy.linalg import norm #벡터의 크기
#벡터의 내적 
#a1 = [1,2]
#a2= [3,4]
#a1'a2 = 1*3 +2*4 (같은 인덱스 끼리 곱의 합)
#벡터의 크기
#|a1| = sqrt(1^2+2^2) (각 요소 제곱의 합의 루트값)
#|a2| = sqrt(3^2+4^2) (각 요소 제곱의 합의 루트값)
def cos(v1,v2) :
    return dot(v1,v2)/(norm(v1)*norm(v2))
tfVec = TfidfVectorizer()
tfVec.fit(text)
textVec = tfVec.transform(text)
textVec.toarray()
sorted(tfVec.vocabulary_.items())
v1 = textVec.toarray()[0]
v2 = textVec.toarray()[1]
v3 = textVec.toarray()[2]
cos(v1,v2), cos(v1,v3), cos(v2,v3)
# (0.3953976635190176, 0.22822744096614822, 0.1964177656475973)
text = ['I go to my home my home is very large', 
    'I went out my home I go to the market', 
    'I bought a yellow lemon I go back to home']
#--------------------------------------------------------
#줄거리가 비슷한 영화 추천
#사용자가 아는 영화의 제목을 입력
#그 영화와 비슷한 줄거리의 다른 영화를 추천
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv('./텍스트분석/data/movies_metadata.csv',
    low_memory=False)
df.info()
df.head()

#데이터 수가 너무 많아 토큰화하는데 오래걸림
df = df.head(20000) #2만개 데이터만 사용
df['title'] #영화제목
df['overview'] #영화줄거리

#결측치 확인
df.info() 
# title                  19998 non-null  object
# overview               19865 non-null  object
#결측치 있는 행 제거
df.dropna(subset=['overview', 'title'], inplace=True)
df.info()
df['overview'].isna().sum()
df['title'].isna().sum()

#인덱스재배열 : 결측이 있는 행 제거했으므로 인덱스가 연속적이지않다.
df.reset_index(drop=False, inplace=True)
len(df) 
df #0~19862

#줄거리를 tf-idf 벡터화
#줄거리 텍스트 전처리
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#(1) 정제
df['전처리_줄거리'] = df['overview'].str.replace(r'[^a-zA-Z0-9]+'," ")
df['전처리_줄거리'].head()
#(2) 정규화
df['전처리_줄거리'] = df['전처리_줄거리'].str.lower()
#(3) 토큰화
df['전처리_줄거리'] = df['전처리_줄거리'].apply(lambda x:word_tokenize(x))
df['전처리_줄거리']
#(4) 불용어 제거
stopWords = set(stopwords.words('english'))
stopWords
df['전처리_줄거리'] = df['전처리_줄거리'].\
        apply(lambda wList:[w for w in wList if w not in stopWords])
df['전처리_줄거리'] 
#(5) 어간추출 및 표제어 추출
lemma = WordNetLemmatizer()
df['전처리_줄거리'] = df['전처리_줄거리'].\
        apply(lambda wList:[lemma.lemmatize(w) for w in wList])
df['전처리_줄거리']
#(6) 단어 합치기
df['전처리_줄거리'] = df['전처리_줄거리'].\
        apply(lambda wList:" ".join(wList))
df['전처리_줄거리']

tfVec = TfidfVectorizer(min_df=2, max_df=0.9, stop_words='english') 
tfVec.fit(df['전처리_줄거리'])
overviewVec = tfVec.transform(df['전처리_줄거리'])
sorted(tfVec.vocabulary_.items())
overviewVec.shape #(19863, 21919) #문장 19863, 벡터의 단어수 21919
overviewVec.toarray()[0]
# array([0., 0., 0., ..., 0., 0., 0.])

#줄거리 데이터 끼리 코사인 유사도 구하기
from sklearn.metrics.pairwise import cosine_similarity
cos_sim = cosine_similarity(overviewVec, overviewVec)
cos_sim.shape #(19863, 19863)
cos_sim[0][0] #첫번째 영화와 첫번째 영화의 코사인값 1.000
cos_sim[0][1] #첫번째 영화와 두번째 영화의 코사인값 0.015
cos_sim[0][2] #첫번째 영화와 세번째 영화의 코사인값 0.0
cos_sim[1][0] #두번째 영화와 첫번째 영화의 코사인값 0.015
cos_sim[1][2] #두번째 영화와 세번째 영화의 코사인값 0.04284
#--------------------------------------------------------
#영화 제목에 따른 인덱스 값 시리즈형태로 가져옴
indexes = pd.Series(df.index, index=df['title']).drop_duplicates()
#drop_duplicates 중복제거
indexes.head()

#해당 영화의 인덱스 값 얻기
idx = indexes['Toy Story']
#해당 영화와 모든 영화와의 유사도 값 얻기
sim_scores = list(enumerate(cos_sim[idx]))
sim_scores
#유사도 내림차순으로 영화 정렬
sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)
#가장 유사한 10개의 영화만 얻기
sim_scores = sim_scores[1:11] #인덱스 0은 자기자신이므로 제외
sim_scores
#인덱스만 가져오기
movie_idx = [x[0] for x in sim_scores]
#영화제목 얻기
df['title'].iloc[movie_idx]
# 15282               Toy Story 3
# 2979                Toy Story 2

def get_recommedation(title, cos_sim=cos_sim) :
    #해당 영화의 인덱스 값 얻기
    idx = indexes[title]
    #해당 영화와 모든 영화와의 유사도 값 얻기
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores
    #유사도 내림차순으로 영화 정렬
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)
    #가장 유사한 10개의 영화만 얻기
    sim_scores = sim_scores[1:11] #인덱스 0은 자기자신이므로 제외
    sim_scores
    #인덱스만 가져오기
    movie_idx = [x[0] for x in sim_scores]
    #영화제목 얻기
    return df['title'].iloc[movie_idx]
get_recommedation('The Dark Knight')

#숙제
#한국영화 추천 코드 완성
#위에 것과 동일한 원리로 하면됨
#데이터 : movies04293.csv

