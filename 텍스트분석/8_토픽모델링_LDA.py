#8_토픽모델링_LDA.py
#LDA 가장 대표적인 토픽 모델링 알고리즘
#문서에서 토픽(주제)들을 추출하는 방법

#원리 : 문서들은 토픽들의 혼합으로 구성되어져 있으며,
#토픽들은 확률 분포에 기반하여 단어들을 생성한다고 가정

import pandas as pd
#1) 데이터 로드
df = pd.read_csv('./텍스트분석/data/abcNewsData.csv')
df.head()
df.info()

df = df.head(10000)

#2) 데이터 전처리
import nltk
from nltk.corpus.reader import wordlist #영어 형태소 분석기
from nltk.tokenize import word_tokenize #문장 토큰화작업(단어별로 분리)
from nltk.corpus import stopwords #불용어 정보제공(분석할 가치가 없는 단어들)
from nltk.stem import WordNetLemmatizer #표제어추출(단어 원형 추출)
#(2) 데이터 전처리
#(2-1) 정제 : 불필요한 기호나 문자 제거
df['headline_text']
df['전처리_제목'] = df['headline_text'].str.replace(r'[^a-zA-Z]+', " ")
df['전처리_제목']
#(2-2) 정규화 : 대소문자 통합, 유사의미 단어 통합
df['전처리_제목'] = df['전처리_제목'].str.lower() #소문자화
df['전처리_제목'] 
#(2-3) 토큰화 : 데이터를 토큰으로 정한 기본 단위로 분리
#apply() 메서드를 통해 한행의 데이터를 다음의 람다식 적용하여 결과값 반환됨
df['전처리_제목'] = df['전처리_제목'].apply(lambda x:word_tokenize(x))
df['전처리_제목']
#(2-4) 불용어 제거 : 분석할 의미가 없는 불필요한 단어 제거
stopwordData = set(stopwords.words('english')) #영어 불용어 목록 불러오기
stopwordData #of, by.....
df['전처리_제목'] = df['전처리_제목'].\
        apply(lambda wList:[w for w in wList if w not in stopwordData])
df['전처리_제목'] 
#(2-5) 어간추출 및 표제어 추출 : 단어의 원형을 가져온다
lemma = WordNetLemmatizer() #표제어 추출 객체 생성
df['전처리_제목'] = df['전처리_제목'].\
        apply(lambda wList:[lemma.lemmatize(w) for w in wList])
df['전처리_제목'] 
#(2-6) 문자열 합치기
df['전처리_제목'] = df['전처리_제목'].\
        apply(lambda wList:' '.join(wList))
df['전처리_제목']

#(2-7) TF-IDF 벡터화
from sklearn.feature_extraction.text import TfidfVectorizer
tfVec = TfidfVectorizer(stop_words='english', max_features=1000,
            smooth_idf=True) #smooth_idf:값을 완만하게 처리
tfVec.fit(df['전처리_제목'])
titleVec = tfVec.transform(df['전처리_제목'])
titleVec.shape #(10000, 1000) #만개의 문장, 각 문장별 1000개의 단어로 벡터화
tfVec.vocabulary_.items()

#3) 모델 구축 및 훈련
from sklearn.decomposition import LatentDirichletAllocation
#n_component : 토픽 수
lda_model = LatentDirichletAllocation(n_components=5, learning_method='online',
        random_state=35, max_iter=20)
lda_model.fit(titleVec)
lad_topic = lda_model.transform(titleVec)
lad_topic.shape #(10000, 5) #각 기사별 5개의 토픽 비중
lad_topic[:5]

lda_model.components_.shape #(5, 1000) 각 토픽에 따른 단어 비중
lda_model.components_
tfVec.get_feature_names() # 1000개의 단어 이름
term = tfVec.get_feature_names()

#각 토픽별 상위 5개의 단어 출력
for i, topic in enumerate(lda_model.components_) :
    #단어별 토픽 비중 오름 차순
    sortList = [(term[i], topic[i].round(5)) for i in topic.argsort()]
    #토픽 비중이 높은 단어 5개 출력
    print(f'Topic {i} : ', sortList[-1:-11:-1])

# Topic 0 :  [('world', 64.46342), ('anti', 56.17383), ('attack', 54.72224), ('iraq', 53.79076), ('cup', 52.01087)]      
# Topic 1 :  [('baghdad', 87.6414), ('woman', 50.74395), ('lead', 49.12564), ('hit', 46.70261), ('urged', 46.4277)]
# Topic 2 :  [('iraq', 129.82176), ('iraqi', 95.9436), ('police', 87.91444), ('war', 80.73754), ('death', 62.0871)]
# Topic 3 :  [('say', 80.13526), ('win', 71.05685), ('hospital', 51.0581), ('protest', 49.09619), ('hope', 41.46936)]    
# Topic 4 :  [('plan', 94.97873), ('new', 81.21996), ('govt', 72.09126), ('troop', 65.23369), ('council', 64.82116)]   

#LDA 기법은 단순히 주제만 분류해주는 것이 아니라
#주제에 포함되는 키워드들을 보여주기 때문에 그 키워드들로 해당 
#주제를 해석하고 정의할 수 있음
#기계적 분류인 만큼 정확하지 않을 수 있으며 후보정 필요


