from matplotlib.pyplot import stem
import pandas as pd
#1) 데이터 로드
df = pd.read_json('./텍스트분석/data/네이버베스트셀러.json')
df.head()
df.info()

#2) 데이터 전처리
#2-1) 정제
df['desc'] = df['desc'].str.replace(r'[^가-힣]+', " ")
#2-2) 토큰화
from konlpy.tag import Okt
okt = Okt()
df['desc'] = df['desc'].apply(lambda x : okt.morphs(x, stem=True))
df['desc'].head()
#2-3) 불용어 제거 및 글자 길이 2이상인 단어만 추리기
stopwords = ['을','에게','인','제','의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
df['desc'] = df['desc'].\
apply(lambda x : [w for w in x if w not in stopwords and len(w)>1])
df['desc']
#2-4) 단어 합치기
#[가-힣] 단어들만 남음
#[ㄱ-ㅎ 가-힣 ㅏ-ㅣ] ㅋㅋㅋㅋ ㅜㅜ 및 단어 남음
df['desc'] = df['desc'].apply(lambda x : ' '.join(x))
df['desc']
#(2-5) TF-IDF 벡터화
from sklearn.feature_extraction.text import TfidfVectorizer
tfVec = TfidfVectorizer(min_df=3, max_df=0.9, smooth_idf=True) 
tfVec.fit(df['desc'])
titleVec = tfVec.transform(df['desc'])
titleVec.shape #(1350, 3426) #1350개의 문장, 각 문장별 3426개 단어로 벡터화
tfVec.vocabulary_.items() #벡터에 각 단어 목록

#3) 모델 구축 및 훈련
from sklearn.decomposition import LatentDirichletAllocation
#n_component : 토픽 수
lda_model = LatentDirichletAllocation(n_components=5, learning_method='online',
        random_state=35, max_iter=10)
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
    print(f'Topic {i} : ', sortList[-1:-6:-1])
# Topic 0 :  [('심리학', 9.68101), ('있다', 9.13583), ('되다', 8.42996), ('행복', 8.36321), ('에서', 8.2403)]
# Topic 1 :  [('있다', 8.48216), ('지식', 7.49634), ('논어', 6.62791), ('문화', 6.16795), ('이보', 5.73858)]
# Topic 2 :  [('철학', 22.86603), ('있다', 22.14739), ('되다', 18.75871), ('에서', 18.57225), ('글쓰기', 18.40361)]
# Topic 3 :  [('대통령', 4.93494), ('치료', 4.41071), ('히스토리', 4.3786), ('그림책', 4.16976), ('그래픽', 3.73614)]
# Topic 4 :  [('이다', 12.17346), ('마음', 11.51146), ('있다', 11.20985), ('가지', 10.74048), ('사람', 9.16373)]


# Topic 0 :  [('능력', 7.77972), ('용기', 5.68157), ('사람', 5.64883), ('아들러', 5.3302), ('있다', 5.02639)]
# Topic 1 :  [('문화', 5.28658), ('사랑', 5.28317), ('생각', 4.34934), ('실천', 4.14001), ('새롭다', 3.95289)]
# Topic 2 :  [('강의', 8.22669), ('교수', 8.00204), ('행복', 7.11968), ('맞춤법', 6.98053), ('세계사', 6.59445)]
# Topic 3 :  [('작가', 8.4056), ('욕망', 7.30134), ('웹소설', 6.97651), ('사랑', 5.7433), ('글쓰기', 5.28834)]
# Topic 4 :  [('있다', 19.10005), ('마음', 14.85661), ('우리', 12.62593), ('이다', 11.90583), ('사람', 11.71746)]
# Topic 5 :  [('인간', 9.75736), ('글쓰기', 9.05424), ('심리학', 7.78779), ('되다', 7.75709), ('논어', 7.16442)]
# Topic 6 :  [('니체', 4.40051), ('철학', 3.21985), ('고통', 3.02662), ('새롭다', 2.75036), ('해설', 2.72487)]
# Topic 7 :  [('어원', 6.82047), ('이야기', 6.43838), ('과학', 4.76734), ('이다', 4.75847), ('인류', 4.72297)]
# Topic 8 :  [('아이', 6.80229), ('노자', 6.65183), ('엄마', 5.98757), ('지식', 4.41949), ('용소', 3.55999)]
# Topic 9 :  [('철학', 9.96265), ('세상', 6.54982), ('위로', 6.2378), ('있다', 5.61975), ('니체', 5.20928)]

#과제 : survey.csv 앙케이드 분석 데이터
#긍정토픽(만족도가 3이상인 데이터), 부정토픽(만족도가 2이하인 데이터) 각각 3개씩 얻기
