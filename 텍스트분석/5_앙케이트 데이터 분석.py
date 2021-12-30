import pandas as pd
from konlpy.tag import Komoran

#1) 데이터 불러오기
df = pd.read_csv('./텍스트분석/data/survey.csv')
df.head()
df.info()

#2) 데이터 전처리
#2-1) 결측치 제거
df.dropna(inplace=True)
df.info()
#2-2) 정제
df['전처리_comment'] = df['comment'].str.replace(r'[^ㄱ-ㅎ|가-힣|ㅏ-ㅣ]+', " ")
df['전처리_comment']
#2-3) 토큰화, 명사추출
komoran = Komoran()
df['전처리_comment'] = df['전처리_comment'].apply(lambda x:komoran.nouns(x))
df['전처리_comment']
#2-4) 불용어제거
stopwords = ['을','에게','인','제','의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
df['전처리_comment'] = df['전처리_comment'].\
        apply(lambda wList:[w for w in wList if w not in stopwords])
df

#각 단어별 만족도
wList = []
sList = []
for i,data in df.iterrows() :
    comment = data['전처리_comment']
    wList += comment
    sList += [data['satisfaction']]*len(comment)
wList
sList
len(wList), len(sList) #(262, 262)

#데이터 프레임으로 변환
ns_df = pd.DataFrame({'단어':wList,'만족도':sList})
ns_df.head()

#단어별 평균 만족도와 단어의 빈도수
ns_group = ns_df.groupby('단어', as_index=False)['만족도']\
    .agg(['mean', 'count'])
ns_group

#자주 나오는 단어들의 만족도
#count >= 3
ns_group[ns_group['count']>=3]
word_df = ns_group[ns_group['count']>=3].sort_values(by='mean', ascending=False)
word_df[word_df['mean'] >= 3]
# 안심   4.333333      3
# 육아   4.333333      3
# 수    3.750000      8
# 관공서  3.500000      4
# 길    3.500000      4
# 활동   3.333333      3
# 시    3.000000      3
# 장소   3.000000      6
word_df[word_df['mean'] < 3]
# 지원   2.750000      4
# 것    2.666667      3
# 처리   2.333333      3
# 상가   2.200000      5
# 공원   2.200000      5
# 가로수  1.666667      3
# 앞    1.166667      6
# 역    1.142857      7
# 주차장  1.000000      5
#---------------------------------------------------------------
#코사인 유사도를 활용해서
#만족도가 높은 단어가 포함된 앙케이드 추출 
#만족도가 낮은 단어가 포함된 앙케이드 추출 

#단어 합치기
df['설문지'] = df['전처리_comment'].apply(lambda x:" ".join(x))
df['설문지']

#TF-IDF 벡터화
from sklearn.feature_extraction.text import TfidfVectorizer
tfVec = TfidfVectorizer()
tfVec.fit(df['설문지'])
textVec = tfVec.transform(df['설문지'])
textVec.shape #(84, 156) 84개의 문장, 각 벡터별 156개의 단어
tfVec.vocabulary_.items()


pos_text = ' '.join(list(word_df[word_df['mean'] >= 3].index))
pos_text
neg_text = ' '.join(list(word_df[word_df['mean'] < 3].index))
neg_text

#긍,부정 텍스트 벡터화
pos_vec = tfVec.transform([pos_text])
neg_vec = tfVec.transform([neg_text])
pos_vec.toarray()

#모든 문건과 긍정,부정 텍스트간의 유사도값 얻기
from sklearn.metrics.pairwise import cosine_similarity
pos_cos_sim = cosine_similarity(textVec, pos_vec)
neg_cos_sim = cosine_similarity(textVec, neg_vec)
df['pos_cos_sim'] = pos_cos_sim
df['neg_cos_sim'] = neg_cos_sim

df.sort_values(by='pos_cos_sim', ascending=False).head(10)
#       datetime              comment  satisfaction            설문지  pos_cos_sim  neg_cos_sim
# 40  2019-01-09      [소방, 활동, 안심, 수]             4     소방 활동 안심 수     0.505913     0.000000
# 15  2019-01-02                 [육아]             4             육아     0.464433     0.000000
# 52  2019-02-02  [자치, 단체, 활동, 안심, 수]             5  자치 단체 활동 안심 수     0.448265     0.000000
# 6   2019-02-02                 [장소]             2             장소     0.400359     0.000000
# 2   2019-02-18             [육아, 지원]             5          육아 지원     0.337557     0.275431
# 24  2019-02-25             [육아, 최고]             4          육아 최고     0.316971     0.000000
# 50  2019-01-22       [보도, 길, 안심, 수]             4      보도 길 안심 수     0.301620     0.000000
# 1   2019-02-25       [운동, 수, 장소, 것]             5      운동 수 장소 것     0.251043     0.000000
# 76  2019-01-29     [관공서, 상담, 때, 친절]             5    관공서 상담 때 친절     0.217535     0.000000
# 65  2019-02-13       [관공서, 출장소, 역전]             3     관공서 출장소 역전     0.217535     0.000000
df.sort_values(by='neg_cos_sim', ascending=False).head(10)
#       datetime                      comment  satisfaction                  설문지  pos_cos_sim  neg_cos_sim
# 21  2019-03-19                        [가로수]             3                  가로수     0.000000     0.454457
# 84  2019-03-09                         [공원]             2                   공원     0.000000     0.381942
# 4   2019-01-06                   [역, 앞, 상가]             2               역 앞 상가     0.000000     0.381942       
# 17  2019-02-12                         [상가]             4                   상가     0.000000     0.381942
# 48  2019-01-11           [가로수, 낙엽, 청소, 가로수]             1        가로수 낙엽 청소 가로수     0.000000     0.359539
# 55  2019-03-18  [주차장, 수, 요금, 역, 앞, 공공, 주차장]             1  주차장 수 요금 역 앞 공공 주차장     0.000000  0.307125
# 19  2019-02-20                    [쓰레기, 처리]             4               쓰레기 처리     0.000000     0.289622     
# 2   2019-02-18                     [육아, 지원]             5                육아 지원     0.337557     0.275431       
# 85  2019-04-02              [역, 앞, 주차장, 불편]             1           역 앞 주차장 불편     0.000000     0.265330 
# 38  2019-03-20              [역, 앞, 주차장, 부족]             1           역 앞 주차장 부족     0.000000     0.251857

#만족도 단어에 따라서 워드클라우 시각화
high_word = word_df[word_df['mean'] >= 3]
low_word = word_df[word_df['mean'] < 3]
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
from wordcloud import WordCloud
word_dic = {}
for idx, data in high_word.iterrows():
    word_dic[idx] = data['count']
word_dic

font_path = 'c:/Windows/Fonts/malgun.ttf'
wc = WordCloud(font_path, background_color='ivory', width=800, height=600)
cloud = wc.generate_from_frequencies(word_dic)
plt.imshow(cloud)
plt.axis('off')
plt.show()
