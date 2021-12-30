#빈도분석 : 텍스트 분석에서 가장 많이 쓰이는 분석 방법
#           빈도수가 높은 단어를 추려낼때 사용
#텍스트 분석에서 가장 기본적인 방법
#각 단어별로 빈도수를 얻는다

#pip install nltk
#영어 형태소 분석기
import glob #여러 파일들 한꺼번에 불러올때 유용한 모듈
import nltk
from nltk.corpus.reader import wordlist #영어 형태소 분석기
from nltk.tokenize import word_tokenize #문장 토큰화작업(단어별로 분리)
from nltk.corpus import stopwords #불용어 정보제공(분석할 가치가 없는 단어들)
from nltk.stem import WordNetLemmatizer #표제어추출(단어 원형 추출)
from collections import Counter #리스트내 데이터 개수 얻기
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS, WordCloud #워드클라우드 : 빈도분석 시각화 방법
#pip install wordcloud
import re
import pandas as pd
#nltk 데이터 다운
nltk.download('stopwords') #불용어 데이터 다운
nltk.download('punkt') #토크나이저 데이터 다운
nltk.download('wordnet') #표제어 데이터 다운
#다운 후 주석처리, 딱 한번만 다운로드 하면 된다.

#(1) 영어논문데이터 가져오기
all_files = glob.glob('./텍스트분석/영어논문데이터/myCabinetExcelData*.xls')
#                                                         *은 어떤 단어가 와도됨
all_files #데이터 경로 리스트로 불러옴

data_list = []
for file in all_files :
    #엑셀을 읽어들때 모듈 설치
    df = pd.read_excel(file) #pip install xlrd
    data_list.append(df)
data_list

#행기준으로 데이터 병합
data_df = pd.concat(data_list, axis=0, ignore_index=True)
data_df.head()
data_df.info()

#불필요한 칼럼제거
#Unnamed: 0
del data_df['Unnamed: 0']
data_df.info()

#(2) 데이터 전처리
#(2-1) 정제 : 불필요한 기호나 문자 제거
data_df['제목']
data_df['전처리_제목'] = data_df['제목'].str.replace(r'[^a-zA-Z]+', " ")
data_df['전처리_제목']
#(2-2) 정규화 : 대소문자 통합, 유사의미 단어 통합
data_df['전처리_제목'] = data_df['전처리_제목'].str.lower() #소문자화
data_df['전처리_제목'] 
#(2-3) 토큰화 : 데이터를 토큰으로 정한 기본 단위로 분리
#apply() 메서드를 통해 한행의 데이터를 다음의 람다식 적용하여 결과값 반환됨
data_df['전처리_제목'] = data_df['전처리_제목'].apply(lambda x:word_tokenize(x))
data_df['전처리_제목']
#(2-4) 불용어 제거 : 분석할 의미가 없는 불필요한 단어 제거
stopwordData = set(stopwords.words('english')) #영어 불용어 목록 불러오기
stopwordData #of, by.....
data_df['전처리_제목'] = data_df['전처리_제목'].\
        apply(lambda wList:[w for w in wList if w not in stopwordData])
data_df['전처리_제목'] 
#(2-5) 어간추출 및 표제어 추출 : 단어의 원형을 가져온다
lemma = WordNetLemmatizer() #표제어 추출 객체 생성
data_df['전처리_제목'] = data_df['전처리_제목'].\
        apply(lambda wList:[lemma.lemmatize(w) for w in wList])
data_df['전처리_제목'] 

#(3) 단어별 빈도수 얻기
#단어 리스트 병합
wordList = [] #단어가 저장될 리스트
for data in data_df['전처리_제목'] :
    wordList += data
wordList
countWord = Counter(wordList)
countWord

#(3-1) 상위 50개의 단어만 얻기
countWord.most_common(50)
#(3-2) 상위 50개의 단어중에서 단어의 길이가 2이상인 단어만 추출
countWordDic = {}
for w, cnt in countWord.most_common(50) :
    if len(w) > 1 :
        countWordDic[w] = cnt
countWordDic

#(4) 시각화
#(4-1) 막대그래프로 시각화
plt.bar(range(len(countWordDic)), countWordDic.values(), align='center')
plt.xticks(range(len(countWordDic)), countWordDic.keys(), rotation=80)
plt.show()
#bigdata 논문에서 가장 많이 쓰이는 단어목록 시각화

#(4-2) 년도별 빅데이터 논문 수 시각화
#년도에 몇개의 논문이 출판되었는지 추세 확인
sorted(data_df['출판일'].unique(), reverse=True)
#데이터 프레임 특정 칼럼 기준 정렬
data_df_sorted = data_df.sort_values(by=['출판일'])
data_df_sorted['출판일'].head() #내림차순 정렬

#group by를 통해 년도별 논문수 집계
group_df = data_df_sorted.groupby('출판일', as_index=False)['제목'].count()
group_df

plt.figure(figsize=(12,5))
plt.xlabel('year')
plt.ylabel('doc_count')
plt.grid(True)
plt.plot(group_df['출판일'], group_df['제목'])
plt.show()

#(4-3) 워드클라우드 - 단어 빈도 시각화
del countWordDic['big']
del countWordDic['data']
wc = WordCloud(background_color='ivory', width=800, height=600)
cloud = wc.generate_from_frequencies(countWordDic)
plt.imshow(cloud)
plt.axis('off') #축 제거
plt.savefig('./텍스트분석/영어논문 클라우드 시각화.jpg')
plt.show()

