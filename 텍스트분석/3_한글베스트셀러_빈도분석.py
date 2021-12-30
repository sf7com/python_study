#pip install konlpy #한글 형태소 분석기
#형태소 : 언어에서 의미가 있는 가장 작은 단위
#형태소 분석 : 형태소, 어근, 접두사/접미사 품사 등 
#             다양한 언어학적 속성으로 문장의 구조를 파악하는 것

#Python 3.8 버전
#Java 설치, 환경변수 세팅

#pip install --upgrade pip --user
#pip install j입력하고 tab누르기

from konlpy.tag import Okt
from matplotlib.pyplot import stem
#AttributeError: module 'tweepy' has no attribute 'StreamListener'
#pip install tweepy==3.10.0
okt = Okt()

okt.morphs('단독입찰보다 복수입찰의 경우') #형태소 분석
#['단독', '입찰', '보다', '복수', '입찰', '의', '경우']
okt.morphs('단독입찰보다 복수입찰의 경우', stem=True) #단어의 원형 
#형태소 + 태깅(품사)
okt.pos('단독입찰보다 복수입찰의 경우')
# [('단독', 'Noun'), ('입찰', 'Noun'), ('보다', 'Josa'), ('복수', 'Noun'), ('입찰', 'Noun'), ('의
# ', 'Josa'), ('경우', 'Noun')]

okt.pos('안녕하세요. 홍길동입니다. ^ ^ ㅋㅋㅋㅋ')
# [('안녕하세요', 'Adjective'), ('.', 'Punctuation'), ('홍길동', 'Noun'), ('입니다', 'Adjective'), ('.', 'Punctuation'), ('^', 'Punctuation'), ('^', 'Punctuation'), ('ㅋㅋㅋㅋ', 'KoreanParticle')]
okt.pos('안녕하세요. 홍길동입니다. ^ ^ ㅋㅋㅋㅋ', stem=True)
#어간 및 표제어추출

#명사분석
okt.nouns('유일하게 항공기 체계 종합개발을 갖고 있는 KAI는')
#['항공기', '체계', '종합', '개발']

#구문(Phrase) 분석
okt.phrases('날카로운 분석과 신뢰감 있는 진행으로')
# ['날카로운 분석', '날카로운 분석과 신뢰감', '날카로운 분석과 신뢰감 있는 진행', '분석', '신뢰', '진행']

#---------------------------------------------------------------
from konlpy.tag import Komoran
komoran = Komoran()
komoran.morphs('유일하게 항공기 체계 종합개발을 갖고 있는 KAI는')
#['유일', '하', '게', '항공기', '체계', '종합', '개발', '을', '갖', '고', '있', '는', 'KAI', '는']
komoran.pos('유일하게 항공기 체계 종합개발을 갖고 있는 KAI는')
# [('유일', 'NNG'), ('하', 'XSV'), ('게', 'EC'), ('항공기', 'NNP'), ('체계', 'NNG'), ('종합', 'NNG'), ('개발', 'NNG'), ('을', 'JKO'), ('갖', 'VV'), ('고', 'EC'), ('있', 'VV'), ('는', 'ETM'), ('KAI', 'SL'), ('는', 'JX')]
komoran.nouns('유일하게 항공기 체계 종합개발을 갖고 있는 KAI는')
#['유일', '항공기', '체계', '종합', '개발']
#--------------------------------------------------------------
import json
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
plt.rc('font', family='malgun gothic')

#(1) 데이터 불러오기
with open('./텍스트분석/data/네이버베스트셀러.json', 'r' , 
                encoding='utf-8') as f :
    data = json.loads(f.read())
data

#(2) 텍스트 전처리
#(2-1) 정제 : 한글이외의 문자 공백 처리
desc  = ""
title = ""
for item in data :
    if 'desc' in item.keys() :
        desc += re.sub(r'[^ㄱ-ㅎ가-힣ㅏ-ㅣ]', " ", item['desc']) + " "
    if 'title' in item.keys() :
        title += re.sub(r'[^ㄱ-ㅎ가-힣ㅏ-ㅣ]', " ", item['title']) + " "
desc
title

all_text = title +" " +desc
#(2-2) 토큰화 : 명사만 추려서 리스트로 반환
nouns = okt.nouns(all_text)
#단어 길이가 1인 명사 제거
nouns = [w for w in nouns if len(w)!=1]
nouns

#3) 단어별 빈도수 세기
nounCount = Counter(nouns)

#4)시각화 
#4-1) 막대그래프
plt.figure(figsize=(12,5))
plt.xlabel('키워드')
plt.ylabel('빈도수')
plt.grid(True)
top30List = nounCount.most_common(30)
plt.bar(range(len(top30List)), [x[1] for x in top30List], align='center')
plt.xticks(range(len(top30List)), [x[0] for x in top30List], rotation=75)
plt.show()

#4-2) 워드클라우드
#데이터 딕셔너리 형태
wordDic = {}
for word, cnt in top30List :
    wordDic[word] = cnt
font_path = 'c:/Windows/Fonts/malgun.ttf'
wc = WordCloud(font_path, background_color='ivory', width=800,
                height=600)
cloud = wc.generate_from_frequencies(nounCount)
plt.imshow(cloud)
plt.axis('off')
plt.show()
