import pandas as pd
movies = pd.read_csv('./라이브러리/movie.csv')
movies.head()

movies.columns #칼럼정보
movies.index #인덱스정보  #RangeIndex(start=0, stop=4916, step=1)
movies.to_numpy() #numpy 객체로 변환 (array, 행렬)

#pandas의 데이터 타입
#float : 부동소수점 형식, 결측치 지원(Null값)
#int : 정수형식, 결측치 지원X
#int64 : 정수형식, 결측치 지원
#object : 문자열 및 기타 형식, 결측치 지원
#category : 범주형 데이터, 결측치 지원
#bool : 불리언 , 결측치 지원 X
#boolean : 불리언, 결측치 지원
#datetime64 : 날짜형식, 결측치 지원

movies.dtypes #각 열별 타입정보
movies.dtypes.value_counts() #타입별 개수
# float64    13
# object     12
# int64       3

#데이터 프레임의 전체 정보를 요약
movies.info()

#-------------------------------------------------------------
#단일 열 선택 (Series)
movies.director_name #director_name열 데이터 가져오기
type(movies.director_name) #<class 'pandas.core.series.Series'>
type(movies) #<class 'pandas.core.frame.DataFrame'>
movies['director_name']
#열이름을 통한 데이터 가져오기
#       가져올 인덱스 범위 지정, 열이름
movies.loc[:, 'director_name']
movies.loc[10:30, 'director_name']

#열 위치(인덱스)를 통한 데이터 가져오기
movies.info()
movies.iloc[:, 1]

directors = movies.director_name

#열정보
directors.size #데이터 수
directors.count() #결측치 제외 데이터 수
directors.shape #(행열의 크기)
len(directors) #데이터의 수
directors.unique() #중복없이 데이터 가져오기
directors.value_counts() #데이터별 개수(범주형 변수)
directors.value_counts(normalize=True) #상대빈도값(범주형 변수)
#범주형 변수 : 동일한 성질을 갖는 분류나 범위로 나눌수있는 변수
#              질적변수(데이터의 구분을 위한 변수 ex)혈액형, 이름,..)

#열에서 데이터 가져오기
directors.head() #가장 처음 5개의 데이터 조회
directors.head(10) #가장 처음 10개의 데이터 조회
directors.sample(n=5, random_state=13) #랜덤하게 5개의 데이터를 가져오기
#               데이터수, 랜덤시드값(시드값에 따라 랜덤 추출 인덱스 결정)
# 4322             Roger Avary
# 3753    Michael Winterbottom
# 1057       Christopher Nolan
# 2673       Tina Gordon Chism
# 2473          Chris Robinson
directors[:5] #인덱스를 통한 데이터 가져오기
directors[:10:2] #인덱스 슬라이싱 증감값까지 지원

#열 집계 함수
fb_likes = movies.actor_1_facebook_likes
fb_likes
fb_likes.min()
fb_likes.max()
fb_likes.mean() #평균
fb_likes.median() #중앙값
fb_likes.std() #표준편차
fb_likes.describe() #전체 통계정보 요약 #수치형변수
directors.describe() #범주형 변수의 통계정보

fb_likes.quantile(0.25) #Q1 하위 25%
fb_likes.quantile([i*0.1 for i in range(1,10)]) #Q1 하위 25%

#결측치 확인
fb_likes.isna() #각 값 별로 결측치 이면 True, 아니면 False

#인덱싱 마스크 True인 값만 리턴함
fb_likes[fb_likes.isna()] #결측치인 값만 가져온다
fb_likes.isna().value_counts() #결측치가 아닌 개수, 결측치 개수
# False    4909
# True        7
fb_likes.isna().sum() #7

#결측치 채우기
fb_likes_filled = fb_likes.fillna(0)
fb_likes_filled[fb_likes.isna()]
fb_likes_filled.isna().sum()

#결측치 삭제
fb_likes_dropped = fb_likes.dropna()
fb_likes_dropped.isna().sum()
fb_likes_dropped.size

#단일열(시리즈) 연산
imdb_score = movies['imdb_score']
imdb_score
imdb_score+1 #모든 열의 값에 1이 더해진다
imdb_score*2.5
imdb_score > 7
imdb_score[imdb_score > 7]

#열이름 변경
            #바뀔 열 이름:#변경 후 열이름
col_dic = {'director_name':'director', 
                'num_critic_for_reviews':'critic_reviews'}
movies.rename(columns=col_dic).info()
#  0   color                      4897 non-null   object
#  1   director                   4814 non-null   object
#  2   critic_reviews             4867 non-null   float64
cols = movies.columns.to_list()
cols
cols[1] = 'director'
cols[2] = 'critic_reviews'
movies.columns = cols
movies.info()

#열생성
movies['has_seen'] = 0
movies.info()
# 27  movie_facebook_likes       4916 non-null   int64
# 28  has_seen                   4916 non-null   int64
movies.head()
#----------------------------------------------------------------
#열선택 (여러 열, 데이터 프레임)
actors_director = movies[['actor_1_name','actor_2_name','actor_3_name']]
actors_director.head()

movies.loc[:, ['actor_1_name','actor_2_name','actor_3_name']]
movies.info() #10, 6, 14
movies.iloc[:, [10,6,14]]

#열이름에 actor가 포함된 경우
movies.filter(like='actor').head()
movies.filter(like='actor').info()
movies.filter(regex='facebook.*').info()

#데이터 프레임 요약(사이즈 및 통계정보)
movies.shape #(4916, 29) # 4916행, 29열
movies.size # 142564 = 4916*29 = 총 데이터 개수
len(movies) # 4916 행의 수

#집계 함수
movies.min() #열별 최솟값 얻기
movies.mean() #열별 평균값 얻기
movies
movies.min(axis=0) #열별 최솟값
movies.min(axis=1) #행별 최솟값
movies.mean(axis=0) #열별 평균값
movies.mean(axis=1) #행별 평균값
movies.mean(axis='index') #열별 평균값
movies.mean(axis='columns') #행별 평균값
#max(), std(), sum()....

movies.describe() #각 열별 통계 요약 정보
movies.describe().T # T 전치행렬(열과 행을 바꾼다)

movies.isna() #null 데이터 확인
movies.isna().sum() #열별 null 데이터 개수 확인
movies[movies.isna()] #Null 데이터만 확인

movies.count() #열별 null이아닌 데이터의 개수
movies.count(axis=0)
movies.count(axis=1) #행별 null이 아닌 데이터의 개수

movies.sum() #열별 합계
#수치형 데이터만 열별 합계를 보고싶은경우
movies.select_dtypes(['int64', 'float64']).info() #원하는 타입만 뽑아냄
movies.select_dtypes(['int64', 'float64']).sum(axis=0) #열별 합계

#평점 5점 초과인 데이터들의 열별(칼럼이름이 페이스북인) 합계
(movies.filter(regex='facebook.*')>5).sum() 
#--------------------------------------------------------------
#데이터 프레임 생성
#(1) 리스트를 값으로 갖는 딕셔너리를 통한 생성
name = ['홍길동', '임꺽정', '김하나']
age = [30, 40, 20]
peopleDic = {'name':name, 'age':age}
peopleDic

df = pd.DataFrame(peopleDic)
df #인덱스 자동으로 숫자로 형성됨

df = pd.DataFrame(peopleDic, index=['a','b','c'])
df
df.loc['a', :]
df.loc['a':'c', :]

#(2) 딕셔너리를 요소로 갖는 리스트를 이용한 생성
peopleList = [{'name':'홍길동', 'age':30},
            {'name':'임꺽정', 'age':40},{'name':'김하나', 'age':20}]
df =  pd.DataFrame(peopleList)
df

#-------------------------------------------------------------------
#pip install seaborn
import seaborn as sns
df = sns.load_dataset('titanic')
df.info()
df.head()

#Group by 기능
#성별별 생존자 수
df['sex'].value_counts()
# male      577
# female    314
df.groupby('sex').sum()['survived']
# sex
# female    233
# male      109

# GROUP BY에 들어가는 열 데이터형 => 범주형 변수(질적 변수)
df.groupby(['class','sex']).mean()['survived']
# class   sex   
# First   female    0.968085
#         male      0.368852
# Second  female    0.921053
#         male      0.157407
# Third   female    0.500000
#         male      0.135447

df.groupby(['class','sex']).agg(['mean','max','count'])[['age', 'fare']]
#                     age                    fare
#                     mean   max count        mean       max count
# class  sex
# First  female  34.611765  63.0    85  106.125798  512.3292    94
#        male    41.281386  80.0   101   67.226127  512.3292   122
# Second female  28.722973  57.0    74   21.970121   65.0000    76
#        male    30.740707  70.0    99   19.741782   73.5000   108
# Third  female  21.750000  63.0   102   16.118810   69.5500   144
#        male    26.507589  74.0   253   12.661633   69.5500   347

# 16시에 이어서 진행하겠습니다.
group_df = df.groupby(['class','sex']).agg(['mean','max','count'])[['age', 'fare']]
group_df.xs('First')
group_df.xs('First').xs('male') #xs : 멀티인덱스
group_df.xs('First').xs('male')['age']
group_df.xs('First').xs('male')['fare']

group_df.xs(('Second', 'male'), level=['class', 'sex'])
#                    age                   fare
#                   mean   max count       mean   max count
# class  sex
# Second male  30.740707  70.0    99  19.741782  73.5   108
group_df.xs(('Second', 'male'))
# age   mean      30.740707
#       max       70.000000
#       count     99.000000
# fare  mean      19.741782
#       max       73.500000
#       count    108.000000
group_df.xs(('Second', 'male')).xs(('fare','mean'))

#멀티인덱스 해제
group_df.reset_index()
#     class     sex        age                    fare
#                         mean   max count        mean       max count
# 0   First  female  34.611765  63.0    85  106.125798  512.3292    94
# 1   First    male  41.281386  80.0   101   67.226127  512.3292   122
# 2  Second  female  28.722973  57.0    74   21.970121   65.0000    76
# 3  Second    male  30.740707  70.0    99   19.741782   73.5000   108
# 4   Third  female  21.750000  63.0   102   16.118810   69.5500   144
# 5   Third    male  26.507589  74.0   253   12.661633   69.5500   347
group_df.reset_index()['age']
group_df.reset_index()['age']['mean']
conv_df = group_df.reset_index()
conv_df.columns = ['객실등급','성별','평균_나이','최대_나이'
            ,'개수_나이','평균_운임','최대_운임','개수_운임']
conv_df.head()
conv_df.info()
#-------------------------------------------------------------
#데이터 프레임 병합
#(1) concat : 데이터 프레임 이어 붙이기
df1 = pd.DataFrame({"a":['a'+str(i) for i in range(4)],
                    "b":['b'+str(i) for i in range(4)], 
                    "c":['c'+str(i) for i in range(4)]}, index=[0,1,2,3]
                )
df1
#     a   b   c
# 0  a0  b0  c0
# 1  a1  b1  c1
# 2  a2  b2  c2
# 3  a3  b3  c3
df2 = pd.DataFrame({"a":['a'+str(i) for i in range(4)],
                    "b":['b'+str(i) for i in range(4)], 
                    "c":['c'+str(i) for i in range(4)],
                    "d":['d'+str(i) for i in range(4)]}, index=[2,3,4,5]
                )
df2
#     a   b   c   d
# 2  a0  b0  c0  d0
# 3  a1  b1  c1  d1
# 4  a2  b2  c2  d2
# 5  a3  b3  c3  d3
concat_df = pd.concat([df1, df2]) #axis=0 행방향으로 결합
concat_df
#     a   b   c    d
# 0  a0  b0  c0  NaN
# 1  a1  b1  c1  NaN
# 2  a2  b2  c2  NaN
# 3  a3  b3  c3  NaN
# 2  a0  b0  c0   d0
# 3  a1  b1  c1   d1
# 4  a2  b2  c2   d2
# 5  a3  b3  c3   d3
concat_df.reset_index()
#    index   a   b   c    d
# 0      0  a0  b0  c0  NaN
# 1      1  a1  b1  c1  NaN
# 2      2  a2  b2  c2  NaN
# 3      3  a3  b3  c3  NaN
# 4      2  a0  b0  c0   d0
# 5      3  a1  b1  c1   d1
# 6      4  a2  b2  c2   d2
# 7      5  a3  b3  c3   d3
concat_df.reset_index(drop=True)
#     a   b   c    d
# 0  a0  b0  c0  NaN
# 1  a1  b1  c1  NaN
# 2  a2  b2  c2  NaN
# 3  a3  b3  c3  NaN
# 4  a0  b0  c0   d0
# 5  a1  b1  c1   d1
# 6  a2  b2  c2   d2
# 7  a3  b3  c3   d3
concat_df = pd.concat([df1, df2], ignore_index=True) #기존 인덱스 무시
concat_df
#     a   b   c    d
# 0  a0  b0  c0  NaN
# 1  a1  b1  c1  NaN
# 2  a2  b2  c2  NaN
# 3  a3  b3  c3  NaN
# 4  a0  b0  c0   d0
# 5  a1  b1  c1   d1
# 6  a2  b2  c2   d2
# 7  a3  b3  c3   d3
pd.concat([df1, df2], axis=1) #열방향 결합 #같은 인덱스 별로 값을 병합
#      a    b    c    a    b    c    d
# 0   a0   b0   c0  NaN  NaN  NaN  NaN
# 1   a1   b1   c1  NaN  NaN  NaN  NaN
# 2   a2   b2   c2   a0   b0   c0   d0
# 3   a3   b3   c3   a1   b1   c1   d1
# 4  NaN  NaN  NaN   a2   b2   c2   d2
# 5  NaN  NaN  NaN   a3   b3   c3   d3
pd.concat([df1, df2], axis=1, ignore_index=True) #열 이름이 새롭게 만들어짐

# 두 데이터 프레임의 합집합
pd.concat([df1, df2], axis=1, ignore_index=True, join='outer') #외부 Full 조인
#outer 방식이 디폴트

#교집합-같은 인덱스 있는 것 끼리만 결합
pd.concat([df1, df2], axis=1, ignore_index=True, join='inner') 
#교집합-같은 열이 있는 것 끼리만 결합
pd.concat([df1, df2], axis=0, ignore_index=True, join='inner') # df2의 d열이 사라짐

#(2) merge() : 두 데이터프레임을 각 데이터에 존재하는 고유값(key)값 기준 병합
#concat은 열이나, 인덱스 기준으로 결합
#merge() : 특정 값(key로 활용되는 값들) 기준으로 결합
merge_df = pd.merge(df1, df2, on='a')
merge_df
#       df1  df1 df2 df2
#     a b_x c_x b_y c_y   d
# 0  a0  b0  c0  b0  c0  d0
# 1  a1  b1  c1  b1  c1  d1
# 2  a2  b2  c2  b2  c2  d2
# 3  a3  b3  c3  b3  c3  d3