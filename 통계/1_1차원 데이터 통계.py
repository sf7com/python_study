#1. 데이터 중심지표 - 평균, 중앙값, 최빈값
#2. 데이터 산포도 지표 - 분산, 표준편차, 범위, 사분위범위
#3. 데이터의 정규화 - 표준화, 편차값
#4. 1차원 데이터의 시각화 -도수분포표, 히스토그램, 박스플롯

#1. 데이터 중심지표
import numpy as np
import pandas as pd

df = pd.read_csv('./통계/data/ch2_scores_em.csv',index_col='student number')
df.head()

#처음 10명의 영어 점수 얻기
en_scores = np.array(df['english'])[:10]
en_scores #array([42, 69, 56, 41, 57, 48, 65, 49, 65, 58], dtype=int64)

#1-1) 평균값
avg = sum(en_scores) / len(en_scores)
avg
np.mean(en_scores)

#1-2)중앙값 : 데이터를 크기 순으로 나열했을 때 중앙에 위치한 값
#특징 : 평균값에 비해서 이상값(극단적으로 높은값 혹은 적은값)의 영향을 덜 받는다.
#데이터의 개수가 홀수인경우 : (n+1)/2의 인덱스값
#데이터의 개수가 짝수인 경우 : n/2, n/2-1 인덱스 값들의 평균
sort_data = np.sort(en_scores)
n = len(sort_data) #데이터 개수
if n%2 == 1 :
    #데이터의 개수가 홀수인 경우
    median = sort_data[(n+1)/2-1]
else :
    m0 = sort_data[n//2-1]
    m1 = sort_data[n//2]
    median = (m0+m1) /2
median

np.median(en_scores)

#(1-3) 최빈값 : 데이터에서 가장 많이 나타나는 값
#scipy라이브러리 활용 최빈값 구하기
#scipy - 통계관련 라이브러리
#pip install scipy
from scipy.stats import mode
mode(en_scores)
#ModeResult(mode=array([65], dtype=int64), count=array([2]))
mode(en_scores)[0]
mode(en_scores)[1]
#------------------------------------------------------------------------------
#2.데이터의 산포도 지표
#데이터가 중심지표로 부터 얼마나 퍼져있는지 정도를 나타내는 값
#(1) 편차 : (데이터-데이터의 평균)
mean = np.mean(en_scores)
deviation = en_scores-mean
deviation
np.sum(deviation) #편차의 합은 0이다.

#(2) 분산 - 편차 제곱의 평균
devSquare = deviation**2
devSquare
np.mean(devSquare) #86.0
np.var(en_scores) #86.0

#(3) 표준편차 - 분산의 제곱근, 데이터의 단위와 동일
#분산에 루트를 취한다.
np.sqrt(np.var(en_scores)) #9.273618495495704
np.std(en_scores) #9.273618495495704

#(4) 범위 : 데이터의 최대값 - 최소값
np.max(en_scores) - np.min(en_scores)
#이상치에 영향을 많이 받음

#(5) 사분위 범위 : 데이터를 정렬했을 때
# 데이터의 하위 25% 위치의 데이터 Q1
# 데이터의 하위 75% 위치의 데이터 Q3
# 사분위범위 IQR = Q3-Q1
# data = [-50,1,2,3,4,5,6,7,8,9,10,100]
# 이상치에 영향을 덜 받는다.
scores_Q1 = np.percentile(en_scores,25) #하위 25% 위치의 값
scores_Q1
scores_Q3 = np.percentile(en_scores,75) #하위 75% 위치의 값
scores_Q3
scores_Q3 - scores_Q1
#---------------------------------------------------------
#3. 데이터의 정규화
#평균이나 분산에 의존하지 않고도 데이터의 상대적 위치를 알 수 있다.
#(데이터의 단위가 사라진다.)

#3-1) 표준화
#데이터에서 평균을 빼고 표준편차로 나누는 작업
#Z = (x-mean)/std

z = (en_scores-np.mean(en_scores)) / np.std(en_scores)
z
#array([-1.40182605,  1.50965882,  0.10783277, -1.50965882,  0.21566555,
#       -0.75482941,  1.07832773, -0.64699664,  1.07832773,  0.32349832])

#표준화된 데이터는 평균이 0, 표준편차 1
np.mean(z) #-1.6653345369377347e-17
np.std(z) #0.9999999999999999

#3-2) 편차값
# 평균이 50, 표준편차 10이 되도록 정규화
# z=50+10*(x-mean)/std
dev_vals = 50+10*(en_scores-np.mean(en_scores))/np.std(en_scores)
dev_vals
#array([35.98173948, 65.09658825, 51.07832773, 34.90341175, 52.15665546,
#       42.45170588, 60.78327732, 43.53003361, 60.78327732, 53.2349832 ])
np.mean(dev_vals) #50.0
np.std(dev_vals) #10.0000
# 편차값을 통해 어떤 학생이 우수한 성적을 얻었는지 한눈에 파악이 가능
#------------------------------------------------------------------
#4. 1차원 데이터의 시각화
#4-1) 도수분포표 : 분할된 구간과 그 구간에 속한 데이터의 갯수를 표로 정리
#계급 : 구간을 나눈 것   ex) 0~10점, 10~20점,..
#도수 : 각 계급에 속한 학생 수
#계급폭 : 구간의 크기, ex) 0~10점의 계급폭 10, 0~5점의 계급폭 5
#계급수 : 계급의 갯수

eng_scores = np.array(df['english']) #전체 학생의 영어 점수
freq, cls = np.histogram(eng_scores, bins = 10, range=(0,100))
#freq : 도수, cls : 계급, bins : 계급수, range 값의 최소 최대값
freq #array([ 0,  0,  0,  2,  8, 16, 18,  6,  0,  0], dtype=int64)
cls #array([  0.,  10.,  20.,  30.,  40.,  50.,  60.,  70.,  80.,  90., 100.])

#계급 만들기
freq_class = [f'{int(cls[i])}~{int(cls[i+1])}' for i in range(len(cls)) if i!=len(cls)-1]
freq_class #['0~10', '10~20', '20~30', '30~40', '40~50', '50~60', '60~70', '70~80', '80~90', '90~100']

hist_df = pd.DataFrame({'freq':freq}, index=freq_class)
hist_df

#계급값 : 계급을 대표하는 값, 계급의 중앙값
#ex) 60~70점의 계급값 : 65
class_vals = [sum(map(int,cls.split("~")))/2 for cls in freq_class]
class_vals = [sum([int(i) for i in cls.split("~")])/2 for cls in freq_class]
class_vals

#map 함수 리스트의 값을 새로운 값으로 바꿀때
# '10~20' => ['10','20'] => map(int,['10','20']) => [10,20] => sum([10,20])/2

hist_df['class_value'] = class_vals
hist_df

#상대도수 : 전체 데이터에 대해서 해당 계급의 데이터가 어느정도의 비율을 차지하고 있는지 비율
#상대도수 = 도수/전체 데이터수
rel_freq = freq/freq.sum()
hist_df['rel_freq'] = rel_freq
hist_df

#누적상대도수 : 전 계급 부터 해당 계급까지의 상대도수의 합
cum_rel_freq = np.cumsum(rel_freq)
hist_df['cum_rel_freq'] = cum_rel_freq
hist_df

#도수분포표의 최빈값 : 최대가 되는 도수의 계급값
hist_df['freq'].idxmax() #freq(도수)값이 최대인 인덱스 찾기
hist_df.loc[hist_df['freq'].idxmax()]
# freq            18.00
# class_value     65.00
# rel_freq         0.36
# cum_rel_freq     0.88
# Name: 60~70, dtype: float64

#4-2) 히스토그램 시각화 : 도수분포표를 막대그래프로 시각화
#시각화 라이브러리
#matplotlib => pip install matplotlib
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10,6)) #그래프 객체 만들기
ax = fig.add_subplot(111)
freq,_,_ = ax.hist(en_scores, bins=25, range=(0,100)) #그래프 생성
ax.set_xlabel('score') #x축 명
ax.set_ylabel('person Num') #y축 명
ax.set_xticks(np.linspace(0,100,25+1)) #x축 눈금그리기
ax.set_yticks(np.arange(0, freq.max()+1)) #y축 눈금
plt.show()

#4-3) 상자그림 그래프 - 데이터의 분포와 이상값을 시각적으로 파악
fig = plt.figure(figsize=(5,6)) #figsiz 그래프 크기
ax = fig.add_subplot(111)
ax.boxplot(eng_scores, labels=['english'])
plt.show()
#---------------------------------------------------------------------
#matplotlib
#한창에 그래프 2개를 같이 띄운다.
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1) #2행, 1열 창에 (1행)
ax2 = fig.add_subplot(2,1,2) #2행, 1열 창에 (2행)
x = range(0,100)
y = [i*i for i in x]
ax1.plot(x,y) #꺽은선 그래프
ax2.bar(x,y) #막대 그래프
plt.show()

#----------------------------------------------------------------------
fig = plt.figure()
axs = fig.subplots(nrows=1, ncols=2, sharex=False, sharey=True)
#1행,2열로 그래프 각각 출력
#sharex : x축 공유, sharey : y축 공유
x = range(0,100)
y = [i*i for i in x]
axs[0].plot(x,y) #꺽은선 그래프
axs[1].bar(x,y) #막대 그래프
plt.show()