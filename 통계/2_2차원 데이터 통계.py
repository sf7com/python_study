#1. 두 데이터 사이의 관계를 나타내는 지표 - 공분산, 상관계수 
#2. 2차원 데이터 시각화 - 산점도와 회귀선
#3. 앤스컴의 예 - 수치지표는 많은 정보를 잃어버릴 수 있다. (데이터 시각화 중요성)

import numpy as np
import pandas as pd

df = pd.read_csv('./통계/data/ch2_scores_em.csv', index_col='student number')
df.head()
#                 english  mathematics
# student number
# 1                    42           65
# 2                    69           80
# 3                    56           63
# 4                    41           63
# 5                    57           76

#10개의 점수 얻기
en_scores = np.array(df['english'][:10])
ma_scores = np.array(df['mathematics'][:10])
scores_df = df.head(10)
scores_df

#1.두 데이터 사이의 관계를 나타내는 지표
#양의 상관관계 ex) 영어 점수가 높은 학생이 수학 점수도 높은 경향 여부
#음의 상관관계 ex) 영어 점수가 높은 학생이 수학 점수가 낮은 경향 여부
#무 상관관계   ex) 영어 점수와 수학점수가 직접적으로 영향을 미치지 않는 경우

#(1) 공분산 : 각 데이터 편차들의 곱의 평균값
#Cov(X,Y) = E((X-u)*(Y-v)) 
# X는 수학점수, Y는 영어점수
# u는 X의 평균값, v는 Y의 평균값
# 공분산값이 양의 값이면 양의 상관관계
# 공분산값이 음의 값이면 음의 상관관계
# 공분산값이 0에 가까우면 무상관관계

summary_df = scores_df.copy()
summary_df['eng_dev'] = summary_df['english'] - summary_df['english'].mean()
summary_df['ma_dev'] = summary_df['mathematics'] - summary_df['mathematics'].mean()
summary_df['product_dev'] = summary_df['eng_dev']*summary_df['ma_dev']
summary_df

#공분산값
summary_df['product_dev'].mean() #62.8
np.cov(en_scores, ma_scores, ddof=0) #ddof 자유도의 감소값
#      영어점수와 영어점수의 공분산, 영어점수와 수학점수 공분산
#      수학점수와 영어점수의 공분산, 수학점수와 수학점수의 공분산
#array([[86.  , 62.8 ],
summary_df[['english','mathematics']].cov(ddof=0)
#             english  mathematics
# english         86.0        62.80
# mathematics     62.8        68.44
summary_df[['english','mathematics']].cov() 
#표본 데이터의 공분산값 구하는 것과, 모집단의 공분산값을 구하는게 약간 다르다
# 아래는 표본에 대한 공분산 값을 구한 것(표본 데이터는 자유도값이 1 감소)
#평균을 구할때 데이터의 개수가 -1 되서 평균값을 구한다.
#                english  mathematics
# english      95.555556    69.777778
# mathematics  69.777778    76.044444

#상관계수(피어슨 상관계수)
#공분산의 문제점 : X,Y의 단위에 영향을 받아서 수치값을 해석하기 어렵다.
#단위에 의존하지 않는 지표로 상관계수가 나옴
#R=Cov(X,Y)/(Std(X)*Std(Y))
#-1 ~ 1 사이의 값을 갖는다.
#양의 상관관계 : 1에 가까움
#음의 상관관계 : -1에 가까움
#무 상관관계 : 0에 가까움
summary_df[['english','mathematics']].corr()
#               english  mathematics
# english      1.000000     0.818569
# mathematics  0.818569     1.000000
np.corrcoef(en_scores, ma_scores)
#0.818 => 강한 양의 상관관계를 갖는다 해석

#매출액 1000, 2000, 3000
#광고비 100, 200, 300
#---------------------------
#매출액 100, 40, 40
#광고비 100, 200, 300
#0에 가까움 => 광고가 효과가 없다

#주식
#주식값 영향을 주는 변수
#거래량, MA(5일) 이동평균값, .....
#판단 기준
#-------------------------------------------------------------
#2. 2차원 데이터의 시각화
#(1) 산점도 : 점으로 그래프 시각화
import matplotlib.pyplot as plt
plt.scatter(scores_df['english'], scores_df['mathematics'])
plt.xlabel("Eng score")
plt.ylabel("Math score")

for i, data in scores_df.iterrows() :
    #데이터 프레임 한 행씩 가져옴, 인덱스와 데이터
    plt.text(data['english']+0.3, data['mathematics'], i, fontsize=10)
plt.show()
#(2) 회귀 직선 - 두 데이터 사이의 관게를 더욱 잘 나태는 직선
poly_fit = np.polyfit(en_scores, ma_scores, 1) 
#1차식 : Y=aX+b
poly_fit #array([ 0.73023256, 31.2372093 ]) 첫번째값이 a, 두번째 값 b
#수학점수=0.73*영어점수 + 31.23

#위 1차식의 함수반환
poly_1d = np.poly1d(poly_fit)
poly_1d(0) #수학점수=0.73*영어점수 + 31.23, 영어점수=0
poly_1d(30) #수학점수=0.73*영어점수 + 31.23, 영어점수=30

#위 직선의 데이터 얻기
#최솟값과 최댓값사이의 데이터를 균등하게 50개 데이터 얻는다
xs = np.linspace(en_scores.min(), en_scores.max()) 
xs
xs.shape #(50,)
ys = poly_1d(xs)
ys #xs값에 따른 y값

plt.scatter(scores_df['english'], scores_df['mathematics'])
plt.xlabel("Eng score")
plt.ylabel("Math score")
for i, data in scores_df.iterrows() :
    #데이터 프레임 한 행씩 가져옴, 인덱스와 데이터
    plt.text(data['english']+0.3, data['mathematics'], i, fontsize=10)
plt.plot(xs, ys, color='gray', 
        label=f'Y={poly_fit[0]:.2f}X+{poly_fit[1]:.2f}')
plt.legend() #범례 표시
plt.show()

#(3) 히트맵 : 히스토그램의 2차원 버전, 색을 이용하여 시각화
scores_df.describe()
#          english  mathematics
# count  10.000000    10.000000
# mean   55.000000    71.400000
# std     9.775252     8.720347
# min    41.000000    60.000000
# 25%    48.250000    63.500000
# 50%    56.500000    71.000000
# 75%    63.250000    79.500000
# max    69.000000    82.000000
c = plt.hist2d(scores_df['english'], scores_df['mathematics'],
            bins=[6, 5], range=[(40, 70), (60, 85)], cmap='Reds') 
#계급폭 5, 영어의 계급수 6, 수학 계급 5
plt.xlabel("Eng Score")
plt.ylabel("Math Score")
plt.xticks(c[1]) #x축 눈금
plt.yticks(c[2]) #y축 눈금
plt.colorbar(c[3]) #컬러바 표시
plt.show()

#---------------------------------------------------------------------------
#3.앤스컴의 예 - 데이터 분석시에 가능하면 시각화를 해야 좋다.
#총 4개의 2차원 데이터
anscom_data = np.load('./통계/data/ch3_anscombe.npy')
anscom_data
anscom_data.shape #(4, 11, 2)
#                  4종류, 11행, 2열 (2차원 데이터 11행)
#앤스컴 데이터 -> 데이터 프레임
stats_df = pd.DataFrame(index=['X_mean', 'X_var', 'Y_mean','Y_var',
                "X&Y_Corr", "X&Y_reg_line"])
stats_df
for i, data in enumerate(anscom_data) :
    dataX = data[:, 0]
    dataY = data[:, 1]
    poly_fit = np.polyfit(dataX, dataY, 1) #회귀선 Y=aX+b 값 구하기
    stats_df[f'data{i+1}'] = [np.mean(dataX), np.var(dataX),
        np.mean(dataY), np.var(dataY), np.corrcoef(dataX, dataY)[0,1],
        f'Y={poly_fit[0]:.2f}X+{poly_fit[1]:.2f}']
stats_df
#                      data1         data2         data3         data4
# X_mean                 9.0           9.0           9.0           9.0
# X_var                 10.0          10.0          10.0          10.0
# Y_mean            7.500909      7.500909           7.5      7.500909
# Y_var             3.752063       3.75239      3.747836      3.748408
# X&Y_Corr          0.816421      0.816237      0.816287      0.816521
# X&Y_reg_line  Y=0.50X+3.00  Y=0.50X+3.00  Y=0.50X+3.00  Y=0.50X+3.00

#2X2 그래프 그리기
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10),
    sharex=True, sharey=True) #축 공유
xs = np.linspace(0, 30, 100)
for i, data in enumerate(anscom_data) :
    dataX = data[:, 0]
    dataY = data[:, 1]
    poly_fit = np.polyfit(dataX, dataY, 1)
    poly_1d = np.poly1d(poly_fit)
    ys = poly_1d(xs)
    #그래프 그리기 영역
    ax = axes[i//2, i%2] #i=0 (0,0), i=1 (0,1), i=2 (1,0), i=3 (1,1)
    ax.set_xlim([4,20]) #x축 영역 제한
    ax.set_ylim([3,13]) #y축 영역 제한
    #타이틀
    ax.set_title(f'data{i+1}')
    ax.scatter(dataX, dataY)
    ax.plot(xs, ys, color='gray')

plt.tight_layout() #그래프 사이 간격 좁히기
plt.show()



