#10_비지도학습_기초.py
#비지도 학습 : 군집, 차원축소
#군집 대표적인 알고리즘 : K-means 군집화
#군집 : 비슷한 샘플끼리 그룹으로 모르는 작업
#비슷한 유형의 데이터를 찾아준다.

#ex) 과일 3종류 흑백사진 분류
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
fruits = np.load('./데이터분석/data/fruits_300.npy')
fruits.shape #(300, 100, 100) #300장의 100*100픽셀의 사진
fruits[0].shape

plt.imshow(fruits[0], cmap='gray')
plt.show()

#원본이미지
plt.imshow(fruits[0], cmap='gray_r')
plt.show()

fig, axs = plt.subplots(1,3)
axs[0].imshow(fruits[0], cmap='gray_r')
axs[1].imshow(fruits[100], cmap='gray_r')
axs[2].imshow(fruits[200], cmap='gray_r')
plt.show()
#사과100, 파인애플100, 바나나100

#픽셀별 평균값
#100*100 행렬 1차원 배열로 변경
#계산하기 좋음
apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)
apple.shape#(100, 10000)
apple.mean(axis=1) #사과 사진별 픽셀값 평균
apple.mean(axis=1).shape #(100,)
#1,2,3,4,5,6 #사진1 axis=1  ->
#2,3,4,1,2,1 #사진2 axis=1  ->
#2,3,4,1,2,1 #사진3
#aixs=0
apple.mean(axis=0) #사과 픽셀별 100장의 평균값
apple.mean(axis=0).shape  #(10000,)

#히스토그램 - 사진별 픽셀 평균값
plt.hist(np.mean(apple, axis=1), alpha=0.8)
plt.hist(np.mean(pineapple, axis=1), alpha=0.8)
plt.hist(np.mean(banana, axis=1), alpha=0.8)
plt.legend(['apple','pineapple', 'banana'])
plt.show()
np.mean(apple, axis=1)

#막대그래프 - 픽셀별 평균값 
fig, axs = plt.subplots(1,3, figsize=(20,5))
axs[0].bar(range(10000), np.mean(apple, axis=0))
axs[1].bar(range(10000), np.mean(pineapple, axis=0))
axs[2].bar(range(10000), np.mean(banana, axis=0))
plt.show()

#이미지화 - 픽셀별 평균값
dataList = [apple, pineapple, banana]
meanList = [np.mean(x, axis=0).reshape(100,100) for x in dataList]
fig,axs = plt.subplots(1,3, figsize=(20,5))
for i,mean in enumerate(meanList) :
    axs[i].imshow(mean, cmap='gray_r')
plt.show()
#---------------------------------------------------------
#3)분석모델 구축 : 사진분류
#방법 : 300장의 사진중 픽셀의 평균값과 가장 가까운 사진들 찾기
#사과 이미지 분류
#전체 300장의 사진에서 픽셀별 평균 사과이미지 데이터를 빼서
#오차의 평균값이 제일 작은 사진 100장을 사과이미지로 판단
#              abs() : 절댓값, 오차의 절댓값
apple_diff = np.abs(fruits-meanList[2])
apple_diff.shape #(300, 100, 100)

#각 사진 픽셀들의 오차 평균 값
apple_diff_mean = np.mean(apple_diff, axis=(1,2)) #각사진별 100*100픽셀들의 평균
apple_diff_mean.shape #(300,)
apple_diff_mean

#오차가 가장 작은 100개의 데이터 출력
#argsort는 크기 순으로 정렬해서 그 데이터의 인덱스를 리턴
apple_index = np.argsort(apple_diff_mean)[:100]
fig, axs = plt.subplots(10,10, figsize=(10,10))
for i in range(10) :
    for j in range(10) :
        axs[i,j].imshow(fruits[apple_index[i*10+j]], cmap='gray_r')
        axs[i,j].axis('off')
plt.show()

#-------------------------------------------------------------------
#위와 같은 경우에는 타겟을 알고 있어서 각 타겟별 픽셀들들의 평균값을 얻었다.
#실제 비지도학습에서는 타깃을 모르기 때문에 샘플들의 평균값을 미리 구할 수 없다.
#타겟값을 모르면서 세 과일의 평균값을 찾는 방법
#K-Means 알고리즘
