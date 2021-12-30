#13_비지도학습_차원축소
#차원축소 대표적인 알고리즘 : 주성분분석(PCA)
#차원 : 머신러닝에서 특성의 개수를 차원이라고 한다.
#10,000개의 특성 => 10,000 차원

#차원축소 : 데이터를 가장 잘 나타내는 일부 특성을 선택하여, 데이터의 크기를 줄이고
#          머신러닝 모델의 성능도 향상 시킬 수 있는 방법
#          줄어든 차원의 데이터로 다시 원본차원으로 데이터의 손실을 줄이면서 복원도 가능

#주성분 : 데이터의 분산이 큰 방향의 벡터(분산이 큰 벡터는 정보량이 크다, 데이터의 분포를 잘 표현할 수 있다.)
#주성분의 개수는 원본 특성의 개수만큼 찾을 수가 있다.     
#                                 
#주성분은 원본 데이터에서 가장 분산이 큰 것대로 벡터로 표현하여 나타낸 것.
#원본 데이터 셋에서 어떤 특징을 잡아 냈다고 생각하면 좋다.
#------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
#1) 데이터 로드
fruits = np.load('./데이터분석/data/fruits_300.npy')
fruits.shape #(300, 100, 100)
#2) 데이터 전처리
fruits_2d = fruits.reshape(-1,100*100)
fruits_2d.shape #(300, 10000)

def draw_fruits(arr, fig_size_ratio=1):
    n = len(arr)    # n은 샘플 개수입니다
    # 한 줄에 10개씩 이미지를 그립니다. 샘플 개수를 10으로 나누어 전체 행 개수를 계산합니다. 
    rows = int(np.ceil(n/10))
    # 행이 1개 이면 열 개수는 샘플 개수입니다. 그렇지 않으면 10개입니다.
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols, 
                            figsize=(cols*fig_size_ratio, rows*fig_size_ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:    # n 개까지만 그립니다.
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()

#3) 분석모델 구축
from sklearn.decomposition import PCA
pca = PCA(n_components=50) #n_components 주성분 개수
pca.fit(fruits_2d)
fruits_2d.shape #(300, 10000) #특성이 몇개? 10000
#최대 주성분의 개수 10000개
#10000개의 차원 => 50개의 차원으로 축소

pca.components_.shape #(50, 10000) #주성분 벡터
#총 50개의 주성분, 1개의 주성분 10000개의 차원(원본 차원)으로 표현
pca.components_[0] #0번째 주성분
#각 주성분(벡터)은 원본차원으로 표현한다.

draw_fruits(pca.components_.reshape(-1,100,100)) #주성분을 시각화

#차원 축소된 데이터
fruits_pca = pca.transform(fruits_2d)
fruits_pca.shape #(300, 50) 원본 데이터 (300,10000)
fruits_pca[0] #0번째 축소된 사진 데이터 50개의 차원

#원본 데이터로 복구
fruits_inverse = pca.inverse_transform(fruits_pca)
fruits_inverse.shape #(300, 10000)
draw_fruits(fruits_inverse.reshape(-1,100,100)[0:100]) #사과
draw_fruits(fruits_2d.reshape(-1,100,100)[0:100]) #사과

#설명된 분산 : 주성분이 원본데이터의 분산을 얼마나 잘 나타내는지 기록한 값
#첫번째 주성분이 설명된 분산값이 제일 크다. 그 다음은 두번째 주성분..
pca.explained_variance_ratio_.shape #(50,)
pca.explained_variance_ratio_
np.sum(pca.explained_variance_ratio_) #0.92152
#분산 비율을 모두 더하면 50개의 주성분으로 표현하고 있는 총 분산비율을 얻음
#92%넘는 분산 유지(원본 데이터의 92% 정보를 담고 있다.)
#=> 차원축소된 주성분데이터로 원본의 92% 복구가 가능하다.
#----------------------------------------------------------------------
#설명된 분산이 50%에 달하는 주성분을 찾도록 PCA모델 만들기
pca = PCA(n_components=0.5) #n_components 값에 0~1사이의 실수값을 넣으면 된다.(원본데이터의 50% 주성분을 담고 있는 데이터 )
pca.fit(fruits_2d)
pca.components_.shape #(2, 10000) #주성분이 2개, 각 주성분을 10000차원으로 표현

#차원축소된 데이터 얻기
fruits_pca = pca.transform(fruits_2d)
fruits_pca.shape #(300, 2)
fruits_pca[0] #0번째 차원축소된 사진 데이터

#설명된 분산의 값
pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_) #0.52298

#--------------------------------------------------------------
#차원축소된 데이터를 활용해서 다른 모델들의 학습에 사용
#K-means 알고리즘 군집화 하기
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
fruits_pca = pca.transform(fruits_2d)
fruits_pca.shape #(300, 2) #300개의 사진 데이터, 각 사진 마다 2개의 피쳐
km.fit(fruits_pca)
np.unique(km.labels_, return_counts=True) #(array([0, 1, 2]), array([110,  99,  91], dtype=int64))

draw_fruits(fruits[km.labels_==0]) #파인애플 그룹
draw_fruits(fruits[km.labels_==1]) #바나나 그룹
draw_fruits(fruits[km.labels_==2]) #사과 그룹

#차원축소된 데이터가 2차원이여서 시각화(클러스터별 시각화)
for label in range(0,3):
    data = fruits_pca[km.labels_==label]
    plt.scatter(data[:,0], data[:,1])
plt.legend(['pineapple', 'banana','apple'])
plt.show()

#원본 데이터로 복구
fruits_inverse = pca.inverse_transform(fruits_pca)
fruits_inverse.shape #(300, 10000)
draw_fruits(fruits_inverse.reshape(-1,100,100)[0:100]) #사과