# 11_비지도학습_K평균.py
#K : 클러스터(군집) 개수, 개발자가 설정해줘야하는 하이퍼파라미터
#평균값이 클러스터(군집)의 중심에 위치하기 때문에 클러스터 중심이라고 부름
#K-means 알고리즘은 K개 개수만큼의 클러스터 중심(데이터 평균값)을 찾아준다.

#알고리즘 원리
#(0) K값 선정(군집개수)
#(1) 무작위로 K개의 클러스터 중심을 정한다.
#(2) 각 샘플에서 가장 가까운 클러스터 중심을 찾아 클러스터의 샘플로 지정한다.
#(3) 클러스터에 속한 샘플의 평균값으로 클러스터 중심을 변경한다.
#(4) 클러스터의 중심의 변화가 없을 때까지 (2)번으로 돌아가 반복한다.

from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
#1) 데이터 로드
fruits = np.load('./데이터분석/data/fruits_300.npy')
fruits.shape #(300, 100, 100)
#2) 데이터 전처리
fruits_2d = fruits.reshape(-1,100*100)
fruits_2d.shape #(300, 10000)
#3) 분석모델구축
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3, random_state=32)
model.fit(fruits_2d)
#4) 결과분석
#비지도학습 성능지표가 따로 없다
model.labels_ #각 사진별 클러스터 라벨 생성
#라벨값 0,1,2
np.unique(model.labels_, return_counts=True) #라벨별 개수
#(array([0, 1, 2]), array([111,  98,  91], dtype=int64))

#라벨별 시각화 함수 구현
import matplotlib.pyplot as plt
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

#라벨별 시각화
draw_fruits(fruits[model.labels_==0]) #label 0 파인애플 그룹
draw_fruits(fruits[model.labels_==1]) #label 1 바나나 그룹
draw_fruits(fruits[model.labels_==2]) #label 2 사과 그룹

#클러스터 중심 시각화
model.cluster_centers_.shape #(3, 10000)
#3개의 중심, 중심별 각 픽셀값들의 평균값 (총픽셀 개수 10000)
draw_fruits(model.cluster_centers_.reshape(-1,100,100),fig_size_ratio=3)
#-------------------------------------------------------------------------
#12월21일 복습
#1. 텍스트 분석
#  1-1)감정분석모델 : 임의의 텍스트가 긍정/부정인 판별하는 모델
#  1-2)토픽모델링 LDA : 텍스트의 토픽을 추출하는 모델

# 2. 머신러닝
#  2-1) 지도학습 : 의사결정나무(설명가능한 모델, 모델에 기여하는 피쳐의 중요성정도 파악)
                #   앙상블학습모델의 기초가 됨.
#  2-2) 비지도학습 : KMeans 군집화 알고리즘 ex) 사과, 파인애플,바나나 사진 데이터 군집

#주가 예측 모델
#주식 종목 미리 생각하기
#LG그룹의 주가 예측 모델

#----------------------------------------------------------------------------
#실제 비지도학습에서는 K값을 알 수가 없다.
#최적의 K값을 찾기
#1) 엘보우 방법
#inertial : 클러스터 중심과 샘플데이터 사이 거리의 제곱합(오차 제곱의 합 비슷)
#클러스터에 속한 샘플이 얼마나 중심에 가깝게 모여 있는지 정도로 생각할 수 있다.
#클러스터가 늘어나면(K값이 커지면, 중심의 개수가 많아지면) 이너셔가 줄어든다.
#클러스터의 개수를 늘리면서 이너셔가 감소하는 속도가 꺽이는 지점
#이 지점을 엘보우 지점. 이 지점의 K값을 최적의 K값으로 선정

inertial = []
for k in range(2,7):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(fruits_2d)
    inertial.append(model.inertia_)
plt.plot(range(2,7), inertial)
plt.xlabel("K")
plt.ylabel('inertial')
plt.show()

#최적의 K값은 3으로 선정

#2)실루엣 분석
#클러스터 내에