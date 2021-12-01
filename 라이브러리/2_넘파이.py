#Numpy(Numerical Python)는 행렬 연산을 위한 핵심 라이브러리.
#데이터 분석에서 중요한 라이브러리

import numpy as np
#(1) Array(행렬 또는 벡터) 정의 및 사용
#행렬 : 행과 열 (2차원 리스트 이상)
#벡터 : 단일한 열 (1차원 리스트)
data = [1,2,3,4,5]
arr1 = np.array(data)
arr1 #array([1, 2, 3, 4, 5])
arr1.shape #(5,) arr1의 크기
arr1.dtype #dtype('int32')

data2 = [1,1.5,3.14]
arr2 = np.array(data2)
arr2 #array([1.  , 1.5 , 3.14])
arr2.shape #(3,) arr1의 크기
arr2.dtype #dtype('float64')

arr3 = np.array([[1,2,3],[4,5,6]])
arr3
#array([[1, 2, 3],
#       [4, 5, 6]])
arr3.shape #(2,3) 2행,3열

#(2) numpy의 기본적인 함수
arr = np.zeros(10) #0값으로 인수 크기만큼 채워준다.
arr #array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
arr = np.zeros((3,5)) #0값으로 인수 크기만큼 채워준다.
arr
#array([[0., 0., 0., 0., 0.],
#       [0., 0., 0., 0., 0.],
#       [0., 0., 0., 0., 0.]])
arr = np.ones(10) #0값으로 인수 크기만큼 채워준다.
arr #array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
arr = np.ones((3,5)) #0값으로 인수 크기만큼 채워준다.
arr
#array([[1., 1., 1., 1., 1.],
#       [1., 1., 1., 1., 1.],
#       [1., 1., 1., 1., 1.]])
arr = np.arange(10) #0~9까지 1씩 증가해서 array를 만듬
arr #array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
arr = np.arange(3,10) #3~9까지 1씩 증가해서 array를 만듬
arr #array([3, 4, 5, 6, 7, 8, 9])
arr = np.arange(3,10,0.1) #3~9까지 0.1씩 증가해서 array를 만듬
arr #일반 range는 증감값으로 실수를 넣으면 안된다.
arr = np.linspace(0,1,5) #시작 0, 끝1, 요소 5개 균등한 간격
arr #array([0.  , 0.25, 0.5 , 0.75, 1.  ])

arr = np.random.randint(5,10,size=10)
arr #5이상 10미만의 랜덤한 수 size 크기만큼 array 만든다.
arr = np.random.randint(1,10,size=(3,3))
arr #1이상 10미만의 랜덤한 수 size 크기만큼 array 만든다.

#(3) Array의 인덱싱 (리스트의 인덱스 참조 하는 방식과 유사)
arr = np.arange(10)
arr[0] # 0
arr[1:3] # 인덱스 1이상 3미만 array([1, 2])
arr[::2] # 처음부터 끝까지 인덱스 2씩 증가 array([0, 2, 4, 6, 8])
arr[::-1] # 거꾸로 가져오기 array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

#(4) Array 연산
#리스트의 연산과 다름
arr1 = np.array([[1,2,3],[4,5,6]])
arr2 = np.array([[10,11,12],[13,14,15]])
arr1+10 #array의 모든 요소에 10이 더해짐
arr2*2 #array의 모든 요소에 2가 곱해짐
arr1/3 #array의 모든 요소에 3이 나눠짐
#array는 사칙연산이 모든 요소에 적용이 됨
arr1**2 #제곱 연산도 가능

#Array간의 사칙연산
#행렬의 크기가 같은 경우 같은 자리의 요소간의 사칙연산이 됨.
arr1+arr2
arr1*arr2
arr1/arr2
arr1-arr2

#Array의 브로드캐스팅
#행의 크기는 다르나, 열의 개수가 같은 경우
#각 행마다 같은 자리의 요소 끼리 사칙연산 수행
arr1 = np.array([[1,2,3], #사칙연산
                [4,5,6]])
arr3 = np.array([10,11,12])
arr1+arr3
arr1-arr3
arr1*arr3
arr1/arr3

#(5) Array 인덱싱 마스크
names = np.array(['김하나','김하나','김둘','김셋'])
names[names=='김하나']

#True인 것만 리턴
names=='김하나' #array([ True,  True, False, False])

#randn() 기댓값 0, 표준편차 1인 정규분포에서 랜덤한 수를 리턴
data = np.random.randn(10)
data
data[data>0]
data[data<0]

data>0

#(6) Numpy의 수치계산 함수
arr1 = np.random.randn(5,3)
np.abs(arr1) #모든 요소를 절댓값(양수)로 만든다.

np.sqrt(arr1) #모든 요소에 루트를 취한다
arr1**(1/2)

np.square(arr1) #모든 요소의 제곱
arr1**2

#각 성분에 자연로그, 상용로그를 취함
np.log(arr1) #자연로그
np.log10(arr1) #밑이10인 로그
np.log2(arr1) #밑이2인 로그

#각 성분의 부호 sign()
#올림,내림 ceil(), floor()
#삼각함수 sin(), cos() 등

#통계함수
arr1 = np.array([[1,2,3],[4,5,6]])
np.sum(arr1) #모든 요소의 합
np.sum(arr1,axis=1) #열간의 합
np.sum(arr1,axis=0) #행간의 합

np.mean(arr1) #모든 요소의 평균
np.mean(arr1,axis=1) #열간의 평균
np.mean(arr1,axis=0) #행간의 평균

np.var(arr1) #모든 요소의 분산
np.var(arr1,axis=1) #열간의 분산
np.var(arr1,axis=0) #행간의 분산

np.str(arr1) #모든 요소의 표준편차
np.std(arr1,axis=1) #열간의 표준편차
np.std(arr1,axis=0) #행간의 표준편차

np.max(arr1) #모든 요소의 최대값
np.max(arr1,axis=1) #열간의 최대값
np.max(arr1,axis=0) #행간의 최대값
#max를 min으로 바꾸면 최소값도 얻을 수 있다.

#정렬
arr1 = np.array([[10,2,14],[3,4,1]])
np.sort(arr1) #열간 정렬, 오름차순
np.sort(arr1, axis=1) #열간 정렬, 오름차순
np.sort(arr1, axis=0) #행간 정렬, 오름차순

#내림차순
#                        행인덱스, 열인덱스
arr1 = np.sort(arr1, axis=1)[:, ::-1] #열간 내림차순
arr1
arr1 = np.array([[10,2,14],[3,4,1]])
arr1 = np.sort(arr1, axis=1)[::-1, :] #행간 내림차순
arr1
