#데이터 셋 : 영국 온라인 소매 플랫폼 2010.12.01 ~ 2011.12.09 거래 데이터
#K-Means 군집 알고리즘으로 소비자 유형 분석
#-------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#1) 데이터 로드
df = pd.read_csv('./데이터분석/data/Online Retail.csv')
df.head()
df.info()

#2) 데이터 탐색 및 전처리
#2-1) 결측치 제거
df.isnull().sum()
# InvoiceNo           0
# StockCode           0
# Description      1454
# Quantity            0
# InvoiceDate         0
# UnitPrice           0
# CustomerID     135080
# Country             0
df = df[df['CustomerID'].notnull()]
df.isnull().sum()
df[df['Quantity'] <= 0] #주문량이 0보다 같거나 작은 데이터 분석할 가치가 없는 데이터
df = df[df['Quantity'] > 0]
#고객 ID 데이터 타입 변경
df['CustomerID'] = df['CustomerID'].astype(int)
df.info()

#2-2) 중복데이터 제거
temp = df[df.duplicated(keep=False)] #모든 열의 값이 같은 행들만 중복 체크
temp.info()
temp[temp['InvoiceNo']=='536409']
df.drop_duplicates(inplace=True)
df[df.duplicated()]

#2-3) 데이터 탐색 - 제품수, 거래건수, 고객수, 나라수 확인
df['StockCode'].value_counts() #상품별 주문 개수
len(df['StockCode'].value_counts()) #주문 전체 상품수 3665 
len(df['InvoiceNo'].value_counts()) #전체 거래수 18536
len(df['CustomerID'].value_counts()) #전체 고객수 4339
#나라별 거래수
df.groupby('Country')['InvoiceNo'].unique()
df.groupby('Country')['InvoiceNo'].unique().apply(lambda x:len(x))
df.groupby('Country')['InvoiceNo'].unique().apply(lambda x:len(x)).\
                                sort_values(ascending=False)
# United Kingdom          16649
# Germany                   457
# France                    389
# EIRE                      260
# Belgium                    98
# Netherlands                95

#2-4) 특성공학 : 분석에 필요한 데이터를 생성
#고개별 군집화: 고객별정보(주문량, 주문금액합계, 마지막 주문날짜, 주문별 전체 구매가격)
#주문별 전체 구매가격
df['SaleAmount'] = df['UnitPrice'] * df['Quantity']
df['SaleAmount'].head()
#고객별 정보 아이디별 Groupby
aggs = {'InvoiceNo' : 'count', 'SaleAmount':'sum', 'InvoiceDate':'max'}
customer_df = df.groupby('CustomerID').agg(aggs)
customer_df.head()
df.info()

#고객의 마지막 주문 후 경과일 : 현재일(2011.12.10)-마지막 주문일
customer_df.info() # 2   InvoiceDate  4339 non-null   object
import datetime
customer_df['InvoiceDate'] =  pd.to_datetime(customer_df['InvoiceDate'])
customer_df.info() # 2   InvoiceDate  4339 non-null   datetime64[ns]
customer_df['InvoiceDate'] = datetime.datetime(2011,12,10) - customer_df['InvoiceDate']
customer_df['InvoiceDate'].head()

#칼럼명 수정
customer_df.rename(columns={'InvoiceNo':'Freq', 'InvoiceDate':"ElapsedDays"},
                    inplace=True)
customer_df.head() #Freq(주문갯수), ElapsedDays(경과일customer_df['Freq_log'] = np.log1p(customer_df['Freq'])
customer_df['Freq_log'] = np.log1p(customer_df['Freq'])
customer_df['SaleAmount_log'] = np.log1p(customer_df['SaleAmount'])
customer_df['ElapsedDays'] #18280   277 days 14:08:00
customer_df['ElapsedDays'] = customer_df['ElapsedDays'].apply(lambda x:x.days)
customer_df['ElapsedDays'].head()
customer_df['ElapsedDays_log'] = np.log1p(customer_df['ElapsedDays'])
customer_df.head()



#2-5) 데이터 탐색 - 데이터 분포도 시각화 (박스 플롯)
plt.boxplot([customer_df['Freq'], customer_df['SaleAmount'], 
        customer_df['ElapsedDays']], sym='bo')
plt.xticks([1,2,3], customer_df.columns)
plt.show()
#데이터 값의 왜곡을 줄이기 위한 로그 함수 분포 조정
#log1p => log(1+x) : log(x) x의값이 0이면 -무한대 수렴해서, log(x+1) x의값이 0이면 0
plt.boxplot([customer_df['Freq_log'], customer_df['SaleAmount_log'], 
        customer_df['ElapsedDays_log']], sym='bo')
plt.xticks([1,2,3], customer_df.columns[3:])
plt.show()




#2-6) 데이터 정규화
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = customer_df.iloc[:,3:] #로그분포로 조정된 데이터로 군집화
# ss.fit(X)
# X_scaled = ss.transform(X)
X_scaled = ss.fit_transform(X) #fit과 transform 같이 사용할 수 있음.
X_scaled.shape #(4339, 3) 총 4339의 행, 3개열

#3) 모델구축 및 훈련
from sklearn.cluster import KMeans
#최적의 값을 찾기
#3-1) 최적의 K값 찾는 엘보우
inertial = []
for i in range(1,11) :
    model = KMeans(n_clusters=i, random_state=32)
    model.fit(X_scaled)
    inertial.append(model.inertia_)
plt.plot(range(1,11), inertial, marker='o')
plt.xlabel("Number of Cluster")
plt.ylabel("inertia")
plt.show()
#최적의 K값은 2~4

#3-2) 실루엣 방법
from matplotlib import cm
from sklearn.metrics import silhouette_score, silhouette_samples
def silhouetteViz(n_cluster, X_features): 
    
    kmeans = KMeans(n_clusters=n_cluster, random_state=0)
    Y_labels = kmeans.fit_predict(X_features)
    
    silhouette_values = silhouette_samples(X_features, Y_labels, metric='euclidean')

    y_ax_lower, y_ax_upper = 0, 0
    y_ticks = []

    for c in range(n_cluster):
        c_silhouettes = silhouette_values[Y_labels == c]
        c_silhouettes.sort()
        y_ax_upper += len(c_silhouettes)
        color = cm.jet(float(c) / n_cluster)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouettes,
                 height=1.0, edgecolor='none', color=color)
        y_ticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouettes)
    
    silhouette_avg = np.mean(silhouette_values)
    plt.axvline(silhouette_avg, color='red', linestyle='--')
    plt.title('Number of Cluster : '+ str(n_cluster)+'\n' \
              + 'Silhouette Score : '+ str(round(silhouette_avg,3)))
    plt.yticks(y_ticks, range(n_cluster))   
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.tight_layout()
    plt.show()

#- 클러스터 수에 따른 클러스터 데이터 분포의 시각화 함수 정의
def clusterScatter(n_cluster, X_features): 
    c_colors = []
    kmeans = KMeans(n_clusters=n_cluster, random_state=0)
    Y_labels = kmeans.fit_predict(X_features)

    for i in range(n_cluster):
        c_color = cm.jet(float(i) / n_cluster) #클러스터의 색상 설정
        c_colors.append(c_color)
        #클러스터의 데이터 분포를 동그라미로 시각화
        plt.scatter(X_features[Y_labels == i,0], X_features[Y_labels == i,1],
                     marker='o', color=c_color, edgecolor='black', s=50, 
                     label='cluster '+ str(i))       
    
    #각 클러스터의 중심점을 삼각형으로 표시
    for i in range(n_cluster):
        plt.scatter(kmeans.cluster_centers_[i,0], kmeans.cluster_centers_[i,1], 
                    marker='^', color=c_colors[i], edgecolor='w', s=200)
    
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

silhouetteViz(2, X_scaled) #0.399
silhouetteViz(3, X_scaled) #0.306
silhouetteViz(4, X_scaled) #0.307
silhouetteViz(5, X_scaled) #0.276
silhouetteViz(6, X_scaled) #0.274
#최적의 값 2,4
#최종적으로 소비자유형 군집을 4로 지정
#군집 시각화 함수
clusterScatter(2, X_scaled)
clusterScatter(3, X_scaled)
clusterScatter(4, X_scaled)

model = KMeans(n_clusters=4, random_state=42)
model.fit(X_scaled)

#4) 결과분석
labels = model.predict(X_scaled)
customer_df['label'] = labels
customer_df.head() #고객 아이디별로 클러스터 번호 생성

#4-1) 클러스터별 고객수
customer_df['size'] = 1
customer_df.groupby('label')['size'].count()
# label
# 0     857
# 1    1239
# 2     854
# 3    1389

#4-2) 클러스터별 특징파악
customer_cluster_df = customer_df.drop(customer_df.columns[3:6],axis=1) #로그컬럼 삭제
customer_cluster_df.head()
#주문 1회당 평균 구매 금액
customer_cluster_df['SaleAmountAvg'] = customer_cluster_df['SaleAmount']/customer_cluster_df['Freq']
customer_cluster_df.head()
#클러스터별로 각 컬럼의 평균값
customer_cluster_df.drop('size', axis=1).groupby('label').mean()
#              Freq   SaleAmount  ElapsedDays  SaleAmountAvg
# label
# 0      281.416569  7090.221995    12.310385      97.833556
# 1       79.163842  1511.061552    94.929782     100.899822
# 2       38.329040   616.100117    18.896956      32.650992
# 3       14.932325   297.000634   183.629950      43.025347
#0번 그룹 특징 : 구매횟수가 제일 큰 그룹, 주문 총액도 가장 크다, 최근 주문한 고객, 
#               1회 주문당 평균 97달러 정도 주문(영국 온라인 소매플랫폼 핵심 그룹)
#1번 그룹 특징 : 주문 횟수는 2위인 그룹, 주문 총액 2위, 3달 전쯤에 구매한 고객들
#               1회 주문당 구매 총액은 제일 큰 그룹
#2번 그룹 특징 : 최근에 구매이력이 있는 고객, 주문 총액과 주문 횟수는 작은편
                # (최근 가입한 고객일 확률이 높음)
#3번 그룹 특징 : 주문한지도 오래됐고, 주문량도 적고 
#               (어쩌다가 한번 이용한 고객, 다시 이용을 안하는 고객)
# label
# 0     857
# 1    1239
# 2     854
# 3    1389