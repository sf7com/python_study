import pandas as pd
import matplotlib.pyplot as plt
#pip install pandas
#pip install matplotlib

path = './데이터수집/API/'
ko_df = pd.read_json(path+"한국_국민해외관광객_201001_202012.json")
jp_df = pd.read_json(path+"일본_방한외래관광객_201001_202012.json")
cn_df = pd.read_json(path+"중국_방한외래관광객_201001_202012.json")
am_df = pd.read_json(path+"미국_방한외래관광객_201001_202012.json")
ko_df.info()
jp_df.info()
cn_df.info()
am_df.info()
ko_df.head()

#년도별 집계
# 2010~202년도 까지 년도별 그래프 시각화 _ 꺽은선 그래프
#년도 열 추가 
ko_df['yyyy'] = ko_df['날짜'].astype('str').str[0:4]
jp_df['yyyy'] = ko_df['날짜'].astype('str').str[0:4]
cn_df['yyyy'] = ko_df['날짜'].astype('str').str[0:4]
am_df['yyyy'] = ko_df['날짜'].astype('str').str[0:4]
ko_df.head()

#데이터프레임의 Group by _ SQL GROUP BY와 유사함
ko_yearGroup = ko_df.groupby('yyyy')['관광객수'].sum()
jp_yearGroup = jp_df.groupby('yyyy')['관광객수'].sum()
cn_yearGroup = cn_df.groupby('yyyy')['관광객수'].sum()
am_yearGroup = am_df.groupby('yyyy')['관광객수'].sum()
ko_yearGroup

yearList = ko_yearGroup.index
yearList

#폰트 변경(한글 표시)
plt.rc('font', family='malgun gothic')
#        X축값    ,y축값,        색깔 또는 표시방법, label
plt.plot(yearList, ko_yearGroup, 'r--', label='한국인 출입국자 수')
plt.plot(yearList, jp_yearGroup, 'b', label='일본인 입국자 수')
plt.plot(yearList, cn_yearGroup, 'g', label='중국인 입국자 수')
plt.plot(yearList, am_yearGroup, 'c', label='미국인 입국자 수')
plt.title('년도별 입국/출국자 수')
plt.xlabel("년도")
plt.ylabel('명')
plt.legend() #범례 표시
plt.grid(True) #격자 표시
plt.show()
