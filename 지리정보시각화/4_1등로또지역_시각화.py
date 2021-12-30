from 데이터수집.WebCrawling import getRequestUrl
import pandas as pd
import urllib.request
import json

lotto = pd.read_csv('./지리정보시각화/data/lotto.csv', index_col=0)
lotto.head()
lotto.info()
addr = lotto.iloc[0]['address']
import re
lotto['address'] = lotto['address'].\
            apply(lambda x:re.split(r'[\,\(]',x)[0])
lotto
#로또 1등 매장의 위도, 경도 좌표 얻고 -> 시각화

#홀리스카페 매장 주소 -> 위도,경도 좌표얻기
def getGeoData(addr) :
    key='0F6FC5BB-2345-3D99-8A89-E5F520FC57EE'
    serviceURL = 'http://api.vworld.kr/req/address?'
    param = 'service=address&request=getcoord&version=2.0&crs=epsg:4326'
    param += f'&address={urllib.parse.quote(addr)}'
    param += f'&refine=true&simple=false&format=json&type=road&key={key}'
    url = serviceURL+param
    data = getRequestUrl(url)
    if data :
        return json.loads(data)
    else :
        None

addr = lotto['address']
addr
lotto['위도'] = 0
lotto['경도'] = 0
lotto.head()

for i, addr in enumerate(addr) :
    try :
        geoData = getGeoData(addr)
        xPos = geoData['response']['result']['point']['x'] #경도
        yPos = geoData['response']['result']['point']['y'] #위도
        print(xPos, yPos)
        lotto.loc[i, '경도'] = xPos
        lotto.loc[i, '위도'] = yPos
    except Exception as e:
        print(e)

#위도경도 데이터 저장
lotto.to_csv('./지리정보시각화/data/로또_위도경도.csv', index=False)

#-------------------------------------------------------
#시각화
import pandas as pd
import folium
#(1) 지도생성
m = folium.Map(location=[lotto.loc[1,'위도'], lotto.loc[1,'경도']],
        zoom_start=7)

#(2) 마커클러스터 생성
from folium.plugins import MarkerCluster
m_cluster = MarkerCluster().add_to(m)

for idx, data in lotto.iterrows() : 
    if data['위도'] == 0 :
        continue
    else :
        folium.Marker(
            location=[data['위도'], data['경도']],
            popup=data['name'],
            icon=folium.Icon(icon='star', color='blue')
        ).add_to(m_cluster)
m.save('./지리정보시각화/data/lotto_map.html')


