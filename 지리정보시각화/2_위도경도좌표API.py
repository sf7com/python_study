#Folium 위도, 경도 좌표 필요
#주소 -> 위도, 경로 바꾸는 실습
#인터넷검색, API를 통해 값 얻기

#vworld 사이트 API
from 데이터수집.WebCrawling import getRequestUrl
import pandas as pd
import urllib.request
import json

hollys = pd.read_csv('./데이터수집/정적웹크롤링/할리스카페매장2.csv')
hollys.head()
hollys.info()
addr = hollys.iloc[0]['주소']
addr 
#'서울특별시 종로구 통일로 230 (무악동, 경희궁롯데캐슬) 상가동  지하1층 117호'

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

addr = hollys['주소']
addr
hollys['위도'] = 0
hollys['경도'] = 0
hollys.head()

for i, addr in enumerate(addr) :
    try :
        geoData = getGeoData(addr)
        xPos = geoData['response']['result']['point']['x'] #경도
        yPos = geoData['response']['result']['point']['y'] #위도
        print(xPos, yPos)
        hollys.loc[i, '경도'] = xPos
        hollys.loc[i, '위도'] = yPos
    except Exception as e:
        print(e)

hollys.head()
hollys[hollys['위도']==0]['주소']
getGeoData('충청북도 청주시 서원구 1순환로 892')
getGeoData('충청북도 청주시 서원구 1순환로 892,1,2층(산남동)')
#주소를 미리 전처리 
# 충청북도 청주시 서원구 1순환로 892,1,2층(산남동)
#강원도 원주시 호저면 마근거리길 110 (원주(춘천방향)휴게소) 옥산리 215
import re
notFoundAddr = hollys[hollys['위도']==0]['주소']
notFoundAddr.iloc[0]
#충청북도 청주시 서원구 1순환로 892,1,2층(산남동)
#강원도 원주시 호저면 마근거리길 110 (원주(춘천방향)휴게소) 옥산리 215
re.split(r'[\,\(]', notFoundAddr.iloc[0])
#['충청북도 청주시 서원구 1순환로 892', '1', '2층', '산남동', '']


#apply는 한 행에대한 데이터를 처리 사용
notFoundAddr = notFoundAddr.apply(lambda addr : re.split(r'[\,\()]',addr)[0])
notFoundAddr = notFoundAddr.str.replace(".", "")
notFoundAddr

for idx, addr in notFoundAddr.iteritems() :
    try :
        geoData = getGeoData(addr)
        xPos = geoData['response']['result']['point']['x'] #경도
        yPos = geoData['response']['result']['point']['y'] #위도
        print(xPos, yPos)
        hollys.loc[idx, '경도'] = xPos
        hollys.loc[idx, '위도'] = yPos
    except Exception as e:
        print(e)

hollys[hollys['위도']==0]['주소'] #[35 rows x 8 columns]
getGeoData('울산광역시 울주군 삼동면 보은리 산193-1') #지번주소
#==> 직접검색을 해서 값을 수동으로 기입
#35.513199381436756, 129.12863729033953
#시간이 오래걸리니 생략

#위도경도 데이터 저장
hollys.to_csv('./지리정보시각화/data/홀리스카페_위도경도.csv', index=False)

