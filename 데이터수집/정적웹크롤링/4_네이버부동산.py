from 데이터수집.WebCrawling import getRequestUrl
import json
import re
from bs4 import BeautifulSoup
import urllib.request
searchStr = '권선구'
url = f'https://m.land.naver.com/search/result/{urllib.parse.quote(searchStr)}'
html = getRequestUrl(url)
soup = BeautifulSoup(html, 'html.parser')
# with open('./데이터수집/정적웹크롤링/네이버부동산.html','w',
#             encoding='utf-8') as f :
#             f.write(soup.prettify())

#(1) 지역의 cortarNo(법정코드), z 얻기
scriptTags = soup.find_all('script', attrs={'type':"text/javascript"})
script = scriptTags[1]
m = re.search(r"cortarNo:\s*'(.+?)'", script.text)
cortarNo = m.group(1) #'4111313700'
cortarNo

m = re.search(r"z:\s*'(.+?)'", script.text)
z = m.group(1) #'14'
z

#(2) 클러스터 정보 얻기(컴플렉스들의 위치 정보), lgeo얻기
clusterUrl = f'https://m.land.naver.com/cluster/clusterList?view=atcl&cortarNo={cortarNo}' + \
            f'&rletTpCd=APT&tradTpCd=A1%3AB1%3AB2&z={z}&addon=COMPLEX'
clusterJson = json.loads(getRequestUrl(clusterUrl))
clusterJson
lgeoList = [data['lgeo'] for data in clusterJson['data']['COMPLEX']]
lgeoList

#(3) 각 Complex의 아파트 매매 정보 얻기
def getComplexInfo(lgeo, z, cortarNo) :
    complexUrl = f'https://m.land.naver.com/cluster/ajax/complexList?' + \
        f'lgeo={lgeo}&rletTpCd=APT&tradTpCd=A1%3AB1%3AB2&z={z}&cortarNo={cortarNo}'
    complexJson = json.loads(getRequestUrl(complexUrl))
    return complexJson['result']


itemList = []
for lgeo in lgeoList :
    complexInfo = getComplexInfo(lgeo, z, cortarNo)
    for item in complexInfo :
        name = item.get('hscpNm', '') #건물이름
        typeName = item.get('hscpTypeNm', '') #건물타입
        totDongCnt = item.get('totDongCnt', '') #아파트 동 수
        totHsehCnt = item.get('totHsehCnt', '') #아파트 호 수
        useAprvYmd = item.get('useAprvYmd', '') #준공일
        dealCnt = item.get('dealCnt', '') #매매건수
        leaseCnt = item.get('leaseCnt', '') #전세건수
        rentCnt = item.get('rentCnt', '') #월세건수
        dealPrcMin = item.get('dealPrcMin', '') #매매 최저가
        dealPrcMax = item.get('dealPrcMax', '') #매매 최고가
        leasePrcMin = item.get('leasePrcMin', '') #전세 최저가
        leasePrcMax = item.get('leasePrcMax', '') #전세 최고가
        dealPrcMin = re.sub(r'[^가-힣\d]+', "", dealPrcMin)
        dealPrcMax = re.sub(r'[^가-힣\d]+', "", dealPrcMax)
        leasePrcMin = re.sub(r'[^가-힣\d]+', "", leasePrcMin)
        leasePrcMax = re.sub(r'[^가-힣\d]+', "", leasePrcMax)

        itemList.append({'건물명':name, '건물타입':typeName, '동수':totDongCnt,
            '호수':totHsehCnt, '준공일':useAprvYmd, '매매건수':dealCnt,
            '전세건수':leaseCnt, '월세건수':rentCnt,'매매최저가':dealPrcMin,
            '매매최고가':dealPrcMax, '전세최저가':leasePrcMin, '전세최고가':leasePrcMax})
itemList
print(f"총 건물 수 : {len(itemList)}")
#데이터 json 형태로 저장
import pandas as pd
df = pd.DataFrame(itemList)
df.to_json(f'./데이터수집/정적웹크롤링/네이버 부동산 {searchStr} 매물정보.json', 
    orient='records', force_ascii=False, indent=4)