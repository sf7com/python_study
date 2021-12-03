from 데이터수집.WebCrawling import getRequestUrl
from bs4 import BeautifulSoup 
serviceUrl = 'https://www.hollys.co.kr/store/korea/korStore2.do?'

itemList = []
for page in range(1, 57) : 
    param = f'pageNo={page}'
    url = serviceUrl+param
    html = getRequestUrl(url)
    soup = BeautifulSoup(html, 'html.parser')
    #print(soup.prettify()) #HTML정보를 예쁘게 정렬해서 출력
    tag_tbody = soup.find('tbody') #tbody태그 하나만 찾아서 리턴
    tag_tbody
    tag_trs = tag_tbody.find_all('tr') #tr태그들 찾아서 리스트로 반환
    tag_trs
    for tr in tag_trs :
        tag_tds = tr.find_all('td')
        itemDic = {}
        itemDic['지역정보'] = tag_tds[0].text #지역정보    
        itemDic['매장명'] = tag_tds[1].text  
        itemDic['매장현황'] = tag_tds[2].text #(영업중, 오픈예정 등)
        itemDic['주소'] = tag_tds[3].text #매장주소    
        itemDic['전화번호'] = tag_tds[5].text #지역정보
        itemList.append(itemDic)

print("총 매장 개수 : ", len(itemList))

#데이터 저장
import json
with open('./데이터수집/정적웹크롤링/할리스카페매장.json', 'w',
    encoding='utf-8') as f :
    retJson = json.dumps(itemList, indent=4, ensure_ascii=False)
    f.write(retJson)

#csv 파일 포맷으로 저장
# 매장명,지역정보,주소,전화번호,..
# 수워홀리스,권선동,권선동,010-311-3333
with open('./데이터수집/정적웹크롤링/할리스카페매장.csv', 'w',
    encoding='utf-8') as f :
    colList = itemList[0].keys()
    colList #dict_keys(['지역정보', '매장명', '매장현황', '주소', '전화번호'])
    ','.join(colList)+"\n" #'지역정보,매장명,매장현황,주소,전화번호\n'
    f.write(','.join(colList)+"\n")
    for item in itemList :
        f.write(','.join(item.values()) + "\n")

import pandas as pd
df = pd.DataFrame(itemList, columns=itemList[0].keys())
df.head()
df.info()
df.to_csv('./데이터수집/정적웹크롤링/할리스카페매장2.csv', encoding='utf-8', index=True)

df.to_json('./데이터수집/정적웹크롤링/할리스카페매장2.json',orient='records', force_ascii=False, indent=4)