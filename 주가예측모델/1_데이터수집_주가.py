#1_데이터수집_주가.py
#주가에 영향을 미치는 변수
#거래량,외국인 보유율,뉴스기사(실적발표), 최대주주 지분율,미국증시 등등

#날짜별로 얻기

#종가,거래량,등락률 3가지 정보 주가예측
#전날 데이터로 -> 다음날 예측?
#5일간 모든 데이터 -> 데이터 예측?
#10일간 모든 데이터 -> 다음날 예측?

#피쳐 : 종가, 등락률, 거래량 (5일치)
#타겟 : 다음날 등락률
#실습 : 5일간 모든 데이터 -> 다음날 예측 모델 만들기

#주가 데이터 API 또는 웹크롤링
#웹크롤링 (3년치)

from 데이터수집.WebCrawling import getRequestUrl
import json
itemList = []
for page in range(1, 51) :
    url = f'https://m.stock.naver.com/api/stock/005180/price?pageSize=60&page={page}'
    rtn = getRequestUrl(url)
    jsonRtn = json.loads(rtn)
    for data in jsonRtn :
        itemDic ={}
        itemDic['날짜'] = data['localTradedAt']
        itemDic['종가'] = data['closePrice'].replace(",","")
        itemDic['등락율'] = data['fluctuationsRatio']
        itemDic['시작가'] = data['openPrice'].replace(",","")
        itemDic['최고가'] = data['highPrice'].replace(",","")
        itemDic['최저가'] = data['lowPrice'].replace(",","")
        itemDic['거래량'] = data['accumulatedTradingVolume']
        itemList.append(itemDic)
print("데이터 수 : ", len(itemList))
#itemList

import os
#데이터 폴더 생성
path = './주가예측모델/data'
if not os.path.exists(path) :
    os.makedirs(path)

import pandas as pd
df = pd.DataFrame(itemList, columns=itemList[0].keys())
df.to_csv(os.path.join(path, f'빙그레 {itemList[0]["날짜"]}~{itemList[-1]["날짜"]}.csv'), 
index=False, encoding='utf-8')


