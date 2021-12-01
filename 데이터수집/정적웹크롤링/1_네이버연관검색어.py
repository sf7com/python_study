#웹크롤링 : 웹상의 원하는 정보를 자동으로 가져오는 기술
#API가 제공되지 않은 경우 활용

#웹크롤링 절차
#(1) 원하는 정보를 선정
#(2) URL분석 (네트워크 분석)
#(3) 태그 분석
#(4) 해당 태그의 데이터만 가져오기
#(5) 데이터 저장
import urllib.request #URL을 통해서 웹 정보 가져오는 모듈
import datetime
def getRequestUrl(url, decoding='utf-8') :
    req = urllib.request.Request(url)
    try :
        response = urllib.request.urlopen(req)
        if response.getcode() == 200 :
            print(f"[{datetime.datetime.now()}] URL Request Success")
            return response.read().decode(decoding)
    except Exception as e:
        print(e)
        print(f"[{datetime.datetime.now()}] Error for Url")

print("<< 네이버 연관 검색어를 가져오기 >>")
searchStr = input("검색어 입력 > ")
serviceUrl = 'https://ac.search.naver.com/nx/ac?'
param = f'q={urllib.parse.quote(searchStr)}&st=100'
url = serviceUrl+param
result = getRequestUrl(url)
#print(result)

import json
jsonResult = json.loads(result)
jsonResult['query']
jsonResult['items']

for item in jsonResult['items'][0] :
    print(item[0])

