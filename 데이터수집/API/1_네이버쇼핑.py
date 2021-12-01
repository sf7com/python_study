#API 데이터 가져오는 절차
#(1) 해당 사이트의 인증을 받는다.
#(2) Service URL 팡ㄱ
#(3) 입력 파라미터 무엇인지 파악
#(4) 결과 데이터 형식이 어떤 것인지 파악
#(5) 원하는 데이터만 가져오기
id = 'jus1qXA5TBSvIew5VfrX'
pw = 'J5Nq_LWHSY'

#URL을 통해 서버에 데이터 요청 방법
import urllib.request #URL을 통해서 웹 정보 가져오는 모듈
import datetime
import json
import pprint
import re
# if response.getcode() == 200 : #정상응답
#     with open('네이버.html','w',encoding='utf-8') as f :
#         f.write(response.read().decode('utf-8'))

def getRequestURL(url, decoding='utf-8') :
    header = {'X-Naver-Client-Id':id, 'X-Naver-Client-Secret':pw}
    req = urllib.request.Request(url, headers=header)
    try : 
        response = urllib.request.urlopen(req)
        if response.getcode() == 200 :
            print(f"[{datetime.datetime.now()}] URL Request Success")
            return response.read().decode(decoding)
    except Exception as e:
        print(e)
        print(f"[{datetime.datetime.now()}] Error for URL")

def getNaverShopping(searchStr, start, display):
    serviceUrl = "https://openapi.naver.com/v1/search/shop.json"
    param = "?query=" + urllib.parse.quote(searchStr) #utf-8인코딩으로 변환
    param += f"&display={display}&start={start}"
    url = serviceUrl+param
    rtn = getRequestURL(url)
    if rtn :
        return json.loads(rtn)
    else :
        return None

shopping = getNaverShopping('노트북',1,100)
print(shopping)

#뉴스 1000건을 가져와서 저장하기
searchStr = "노트북"
itemList = []
for start in range(1,1000,100) :
    jsonResult = getNaverShopping(searchStr,start,100)
    for item in jsonResult['items'] :
        itemDic = {}
        itemDic['상품명'] = re.sub(r'<.+?>', "", item['title']) 
        itemDic['상품명'] = re.sub(r'[^가-힣\da-zA-Z ]', "", itemDic['상품명'])
        itemDic['링크'] = item['link']
        itemDic['이미지링크'] = item['image']
        itemDic['최저가'] = item['lprice']
        itemDic['최고가'] = item['hprice']
        itemDic['브랜드'] = item['brand']
        itemList.append(itemDic)

pprint.pprint(itemList)
print(f'가져온 뉴스 수 : {len(itemList)}')

with open(f'./데이터수집/api/네이버 쇼핑 {searchStr} {len(itemList)}건.json','w',encoding='utf-8') as f:
    #newsList를 json파일 포맷으로 변경
    jsonData = json.dumps(itemList,indent=4, sort_keys=False,ensure_ascii=False)
    #indent :4칸 띄워서 저장
    #sort_key : 키값 별 정렬
    #ensure_ascii : 아스키코드가 아닌 것들은 이스케이프문자로 저장
    #               False해야 한글로 저장이 된다.
    f.write(jsonData)

