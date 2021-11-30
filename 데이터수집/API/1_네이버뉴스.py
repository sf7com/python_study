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

def getNaverNews(searchStr, start, display):
    serviceUrl = "https://openapi.naver.com/v1/search/news.json"
    param = "?query=" + urllib.parse.quote(searchStr) #utf-8인코딩으로 변환
    param += f"&display={display}&start={start}"
    url = serviceUrl+param
    rtn = getRequestURL(url)
    if rtn :
        return json.loads(rtn)
    else :
        return None

news = getNaverNews('IT기업',1,100)
print(news)

#뉴스 1000건을 가져와서 저장하기
searchStr = "카카오"
newsList = []
for start in range(1,1000,100) :
    jsonResult = getNaverNews(searchStr,start,100)
    for item in jsonResult['items'] :
        newsDic = {}
        # newsDic['제목'] = item['title']
        # newsDic['요약'] = item['description']
        # newsDic['링크'] = item['link']
        newsDic['제목'] = re.sub(r'<.+?>', "", item['title']) 
        newsDic['요약'] = re.sub(r'<.+?>', "", item['description'])
        newsDic['제목'] = re.sub(r'[^\w ]', "", newsDic['제목'] ) 
        newsDic['요약'] = re.sub(r'[^\w ]', "", newsDic['요약'])
        newsDic['링크'] = item['link']
        pDate = datetime.datetime.strptime(item['pubDate'],'%a, %d %b %Y %H:%M:%S +0900')
        #'pubDate' : 'Thu, 19 Aug 2021 15:54:00 +0900
        #날짜 형식 바꾸기
        pDate = pDate.strftime('%Y-%m-%d %H:%M:%S')
        #'pubDate' : '2021-11-30 15:53:23'
        newsDic['날짜'] = pDate
        newsList.append(newsDic)

pprint.pprint(newsList)
print(f'가져온 뉴스 수 : {len(newsList)}')

with open(f'./데이터수집/api/네이버 뉴스 {searchStr} {len(newsList)}건.json','w',encoding='utf-8') as f:
    #newsList를 json파일 포맷으로 변경
    jsonData = json.dumps(newsList,indent=4, sort_keys=False,ensure_ascii=False)
    #indent :4칸 띄워서 저장
    #sort_key : 키값 별 정렬
    #ensure_ascii : 아스키코드가 아닌 것들은 이스케이프문자로 저장
    #               False해야 한글로 저장이 된다.
    f.write(jsonData)

