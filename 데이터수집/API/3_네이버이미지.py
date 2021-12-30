#API 데이터 가져오는 절차
#(1) 해당 사이트의 인증을 받는다.
#(2) Service URL 파악
#(3) 입력 파라미터 무엇인지 파악
#(4) 결과 데이터 형식이 어떤 것인지 파악
#(5) 원하는 데이터만 가져오기

id = '30KhrOLi_ppVXTpsFyIL'
pw = 'XN4SPXwGmC'
#URL를 통해 서버에 데이터 요청 방법
import urllib.request #URL을 통해서 웹 정보 가져오는 모듈
import datetime
import json
import re
def getRequestUrl(url, decoding='utf-8') :
    header = {'X-Naver-Client-Id':id, 'X-Naver-Client-Secret':pw}
    req = urllib.request.Request(url, headers=header)
    try :
        response = urllib.request.urlopen(req)
        if response.getcode() == 200 :
            print(f"[{datetime.datetime.now()}] URL Request Success")
            return response.read().decode(decoding)
    except Exception as e:
        print(e)
        print(f"[{datetime.datetime.now()}] Error for Url")

def getNaverImg(searchStr, start, display, filter='small') :
    serviceUrl = "https://openapi.naver.com/v1/search/image"
    param = "?query=" + urllib.parse.quote(searchStr) #utf-8인코딩으로 변환
    param += f"&display={display}&start={start}&filter={filter}"
    url = serviceUrl+param
    rtn = getRequestUrl(url)
    if rtn :
        return json.loads(rtn)
    else :
        return None


shopping = getNaverImg('고양이', 1, 100)
print(shopping)

searchStr = "고양이"
itemList = []
for start in range(1, 1000, 100) :
    jsonResult = getNaverImg(searchStr, start, 100)
    for item in jsonResult['items'] :
        itemDic = {}
        itemDic['제목'] = re.sub(r'<.+?>', "", item['title']) 
        itemDic['제목'] = re.sub(r'[^가-힣\da-zA-Z ]', "",  itemDic['제목']) 
        itemDic['링크'] = item['link']
        itemList.append(itemDic)
print(f'가져온 이미지 수 : {len(itemList)}')
print(itemList)


#이미지 저장
import os
path = './데이터수집/API/네이버 이미지'
if not os.path.exists(path) :
    os.makedirs(path)
import dload

for idx, item in enumerate(itemList) :
    title = item['제목']
    title = title.strip() #문자열 앞뒤에 공백 지우기
    link = item['링크']
    fileName = str(idx)+"_"+title
    #파일의 확장자 정보 찾기
    m = re.search(r'\.(\w+?)$', link) #greedy, non-greedy
    fileExtention = "jpg" if m==None else m.group(1)
    fileName += "." + fileExtention
    filePath = path + "/" + fileName
    #링크정보로 데이터 다운로드시#
    #dload
    #pip install dload
    if idx==100 :
        break
    try :
        urllib.request.urlretrieve(link, filePath)
        #dload.save(link, filePath)
        print(f"{fileName} 다운로드 완료")
    except :
        print(f'[다운로드 오류]-{link}')

        

