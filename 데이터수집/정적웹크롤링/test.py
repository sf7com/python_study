from bs4 import BeautifulSoup
from selenium import webdriver
from time import sleep
import urllib.request
import pandas as pd
import time
import json
import datetime
import re

#[CODE 1]
#네이버 보안 정책으로 간접 URL를 selenium 라이브러리를 통해 얻음
wd = webdriver.Chrome('./chromedriver.exe')
def getRequestUrl(url):
    req = urllib.request.Request(url)
    try:
        response = urllib.request.urlopen(req)
        if response.getcode() == 200:
            print("[%s] Url Request Success" % datetime.datetime.now())
            retData = response.read().decode('utf-8')
            return json.loads(retData)
    except Exception as e:
        print(e)
        print("[%s] Error for URL : %s" % (datetime.datetime.now(), url))
        return None
    
def getRequestHTML(url):
    try:
        wd.get(url)
        sleep(1)
        reUrl = wd.current_url
        html = wd.page_source
    except Exception as e:
        print(e)
        print("[%s] Error for URL : %s" % (datetime.datetime.now(), url))
        return ""


def getComplexInfo(key, lgeo, z, cortarNo) :
    service_url = "https://m.land.naver.com/cluster/ajax/complexList"
    parameters = "?itemId=" + key
    parameters +="&lgeo=" + lgeo
    parameters +="&rletTpCd=APT%3AJGC"
    parameters +="&tradTpCd=A1%3AB1%3AB2"
    parameters +="&z=" + z
    parameters +="&cortarNo=" + cortarNo
    
    url = service_url + parameters
    rtnData = getRequestHTML(url)
    return rtnData

#[CODE 0]
def main():
    searchStr = "권선구 권선동"
    service_url = "https://m.land.naver.com/search/result/"
    parameters = urllib.parse.quote(searchStr)
    url = service_url + parameters
    html = getRequestHTML(url)
    html
    soupCB1 = BeautifulSoup(html, 'html.parser')
    #print(soupCB1.prettify())

    #태그 속성 중 id를 통해 가져올때 '#'을 붙인다
    localInfo = soupCB1.select("#mapSearch > script")[0].string

    #정규표현식을 통해 패턴에 맞는 문자열을 가져온다
    lat = re.search("lat: '(.*)'" , localInfo).groups()[0]
    lon = re.search("lon: '(.*)'" , localInfo).groups()[0]
    z = re.search("z: '(.*)'" , localInfo).groups()[0]
    cortarNo = re.search("cortarNo: '(.*)'" , localInfo).groups()[0]
    cortarNm = re.search("cortarNm: '(.*)'" , localInfo).groups()[0]
    rletTpCds = re.search("rletTpCds: '(.*)'" , localInfo).groups()[0]
    tradTpCds = re.search("tradTpCds: '(.*)'" , localInfo).groups()[0]

    complexInfo = soupCB1.select("div.ARTICLE_container > div.marker_circle")

    #지역내의 단지 정보를 가져와 List에 저장한다.
    complexMapList = []
    for complexData in complexInfo :
        complexMap = {}
        complexMap["lat"] = complexData["lat"]
        complexMap["lon"] = complexData["lon"]
        complexMap["count"] = complexData["count"]
        complexMap["lgeo"] = complexData["lgeo"]
        complexMap["key"] = complexData["key"]
        
        
        rtnData = getComplexInfo(complexMap["key"], complexMap["lgeo"], z, cortarNo)
        

        #pre 태그 찾아서 내용 json으로 바꾸기
        complex_soup = BeautifulSoup(rtnData, 'html.parser')
        tag = complex_soup.find("pre")
        complexJson = json.loads(tag.text)
        complexMap['detail'] = complexJson["result"]
        #print(complexMap['detail'][0]["hscpNm"])
        complexMapList.append(complexMap)

    
    #모든 단지 정보 합치기
    i = 0
    tempList = []
    for complexInfo in complexMapList :
        for building in complexInfo['detail'] :
            tempList.append(building)
    df = pd.json_normalize(tempList)
    df.to_csv(f"./{searchStr}_부동산 정보.csv", index=False, encoding='cp949')  
    
        
    wd.quit()
    
if __name__ == '__main__':
    main()


