from selenium import webdriver
from bs4 import BeautifulSoup
import re
import json

import selenium
#최상위 경로에 드라이버 넣은경우 드라이버 경로 안적어도됨
wd = webdriver.Chrome() 
url = "https://www.coffeebeankorea.com/store/store.asp"
wd.get(url) 

import time
for i in range(1, 20) : #매장 번호
    time.sleep(1) #1초 로딩 시간 
    try : 
        wd.execute_script(f'storePop2({i})')
    except :
        continue
html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
stores = soup.find_all('div', class_='store_popup')
print(f'매장개수 : {len(stores)}') #매장개수 : 11

storeList = []
for item in stores :
    name = item.find('div', class_='store_txt').h2.text
    #name = item.find('div', class_='store_txt').find('h2').text
    info = item.find('p', class_='tag')
    #'디카페인두유와이파이흡연가능드라이브 스루딜리버리 '
    infoList = []
    for tag in info.children : 
        try :
            infoList.append(tag.text)
        except :
            continue
    info = ','.join(infoList)

    tbody = item.find('tbody')
    tds = tbody.find_all('td')
    hours = tds[0].text
    park = tds[1].text
    address = tds[2].text
    phone = tds[3].text
    storeList.append({'매장명':name, '정보':info,
        '영업시간':hours, '주차':park, '주소':address,
        '전화번호':phone})
storeList

#파일저장
import pandas as pd
df = pd.DataFrame(storeList)
df
df.to_json('./데이터수집/동적웹크롤링/커피빈매장.json', 
        orient='records', force_ascii=False, indent=4)
#15시 10분에 시작하겠습니다.
