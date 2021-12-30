from selenium import webdriver
from bs4 import BeautifulSoup
import re
import json
import selenium
#최상위 경로에 드라이버 넣은경우 드라이버 경로 안적어도됨
wd = webdriver.Chrome() 
url = "https://www.starbucks.co.kr/store/store_map.do?disp=locale"
wd.get(url) 

import time
for i in range(1, 20) : #매장 번호
    time.sleep(1) #1초 로딩 시간 
    try : 
        wd.execute_script(f"getStoreDetail('{i}')")
    except :
        continue

html = wd.page_source
soup = BeautifulSoup(html, 'html.parser')
stores = soup.find_all('section', class_='shopArea_pop01_inner')
print(f'매장개수 : {len(stores)}') #매장개수 : 10

storeList = []
for item in stores :
    name = item.find('header', class_='titl').h6.text
    dlTags = item.find_all('dl', class_='shopArea_info')
    addr = list(dlTags[0].dd.children)
    addr =  addr[0]
    try :
        phone = dlTags[1].dd.text
    except :
        phone = ""
    try :   
        park = dlTags[4].dd.text
    except :
        park = ""
    try : 
        directions = dlTags[5].dd.text
    except :
        directions = ""
    try :
        grade = dlTags[6].dt.text
    except :
        grade = ""
    hours = item.find('div', class_='date_time cafetimeWrap').find_all('dd')
    hours
    hours = [data.text for data in hours]
    hours
    weekList = ['월요일','화요일','수요일','목요일','금요일','토요일','일요일']
    hourList = []
    for week in weekList :
        for hour in hours :
            if week in hour :
                hourList.append(hour)
                break
    hourList
    hours = ', '.join(hourList)

    storeList.append({'매장명':name, '정보':grade,
        '오시는길':directions, '주차':park, '주소':addr,
        '전화번호':phone, '매장시간':hours})

storeList
#파일저장
import pandas as pd
df = pd.DataFrame(storeList)
df
df.to_json('./데이터수집/동적웹크롤링/스타벅스매장.json', 
        orient='records', force_ascii=False, indent=4)
#15시 10분에 시작하겠습니다.
