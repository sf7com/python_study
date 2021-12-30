from selenium import webdriver
from bs4 import BeautifulSoup
import re
import json
#최상위 경로에 드라이버 넣은경우 드라이버 경로 안적어도됨
wd = webdriver.Chrome() 
url = 'https://play.google.com/store/movies'
wd.get(url)
html = wd.page_source
html

soup = BeautifulSoup(html, 'html.parser')
movies = soup.find_all('div',class_='j2FCNc')
movieList = []

for movie in movies :
    data = movie.find_all('div', class_='ubGTjb')
    try :
        #(1) 타이틀 정보
        title = data[0].text
        #(2) 관람연령
        age = data[1].text
        #(3) 장르
        genre = data[2].text
        #(4) 평점
        rank = data[3].find('span', class_='w2kbF').text
        #(5) 가격
        price = data[3].find('span', class_='w2kbF ePXqnb').text

        itemDic = {'제목':title, '연령':age, "장르":genre, 
                '평점':rank, '가격':price}
        movieList.append(itemDic)
    except :
        continue
movieList
import pandas as pd
df = pd.DataFrame(movieList)
df.head()
df.info()
df.to_csv('./데이터수집/동적웹크롤링/구글인기영화.csv', encoding='utf-8')
