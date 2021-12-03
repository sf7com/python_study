from 라이브러리.WebCrawling import getRequestUrl
from bs4 import BeautifulSoup

#BS4 : HTML 내에서 원하는 태그 찾는 라이브러리
#단위 데이터 : find(), select_one()
#모든 데이터 : find_all(), select()

#네이버 홈페이지에 있는 뉴스 스탠드 언론사 명
url = "https://www.naver.com"
html = getRequestUrl(url)
html

soup = BeautifulSoup(html, 'html.parser')

#(1) find(태그명, 속성) : 일치하는 태그 하나만 찾아서 리턴
newsStand = soup.find('div', class_='thumb_area')
newsStand

#newsStand 태그 내에서 다음의 태그 검색
news1 = newsStand.find('div', attrs={'class':'thumb_box'}) 
news1

#(2) find_all(태그명, 속성) : 일치하는 모든 태그를 찾아 리스트로 반환
news = newsStand.find_all('div', attrs={'class' : 'thumb_box'})
for tag in news :
    name = tag.find('img')['alt'] # [] 태그의 속성명을 넣으면 속성명을 알 수 있다. 
    print(name)                   # .text 태그의 값을 알려준다.
#-----------------------------------------------------------------
#css selector 경로를 통해 검색
#태그.클래스명
#태그#아이디명
img_tags = soup.select('div.thumb_area > div > a > img')
img_tags

for tag in img_tags :
    print(tag['alt'])

img_tags = soup.select_one('div.thumb_area > div > a > img')
img_tags
#-------------------------------------------------------------------
#실시간 뉴스 언론사 및 내용 가져오기
realTimeNews = soup.find('div', id='NM_ONELINE_NAME')
realTimeNews
realTimeNews.text

realTimeContents = soup.find('div', id='NM_ONELINE_ROLLING')
realTimeContents
realTimeContents.text #태그내의 값을 얻는다

realTimeNews = soup.select_one('div#NM_ONELINE_NAME')
realTimeNews.text

realTimeContents = soup.select_one('div#NM_ONELINE_ROLLING')
realTimeContents.text