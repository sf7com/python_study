from 데이터수집.WebCrawling import getRequestUrl

serviceUrl = 'https://www.hollys.co.kr/store/korea/korStore2.do?'
param = 'pageNo=10'
url = serviceUrl+param

html = getRequestUrl(url)
print(html)

#태그분석 모듈
#pip install bs4
from bs4 import BeautifulSoup 
soup = BeautifulSoup(html, 'html.parser')
print(soup.prettify()) #HTML정보를 예쁘게 정렬해서 출력

tag_tbody = soup.find('tbody') #tbody태그 하나만 찾아서 리턴
tag_tbody
tag_trs = tag_tbody.find_all('tr') #tr태그들 찾아서 리스트로 반환
tag_trs

for tr in tag_trs :
    tag_tds = tr.find_all('td')
    itemDic = {}
    itemDic['지역정보'] = tag_tds[0].text #지역정보    
    itemDic['매장명'] = tag_tds[1].text  
    itemDic['매장현황'] = tag_tds[2].text #(영업중, 오픈예정 등)
    itemDic['주소'] = tag_tds[3].text #매장주소    
    itemDic['전화번호'] = tag_tds[5].text #지역정보
    print(itemDic)