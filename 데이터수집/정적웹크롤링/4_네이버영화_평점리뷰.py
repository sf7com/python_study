from 데이터수집.WebCrawling import getRequestUrl
from bs4 import BeautifulSoup
import pprint

itemList = []
page=1
url = f'https://movie.naver.com/movie/point/af/list.naver?&page={page}'
html = getRequestUrl(url, 'utf-8')
# html
soup = BeautifulSoup(html, 'html.parser')
tag_tbody = soup.find('tbody')
tag_trs = tag_tbody.find_all('tr')
# tag_trs[1].find_all('td')
# pprint.pprint(tag_trs[2].find_all('td'))
tag_tds = tag_trs.find_all('td')
tag_tds
tag_ems = tag_tds.find('em')
print(tag_ems)
for tr in tag_trs :
    tds = tr.find_all('td')
    print(tds)
#     item = {}
#     item['날짜'] = tds[0].text
#     item['종가'] = int(tds[1].text.replace(",",""))
#     className = ''.join(tds[2].find('span')['class'])
#     #className
#     if 'nv' in className :
#         item['전일비'] = "-"+tds[2].text
#     else : 
#         item['전일비'] = tds[2].text
#     item['전일비'] = int(item['전일비'].replace(",","").replace("\n","").replace("\t",""))
#     item['시가'] = int(tds[3].text.replace(",",""))
#     item['고가'] = int(tds[4].text.replace(",",""))
#     item['저가'] = int(tds[5].text.replace(",",""))
#     item['거래량'] = int(tds[6].text.replace(",",""))
#     itemList.append(item)

# print("데이터 수 : ", len(itemList))
# itemList


# import pandas as pd
# df = pd.DataFrame(itemList, columns=itemList[0].keys())
# df.to_csv('./데이터수집/정적웹크롤링/삼성전자주가.csv', index=False, encoding='utf-8')