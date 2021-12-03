from 데이터수집.WebCrawling import getRequestUrl
from bs4 import BeautifulSoup

itemList = []
for page in range(1, 11) :
    url = f'https://finance.naver.com/item/sise_day.naver?code=005930&page={page}'
    html = getRequestUrl(url, 'euc-kr')
    #html
    soup = BeautifulSoup(html, 'html.parser')
    tag_trs = soup.find_all('tr', attrs={"onmouseover" : "mouseOver(this)"})
    for tr in tag_trs :
        tds = tr.find_all('td')
        item = {}
        item['날짜'] = tds[0].text
        item['종가'] = int(tds[1].text.replace(",",""))
        className = ''.join(tds[2].find('span')['class'])
        #className
        if 'nv' in className :
            item['전일비'] = "-"+tds[2].text
        else : 
            item['전일비'] = tds[2].text
        item['전일비'] = int(item['전일비'].
                        replace(",","").replace("\n","").replace("\t",""))
        item['시가'] = int(tds[3].text.replace(",",""))
        item['고가'] = int(tds[4].text.replace(",",""))
        item['저가'] = int(tds[5].text.replace(",",""))
        item['거래량'] = int(tds[6].text.replace(",",""))
        itemList.append(item)

print("데이터 수 : ", len(itemList))
itemList

#날짜별 종가의 시각화
import matplotlib.pyplot as plt
dateList = [item['날짜'] for item in itemList[::-1]] #날짜 오름 차순
priceList = [item['종가'] for item in itemList[::-1]]
plt.plot(dateList, priceList)
plt.title("Samsung Stock")
plt.xlabel("Date")
plt.ylabel('Price')
plt.xticks(rotation=70) #x축 눈금을 70도 기울인다.
plt.show()

import pandas as pd
df = pd.DataFrame(itemList, columns=itemList[0].keys())
df.to_csv('./데이터수집/정적웹크롤링/삼성전자주가.csv', index=False, encoding='utf-8')