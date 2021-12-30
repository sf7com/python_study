from 데이터수집.WebCrawling import getRequestUrl
import json
import re
def getTourismStats(yyyymm, nat_cd, ed_cd) :
    serviceKey = 'UONoET2Grg2f0je6CsDmMP%2BsJ%2B1fT93YXiwOrW0vSx9x%2FGmRf%2BJkVpL%2BTtntfYLXec8mro6YEsU5OI94WOfr7w%3D%3D'
    serviceUrl = 'http://openapi.tour.go.kr/openapi/service/EdrcntTourismStatsService/getEdrcntTourismStatsList'
    param = "?_type=json"
    param += "&serviceKey=" + serviceKey
    param += "&NAT_CD=" + nat_cd
    param += "&ED_CD=" + ed_cd
    param += "&YM=" + yyyymm
    url = serviceUrl + param
    result = getRequestUrl(url)
    if result :
        return json.loads(result)
    else :
        return None
#getTourismStats('202011', '112', 'E')

nat_cd = '275' #중국 112, 한국 100, 일본 130, 미국 275
startYear = 2010
endYear = 2020
ed_cd = "D" if nat_cd == '100' else 'E' #E방한외래관광객, D한국사람이 외국
itemList = []
for year in range(startYear, endYear+1) : 
    for mon in range(1, 13) :
        yyyymm = f'{year}{mon:0>2}' # 202002
        jsonData = getTourismStats(yyyymm, nat_cd, ed_cd)
        jsonData
        try :
            item = jsonData['response']['body']['items']
            if item == "" :
                continue
            else :
                item = item ['item']
                itemDic = {}
                itemDic['날짜'] = item['ym']
                itemDic['국가명'] = item['natKorNm']
                itemDic['관광객수'] = item['num']
                itemDic['출입국'] = item['ed']
                itemList.append(itemDic)
        except :
            continue
itemList

itemFirst = itemList[0]
itemLast = itemList[-1]
itemFirst
path = './데이터수집/API/'
fileName = f'{itemFirst["국가명"]}_' + \
    f'{itemFirst["출입국"]}_{itemFirst["날짜"]}_{itemLast["날짜"]}.json'

with open(path+fileName, 'w', encoding='utf-8') as f :
    f.write(json.dumps(itemList, indent=4, ensure_ascii=False))
print(f"{fileName} 저장 되었습니다.")
