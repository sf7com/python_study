from 데이터수집.WebCrawling import getRequestUrl
import json
itemList = []
for page in range(1, 2) :
    url = f'https://m.stock.naver.com/api/stock/005180/price?pageSize=60&page={page}'
    rtn = getRequestUrl(url)
    jsonRtn = json.loads(rtn)
    for data in jsonRtn :
        itemDic ={}
        itemDic['날짜'] = data['localTradedAt']
        itemDic['종가'] = data['closePrice'].replace(",","")
        itemDic['등락율'] = data['fluctuationsRatio']
        itemDic['시작가'] = data['openPrice'].replace(",","")
        itemDic['최고가'] = data['highPrice'].replace(",","")
        itemDic['최저가'] = data['lowPrice'].replace(",","")
        itemDic['거래량'] = data['accumulatedTradingVolume']
        itemList.append(itemDic)
X_pred =[]
cols = ['거래량','등락율','시작가','종가','최고가','최저가']
for col in cols :
    for i in range(5) :
        X_pred.append(itemList[i][col])
X_pred
# [6424, 12799, 12579, 15325, 8109,
#  '0.58', '-0.38', '0.00', '0.00', '0.58',
#  '52100', '52300', '52400', '51500', '51800',
#  '52100', '51800', '52000', '52000', '52000',
#  '52500', '52300', '52500', '52400', '52100',
#  '51700', '51600', '51700', '51500', '51500']

#머신러닝 모델 불러오기
import pickle
with open('./주가예측모델/data/빙그레_주가_모델.pickle','rb') as f :
    model = pickle.load(f)
with open('./주가예측모델/data/빙그레_정규화객체.pickle','rb') as f :
    ss = pickle.load(f)

X_pred_Scaled = ss.transform([X_pred])
X_pred_Scaled

model.predict(X_pred_Scaled) #array([52077.68027057])
pred_y = model.predict(X_pred_Scaled)
past_price = itemList[0]['종가'] #예측 전날의 종가
#등락율 계산
pred_ratio = (pred_y[0]-int(past_price))/int(past_price)*100
print(f"예측 종가 : {pred_y[0]:0f}, 예측 등락률 : {pred_ratio:.2f}%")

