#WebCrawling.py
import urllib.request #URL을 통해서 웹 정보 가져오는 모듈
import datetime
def getRequestUrl(url, decoding='utf-8') :
    header= {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36'}
    req = urllib.request.Request(url, headers=header)
    try :
        response = urllib.request.urlopen(req)
        if response.getcode() == 200 :
            print(f"[{datetime.datetime.now()}] URL Request Success")
            return response.read().decode(decoding)
    except Exception as e:
        print(e)
        print(f"[{datetime.datetime.now()}] Error for Url")
