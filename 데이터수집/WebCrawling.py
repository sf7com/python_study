import urllib.request #URL을 통해서 웹 정보 가져오는 모듈
import datetime
def getRequestUrl(url, decoding='utf-8') :
    req = urllib.request.Request(url)
    try :
        response = urllib.request.urlopen(req)
        if response.getcode() == 200 :
            print(f"[{datetime.datetime.now()}] URL Request Success")
            return response.read().decode(decoding)
    except Exception as e:
        print(e)
        print(f"[{datetime.datetime.now()}] Error for Url")