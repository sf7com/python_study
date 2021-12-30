from selenium import webdriver
from bs4 import BeautifulSoup
import re
import json
import urllib.request

wd = webdriver.Chrome()
searchStr = "파이썬"
for i in range(0,1000, 100) :
    serviceUrl ='http://www.riss.kr/search/Search.do?isDetailSearch=N&searchGubun=true&viewYn=OP&queryText=&exQuery=&exQueryText=&order=%2FDESC&onHanja=false&strSort=RANK&p_year1=&p_year2=&orderBy=&mat_type=&mat_subtype=&fulltext_kind=&t_gubun=&learning_type=&ccl_code=&inside_outside=&fric_yn=&image_yn=&gubun=&kdc=&ttsUseYn=&l_sub_code=&fsearchMethod=&sflag=1&isFDetailSearch=N&pageNumber=1&resultKeyword=%EB%B9%85%EB%8D%B0%EC%9D%B4%ED%84%B0&fsearchSort=&fsearchOrder=&limiterList=&limiterListText=&facetList=&facetListText=&fsearchDB=&icate=re_a_kor&colName=re_a_kor&pageScale=100&isTab=Y&regnm=&dorg_storage=&language=&language_code=&clickKeyword=&relationKeyword='
    param = f'&strQuery={urllib.parse.quote(searchStr)}'
    param += f'&query={urllib.parse.quote(searchStr)}'
    param += f'&iStartCount={i}' # 100씩 증가하도록 해야함
    url = serviceUrl + param
    wd.get(url)

    #(1) 전체선택 클릭
    elem = wd.find_element_by_xpath('//*[@id="divContent"]/div[2]/div/div[4]/div[1]/div[1]/label/span')
    elem.click()

    #(2) 내보내기 실행
    wd.execute_script('exportData()')

    #(3) 새창 핸들링(새창 포커스)
    wd.switch_to.window(wd.window_handles[1])
    # 'WebDriver' object has no attribute 'switch_to_window'
    wd.get_window_position(wd.window_handles[1])

    #(4) 새창이 다 로딩될때까지 대기 
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    #대기방법1 time.sleep(초) 를 통한 강제대기
    #대기방법2 어떤 요소가 나올때까지 대기
    elem = WebDriverWait(wd, 10).until(EC.presence_of_element_located(
                    (By.XPATH, '//*[@id="wrap"]/form/div/div[2]/div[1]/div/ul/li[3]/label'))) #특정 요소가 나올때까지 최대 10초대기
    elem.click()

    #(5) 내보내기 실행
    wd.execute_script('f_submit()')
    #(6) 현재창(새창) 닫기
    wd.close()
    #(7) 본래창 핸들링
    wd.switch_to.window(wd.window_handles[0])
    wd.get_window_position(wd.window_handles[0])
