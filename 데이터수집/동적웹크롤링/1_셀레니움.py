#pip install selenium
#셀레니움 라이브러리 웹 브라우저를 테스트하는
#목적으로 만들어짐
#웹 프로래밍을 통해 조작

#크롬 브라우저 조작
from selenium import webdriver
wd = webdriver.Chrome('./chromedriver.exe')
wd.get('https://naver.com')

#find_element_by_xpath() xpath 경로를 통해서 해당 요소를 찾는다.
elem = wd.find_element_by_xpath('//*[@id="NM_FAVORITE"]/div[1]/ul[2]/li[2]/a')
elem.click() #해당 엘레멘트 마우스 클릭

wd.back() #뒤로가기
wd.forward() #앞으로가기
wd.refresh() #새로고침
wd.back()

elem = wd.find_element_by_id('query')
elem.send_keys("파이썬")
elem.send_keys("클라우드")
elem.clear() #모든 내용 제거
#12시 05분에 시작하겠습니다.

elem.send_keys("파이썬")
from selenium.webdriver.common.keys import Keys
elem.send_keys(Keys.ENTER)

#현재 Page의 html 정보
html = wd.page_source
html

#네이버 지식백과 사전 내용 가져오기
#제목
elems = wd.find_elements_by_class_name('lnk_tit')
#내용
elems_contents = wd.find_elements_by_class_name('api_txt_lines.desc')
elems
elems_contents
for title, desc in zip(elems, elems_contents) :
    print(title.text) #태그내의 내용가져오기
    print(title.get_attribute('href')) #태그내의 속성값 가져오기
    print(desc.text)
    print("-"*50)
