from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

wd = webdriver.Chrome(executable_path='c:\pythonProject\python_study\chromedriver.exe')
wd.get('https://kr.investing.com/currencies/usd-krw')
elems = wd.find_elements_by_id('last_last')
print(elems.text)