#정규표현식 Regular Expression
#문자열 패턴 매칭, 문자열 검색, 파싱, 문자열 바꿈 등에서 활용

# . : 글자 하나를 의미
# ^ : 문자열의 시작
# $ : 문자열의 끝
# ? : 앞문자가 있어도 되고 없어도 된다. ex)appl?e => apple,appe
#(?!) : 대소문자 구분안함
# | : 다자 택일 (or조건) ex) a|bpple => apple, bpple

#반복기호
# + : 1번 이상을 의미 ex) a+pple => apple, aaaaple, pple(X)
# * : 0번 이상을 의미 ex) a*pple => apple, aaaaple, pple
# {m,n} : m번이상 n번이하 반복 ex) ca{2,5}t->caat, caaat, 
#                                       caaaat, caaaaat
# {0,} : 0번이상 의미 '*'와 의미가 같음
# {1,} : 1번이상 의미 '+'와 의미가 같음
# {0,1} : 0번이상 1번이하 '?'와 의미가 같음
# {n} : n번 반복 ex) a{4} -> aaaa , aaa(X)

#문자클래스
# [] : [] 사이의 문자들과 매치 ex) [abc]pple => 
#                                       apple, bpple, cpple
#문자클래스 특수 용도
#[0-9] : [0123456789] => 숫자 하나와 매치
#[a-z] : 알파벳 소문자와 매치
#[A-Z] : 알파벳 대문자와 매치
#[a-zA-Z] : 대소문자 구분없이 알파벳 하나와 매치
#[ㄱ-ㅎ|ㅏ-ㅣ|가-힣] : 한글과 매치
#[가-힣] : 한글 글자 하나와 매치
#[^문자 집합] : 해당 문자들과 아닌것과 매치
             #ex) [^abc]pple =>dpple, fpple, apple(X)

# \d : 모든 숫자와 매치 [0-9]와 같음
# \D : 숫자가 아닌것과 매치 [^0-9]와 같음
# \w : 문자+숫자+"_" 와 매치 (숫자와 알파벳 언더바) 
        #[a-zA-Z0-9_] 와 같음
#\W : (문자+숫자+"_") 가 아닌것과 매치 [^a-zA-Z0-9_]
#\s : whiteSapce 문자와 매치 [ \t\b\r\f\v]
#\S : whiteSapce 문자가 아닌것과 매치 [^ \t\b\r\f\v]

#정규표현식 앞에 r''사용
#\b : 단어의 경계(공백, 탭, 컴마 대시등)과 매치
#\B : 단어의 경계가 아닌것과 매치

#그룹 () : () 사이의 모든 문자와 매칭 및 그룹 기능이 있다.
#---------------------------------------------------------
#파이썬 re모듈 함수
#(1) match() : 문자열의 처음부터 끝까지 정규식와 매치되는지 조사
#              ex) 주민번호, 전화번호, 이메일 등
#(2) search() : 문자열의 전체를 검색하여 정규식과 매치되는 문자열 조사
#(3) findall() : 정규식과 매치되는 모든 문자열 리스트로 반환
#(4) split() : 정규식과 매치되는 문자열을 기준으로 파싱하여 리스트로 반환
#(5) sub() : 정규식과 매치되는 문자열을 다른 문자열로 바꾼다.
#---------------------------------------------------------------
import re

#(1) match()
text = 'a연아'
m = re.match('.연아', text) #매치된 결과가 없으면 None으로 리턴된다.
if m!=None :
    print(m.group())
else : 
    print("매치되지 않습니다.")

#주민번호 매치
text = '881009-1234567'
m = re.match('^\d{6}-\d{7}$', text) #매치된 결과가 없으면 None으로 리턴된다.
if m!=None :
    print(m.group())
else : 
    print("매치되지 않습니다.")

#휴대폰 번호와 매치
text = '010-3141-1233' #(o)
text = '016-3141-1233' #(X)
text = '010-444-1233' #(o)
text = '010-444-12335' #(X)
m = re.match('^[01]{3}-\d{3,4}-\d{4}$', text) #매치된 결과가 없으면 None으로 리턴된다.
if m!=None :
    print(m.group())
else : 
    print("매치되지 않습니다.")

#(2) search() : 특정한 패턴을 가진 문자열 검색시 사용
text = "Good care cere"
m = re.search('c', text)
print(m.end()) #care
print("일치하는 문자열 : ", m.group())
print("입력받은 문자열 : ", m.string)
print("일치하는 문자열 시작 인덱스 : ", m.start())
print("일치하는 문자열 끝 인덱스 : ", m.end())
print("일치하는 문자열 시작 및 끝 인덱스 튜플 : ", m.span())

#이름이 연아인 모든 사람을 검색
nameList = ['연아','이연아', '김현수', '김연아', '박수홍']
for name in nameList :
    m = re.search('[가-힣]+연아', name)
    if m != None :
        print(m.group())

#(3) findall()
mList = re.findall('\w*berry', 'berry 1berry 10berry apple strawberry')
print(mList) #['berry', '1berry', '10berry', 'strawberry']

#a태그의 href 값들 모두 찾기
text = "<a href='www.naver.com'></a> <a href='www.daum.net'></a>"
mList = re.findall('href=\'[\w\.]+\'', text)
print(mList) #["href='www.naver.com'", "href='www.daum.net'"]

#'line'과 일치하지만 line을 포함하는 글자들은 매치되지 않도록 검색
#outline(X), linear(X), line(O)
mList = re.findall(r'\bline\b', "outline linear line line lane")
print(mList)
#-------------------------------------------------------------
#그룹캡쳐 ()
#날짜에서 년도, 월 일 각각 얻기
date = '2021-11-30'
m = re.search('(\d{4})-(\d{1,2})-(\d{1,2})',date)
print(m.group()) #매치된 전체 문자열
print(m.group(0)) #매치된 전체 문자열
print(m.group(1)) #첫번째 그룹(괄호), 년도
print(m.group(2)) #두번째 그룹(괄호), 월
print(m.group(3)) #세번째 그룹(괄호), 일
print(m.group(4)) #오류, 인덱스 벗어남

text = '<a  sdsdsd   href="www.naver.com" id="a01">hello</a>'
m = re.search('<a.*href=\"([\w\.]+)\".*id="a01"', text)
print(m.group()) #<a href="www.naver.com" id="a01
print(m.group(1)) #www.naver.com

#greedy 탐색 - 최대매칭 (기본값)
m = re.search('\d{1,3}', "123456789")
print(m.group()) # 123

#Non-greedy 탐색 - 최소매칭, 반복자 뒤에 ?를 붙여야 한다.
m = re.search('\d{1,3}?', "123456789")
print(m.group()) # 1

#greedy 탐색
liTags = "<li>나이키</li><li>아디다스</li><li>퓨마</li>"
m = re.search('<li>(.*)</li>', liTags)
m.group() #'<li>나이키</li><li>아디다스</li><li>퓨마</li>'
m.group(1) #'나이키</li><li>아디다스</li><li>퓨마'

#Non-greedy 탐색
liTags = "<li>나이키</li><li>아디다스</li><li>퓨마</li>"
m = re.search('<li>(.*?)</li>', liTags)
m.group() #'<li>나이키</li>'
m.group(1) #'나이키'

#그룹캡처를 변수처럼 활용 
text = "abc 토마토 마토토 ABC ABA XYXY"
mList = re.findall(r'((\w)(\w)\2)', text)
mList #[('토마토', '토', '마'), ('ABA', 'A', 'B'), ('XYX', 'X', 'Y')]
#\1은 첫번째 그룹, \2은 두번째 그룹, \3 세번째 그룹 ...

text = "abb 토마토 마토토 ABC ABA XYXY"
mList = re.findall(r'((\w)(\w)\3)', text)
mList #[('abb', 'a', 'b'), ('마토토', '마', '토')]

#그룹캡처를 안쓰고 싶은 경우 
#그룹안에 ?: 넣기
text = 'abc abcabc abcabcabc abbb'
mList = re.findall(r'(?:abc)+', text)
mList #['abc', 'abcabc', 'abcabcabc']

text = 'abc abcabc abcabcabc abbb'
mList = re.findall(r'((abc)+)', text)
mList #[('abc', 'abc'), ('abcabc', 'abc'), ('abcabcabc', 'abc')]

# (4) split() : 정규표현식을 기준으로 문자열을 잘라 List로 반환
text = "15.4, 1.2, 3.14, 55"
mList = re.split(r', ' , text)
mList
mList = text.split(', ')
mList

text = "15.4,       1.2,     3.14,     55,     "
mList = re.split(r',\s*' , text)
mList

#(5) sub() : 정규표현식과 매치되는 문자열을 다른 문자열로 교체
subStr = re.sub(r'\d{7}$', "*"*7, "854321-1234567")
print(subStr) # 854321-*******

subStr = re.sub(r'^(\d{6})-\d{7}$',"\g<1>-*******", 
                                    "854321-1234567")
print(subStr) # 854321-*******

phoneName = "010-1234-7777,홍길동"
#=> 이름:홍길동, 전화번호:010-1234-7777
subStr = re.sub(r'(\d{3}-\d{4}-\d{4}),(\w+)',
    "이름:\g<2>, 전화번호:\g<1>", phoneName)
print(subStr) #이름:홍길동, 전화번호:010-1234-7777

#숫자만 매칭 후 곱하기 10을 하기
#매칭결과를 lambda식 활용하여 바꾸기
text = "100 22 apple 33"
#=> "1000 220 apple 330"
                            #  x는 매칭된 데이터
subStr = re.sub(r'\d+', lambda x:str(int(x.group())*10),text)
print(subStr)

#이름을 매칭 후 님을 붙이기
names = "홍길동 임꺽정 김하나"
#=>"홍길동님 임꺽정님 김하나님"
subStr = re.sub(r'\w+', lambda x:x.group()+"님",names)
print(subStr)

#----------------------------------------------------------
# 연습문제
# (1) cat 앞 뒤에 모두 문자가 있는 경우는 가져오세요.
#ex) Scatter(O), scat(X), cat(X), copycat(X), catter(X)
text = "cat catch copycat scatter powercatter"
#=>['scatter', 'powercatter']
mList = re.findall(r'\b(\w+cat\w+)\b', text)
print(mList)

#(2) 성이 김씨인 사람을 모두 검색하시오
#한글만 가져오도록 매칭하시오
names = "김현수, 배철수, 김구, 김바른, 김사랑11"
mList = re.findall(r'김[가-힣]+', names)
mList

#(3) 휴대전화 끝자리 ****로 바꾸기
phone = '010-3711-2222'
subStr = re.sub(r'\d{4}$','****', phone)
print(subStr)

#(4) 괄호가 없는 사칙연산 수식이 있을 때
#정규표현식 2번 사용 
#숫자만 리스트로, 사칙연산 기호도 리스트로 각각 가져오시오.
text = "12*4+50-200/3+50"
#=> 숫자 ['12','3','50','200','3','50'], 기호['*','+','-','/','+']
numList = re.findall(r'\d+', text)
numList
signList = re.findall(r'\D+', text)
signList





#(5) 날짜 패턴 매칭 검사 : 날짜가 yyyy-mm-dd로 패턴에 맞는지 검사
#ex) 2021-08-19 (O), 2021-3-3 (X), 2021-04-32(X), 2021-13-19 (X)

#(6) txt, pdf, hwp, xls 파일의 확장자만 검색할 수 있도록 정규표현식
#완성하시오.
text = "abc.txt python.pdf, apple.ppt aaaa .xls"
#=> [abc.txt, python.pdf]









#(4) 괄호가 없는 사칙연산 수식이 있을 때
#정규표현식 2번 사용 
#숫자만 리스트로, 사칙연산 기호도 리스트로 각각 가져오시오.
text = "12*4+50-200/3+50"
#=> 숫자 ['12','3','50','200','3','50'], 기호['*','+','-','/','+']

#(5) 날짜 패턴 매칭 검사 : 날짜가 yyyy-mm-dd로 패턴에 맞는지 검사
#ex) 2021-08-19 (O), 2021-3-3 (X), 2021-04-32(X), 2021-13-19 (X)

#(6) txt, pdf, hwp, xls 파일의 확장자만 검색할 수 있도록 정규표현식
#완성하시오.
text = "abc.txt python.pdf, apple.ppt aaaa .xls"
#=> [abc.txt, python.pdf]


