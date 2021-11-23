#1.문자열
from typing_extensions import _AnnotatedAlias


str1="hello"
str2="world"

#1-1) 문자열간의 덧셈
str3 = str1+str2
print(str3)

#1-2)문자열의 곱셈
str3 = str1*5
print(str3)

#1-3) 문자열내의 검색
print('h' in str1)
print('hel' in str1)
print('w' in str1)

#1-4) 문자열 관련함수
#1) 문자열 길이
str1="hello"
print(len(str1))
#2)문자 개수 세기
print(str1.count('l'))
print(str1.count('he'))
#3)문자열 검색 : 인덱스 반환
print(str1.find('l'))
print(str1.find('el'))
print(str1.find('w')) #없으면 -1 반환
#4)대소문자 변환
print(str.upper())
print(str.lower())
#5)문자열 끝 특정 문자 제거
str1="     Hello---"
print(str1.lstrip()) #왼쪽 공백 모두 제거
print(str1.rstrip()) #오른쪽 '-' 모두 제거

str1="------hello---"
print(str1.strip()) #양쪽 '-' 모두 제거

#6) 문자열 교체
str1 = "hello world"
print(str1.replace("hello","Welcome"))
#               바뀌게될 문자열, 바꿀 문자열
#7)문자열 특정 구분자 기준 나누기(리스트로 반환)
str1 = "Life is too short"
print(str1.split())
str1="1,2,3,4,5"
print(str1.split(","))

#인덱스 슬라이싱
numList = [1,2,3,4,5,6,7,8,9,10]
print(numList[0:5]) #인덱스 0이상 5미만 인덱스(0~4)
print(numList[2:6]) #인덱스 2이상 6미만
print(numList[:6]) #인덱스 처음부터 6미만 까지
print(numList[1:]) #인덱스 1부터 끝까지
print(numList[:]) #인덱스 처음부터 끝까지

#인덱스 슬라이싱 증감값 주기
print(numList[::2]) #인덱스 0부터 끝까지 2씩 증가해서 가져오기
print(numList[1:9:3]) #인덱스 1부터 8까지 3씩 증가해서 가져오기
print(numList[::-1]) #인덱스 마지막 부터 -1씩 감소해서 가져옴
print(numList[9:0:-1]) #인덱스 9부터 1까지 -1씩 감소 가져오기
print(numList[-1:-9:-1]) #인덱스 -1부터 -8까지 -1씩 감소 

#-----------------------------------
#(1) append() : 새로운 값 리스트 맨 끝에 추가
colors = []
colors.append("White")
colors.append("red")
print(colors)

#(2) insert() : 리스트 특정 인덱스 위치에 값 삽입
colors.insert(0, 'cyan')
print(colors)

#(3) remove() : 리스트 내의 특ㅈ겅값 제거
colors.remove(0, 'White')
print(colors)

#(4) del 키워드 : 인덱스를 통한 요소 삭제
del colors[0]
print(colors)

#(5) pop() : 리스트의 맨 마지막 요소를 돌려주고 해당 요소 삭제
c= colors.pop()
print(c)
#---------------------------------------------
#리스트 내 값 수정
numList = [1,2,3]
numList[1] = 99
print(numList)

#리스트 정렬
numList = [10,8,22,1,5]
#(1) 기존 list를 정렬해서 바꿈
numList.sort()#오름차순
print(numList)
numList.sort(reverse=True)#내림차순
print(numList)
print(numList.sort())
#(2) 기존 list를 안바꾸고 정렬된 리스트를 리턴
numList = [10,8,22,1,5]
sortList = sorted(numList)#오름차순
print(numList,sortList)
sortList = sorted(numList,reverse=True)#내림차순
print(numList,sortList)
print(sorted(numList))
#------------------------------------------
#2차원 리스트 : 리스트 안에 리스트가 있는 경우
korScores = [50,60,90,44,100]
mathScores = [44,33,22,77,88]
engScores = [87,28,44,87,56]
midtermScores = [korScores,mathScores,engScores]
print(midtermScores)
print(midtermScores[0]) #국어점수 리스트 출력
print(midtermScores[1]) #수학점수 리스트 출력
print(midtermScores[2]) #영어점수 리스트 출력
#0번째 학생 점수
print(midtermScores[0][0]) #0번째 학생 국어점수
print(midtermScores[1][0]) #0번째 학생 수학점수
print(midtermScores[2][0]) #0번째 학생 영어점수
#마지막 학생 점수
print(midtermScores[0][-1]) #마지막 학생 국어점수
print(midtermScores[1][-1]) #마지막 학생 수학점수
print(midtermScores[2][-1]) #마지막 학생 영어점수
#2번째 학생 점수
print(midtermScores[0][4]) #4번째 학생 국어점수
print(midtermScores[1][4]) #4번째 학생 수학점수
print(midtermScores[2][4]) #4번째 학생 영어점수

#----------------------------------------
#파이썬 리스트의 특수성
#(1)언패킹:List같은 컬렉션 데이터들을 각각 변수에 할당
colors = ['red','blue','green']
c1,c2,c3 = colors
print(c1,c2,c3)

c1,c2,c3,c4 = colors #오류 : 리스트의 개수와 변수개수가 다름

#(2) 리스트내의 다양한 데이터 자료형 허용
list1 = ['red',1,3.14,[1,2,3]]
print(list1)
print(list1[0])
print(list1[-1])

#-------------------------------------
#문자열과 리스트의 유사성
#문자열 인덱스
str1 = "hello world"
print(str1[0])
print(str1[1])
print(str1[-1])
print(str1[0:6])
print(str1[::-1])

#문자열 내 특정 문사 검색
#리스트 내 특정 요소 검색
list1 = ['h','e','l','l','o']
print('h' in str1)
print('h' in list1)

#문자열 덧셈,곱셈
#리스트 덧셈,곱셈
str1 = "hello"
str2 = "world"
print(str1+str2)
list1 = [1,2,3]
list2 = [4,5,6]
print(list1+list2)
print(str1*3)
print(list1*3)

#크기 구하기
print(len(str1)) #문자열 길이
print(len(list1)) #리스트 요소수

#문자열 개별 문자 -> 리스트화
print(list(str1))
#리스트내의 문장열 요소 -> 문자열로 합치기
list1 = ['h','e','l','l','o']
print(''.join(list1)) #hello
print('-'.join(list1)) #h-e-l-l-o