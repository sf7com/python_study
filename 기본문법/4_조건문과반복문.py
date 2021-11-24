#1.조건문

# score = int(input("점수를 입력하세요>"))
# if score >=90:
#     grade = "A"
# elif score >=80:
#     grade = "B"
# elif score >=70:
#     grade = "c"
# elif score >=60:
#     grade = "D"
# else :
#     grade = "F"
# print(f"{score}의 학점 : {grade}")

# birthyear = input("태어난 년도 입력>")
# age = 2021 - int(birthyear) +1

# if age<=26 and age>20 :
#     print("대학생")
# elif age<20 and age>=17 :
#     print("고등학생")
# elif 14<=age<17 :
#     print("중학생")
# elif 8<=age<14 :
#     print("초등학생")
# else : 
#     print("아동")

#년수를 입력받아서 평년,윤년, 구하는 예제
#(공식)①,② 두조건을 다 만족해야 윤녀임
#① 년도를 4로 나누어 떨어져야 함
#② 년도를 100으로 나누어 떨어지지 않거나 년도를 400으로 나누어 떨어져야 함
#ex) 4(윤년), 100(평년), 400(윤년)
# and 연산자, or 연산자 활용

# year = int(input("년도를 입력하세요> "))

# if year%4 == 0 and (year%100 != 0 or year%400 == 0):
#     print(f"{year}은(는) 윤년입니다.")
# else:
#     print(f"{year}은(는) 평년입니다.")

#---------------------------------------
#3항 연산자
#java기준 : int num = (조건)? a:b ;
#파이썬 : num = a if 조건 else b

# num = 15

# str1 = "짝수" if num % 2 == 0 else "홀수"
# print(str1)

# year = 2024
# str1 = "윤년" if(year%4==0) and (year%100!=0 or year%400==0) else "평년"
# print(str1)

#---------------------------------
#2. 반복문
#2-1) 증감값을 통한 반복문
#java 기준 for(int i=0; i<반복횟수; i++)와 비슷

#형태
#for 변수 in range(시작가밧, 마지막값,증감값):
# for i in range(1,10) : #1부터 10미만 1씩 증가
#     print(i, end=',')

# for i in range(1,10,2) : #1부터 10미만 2씩 증가
#     print(i, end=',')

# for i in range(10,2,-1) : #10부터 0초과 1씩 감소
#     print(i, end=',')

# for i in range(100) : #0부터 100미만(0~99)
#     print(i)

#2-2) 리스트와 반복문
# java기준 - for each 구문과 비슷

numList = [1,2,3,4,5,6,7,8,9,10]
for i in numList :
    print(i, end=',')

#List의 인덱스 참조하기 위한 For문
for i in range(len(numList)):
    print(numList[i], end=',')

#numList의 요소가 짝수인 경우 값을 0으로 대입
# for i in range(len(numList)):
#     if numList[i]%2 == 0:
#         numList[i] = 0
# print(numList)

for i,val in enumerate(numList) : #enumerate (나열하다), 인덱스와 값을 나역해서 변수에 대입
    print(f"인덱스:{i}, 값:{val}")

for i,val in enumerate(numList) :
    if val % 2 == 0:
        numList[i] = 0
print(numList)

#2-3) 문자열과 반복문
for ch in "hello":
    print(ch)

str1=""
for ch in "hello"[::-1]:
    str1+=ch
print(str1)

#00:00:00 ~ 23:59:59초 까지 3이 몇개 들어있는지 개수 세기

cnt = 0
for hour in range(24):
    for min in range(60):
        for sec in range(60):
            #print(f'{hour}:{min}:{sec}')
            cnt += (str(hour)+str(min)+str(sec)).count('3')
print(f"3의 개수{cnt}")

#숫자 4각형 출력1
#01 02 03 04 05
#06 07 08 09 10
#11 12 13 14 15
#16 17 18 19 20
row,col=4,5
cnt = 0
for i in range(col):
    for j in range(row):
        cnt += 1
        if cnt % col == 0:
            print("%02d" %(cnt))
        else:
            print("%02d" %(cnt),end=" ")

for r in range(1,row+1):
    for c in range(1,col+1):
        print(f'{col*(r-1)+c:0>2}', end=' ')
    print()
            

#숫자 4각형 출력2
#01 02 03 04 05
#10 09 08 07 06
#11 12 13 14 15
#20 19 18 17 16
row,col=4,5

for r in range(1,row+1):
    if r%2==1:
        for c in range(1,col+1):
            print(f'{col*(r-1)+c:02}', end=' ')
    else :
        for c in range(col,0,-1):
            print(f'{col*(r-1)+c:02}', end=' ')
    print()

#직각삼각형
height = 10
for i in range(1,height+1):
    print("*"*i)
#공백이 먼저 출력되는 직각삼각형
for i in range(1,height+1):
    print(" "*(height-1) + "*"*i)
#이등변 삼각형
for i in range(1,height+1):
    print(" "*(height-i)+"*"*(2*i-1))

#다이아몬드
for i in range(1,height+1):
    print(" "*(height-i)+"*"*(2*i-1))
for i in range(height-1,0,-1):
    print(" "*(height-i)+"*"*(2*i-1))

#2-4) while 문
i=1
while i<10:
    print(i)
    i+=1

#-------------------------
#반복문 제어
#1) break
#2) continue

for i in range(10):
    if i==5:
        break
    print(i)

#짝수만 출력
for i in range(10):
    if i%2 ==1 :
        continue
    print(i)

#3) for ~ else 문
#else : 반복문이 완전히 끝났을 때 마지막에 실행함
#잘 안쓰임
for i in range(10):
    print(i)
else :
    print("프로그램 종료")
#else문 실행안함
for i in range(10):
    if i==5:
        break
    print(i)
else :
    print("프로그램 종료")
