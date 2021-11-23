#1) 기본자료형
num1=1
num2=2
print(num1+num2)

floatNum1 = 3.14
floatNum2 = 1.25
print(floatNum1+floatNum2)

str1 = "Hello"
str2 = "World"
print(str1+str2)

bool1 = True
bool2 = False
print(bool1)
print(bool1) #ALT+SHIFT+↓ 한줄 복사
##############################################
#2) 기본연산
num1=10
num2=20
print(num1+num2)
print(num1-num2)
print(num1*num2)
print(num1/num2)
num1,num2 = 2,3
#제곱연산
print(num1**num2)
#몫 연산
print(num2//num1)
#나머지 연산
print(num2%num1)
#증감연산자
num1+=1 #num++ 못함
num2+=3
print(num1,num2)
####################################
#3) 자료형 변환
#3-1) 정수와 실수간 형변환
num1,num2 = 10,3
print(num1/num2) #자동형변환 -> 실수
#어떤값을 정수로 바꿀때 int
print(int(num1/num2))
#어떤값을 실수로 바꿀때 float
print(float(num1/num2))

#3-2) 숫자형과 문자형 간 변환
floatStr1 = "3.14"
intStr1 = '10'

print(float(floatStr1))
#print(int(floatStr1)) #오류
print(int(float(floatStr1)))
print(float(intStr1))
print(int(intStr1))

#숫자 -> 문자열
floatNum = 3.14
intNum = 5
print(str(floatNum), str(intNum))

#------------------------------------
#4) 자료형 확인
num1=10
num2=3.14
numstr="58.11"
print(type(num1))
print(type(num2))
print(type(numstr)) 
