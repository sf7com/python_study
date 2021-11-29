#(1) 예외 처리 구문 : try-except 문
for i in range(1):
    print(10/i) #ZeroDivisionError: division by zero

for i in range(10):
    try :
        print(10/i)
    except ZeroDivisionError:
        print("0으로 나눌 수 없습니다.")

for i in range(10):
    try :
        print(10/i)
    except ZeroDivisionError as e:
        print(e)
        #예외 정보 출력 , division by zero

#모든 예외 처리
for i in range(10):
    try :
        print(10/i)
    except :
        print("예외가 발생하였습니다.") #예외 정보 출력

for i in range(10):
    try :
        print(10/i)
    except Exception as e:
        print(e) #예외 정보 출력

#여러 예외 처리
numList = [0,1,2,3]
for i in range(5) :
    try :
        result = 10/numList[i]
    except ZeroDivisionError as e :
        print(e)
    except IndexError as e :
        print(e)
    else :
        #예외가 발생항지 않을 경우
        print(result)


for i in range(5) :
    try :
        result = 10/numList[i]
    except (ZeroDivisionError,IndexError) as e :
        print(e)
    else :
        #예외가 발생항지 않을 경우
        print(result)

#(2) try-except-finally문
#finally 문은 예외발생 여부가 상관없이 실행되는 코드
#일반적으로 자원을 닫을때 사용

try :
    for i in range(10) :
        result = 10/i
        print(result)
except ZeroDivisionError as e :
    print(e)
finally :
    print("프로그램 종료")

#(3) 예외발생시키기 - raise문
#raise 예외타임(예외정보)
# while True :
    # num = input("변환할 정수 입력 >")
    # if not num.isdigit(): #문자열이 숫자인지 판단 매서드 true/false
    #     raise ValueError("올바른 숫자를 입력하지 않았습니다.")
    # print('변환된 값 : ',int(num))

#(4) 예외발생시키기 - assert문
# assert 예외조건, 예외정보
print(isinstance(100,int))
print(isinstance("Hello",int))
#isinstance : 타입 체크 메서드, True/false 리턴

def get_binary_num(decimal) :
    assert isinstance(decimal, int), "정수가 아닙니다."
    return bin(decimal) #10진수를 2진수로 바꾸는 메서드
print(get_binary_num(10)) # 0b1010, 0x0A
print(get_binary_num("abc"))

try : 
    print(get_binary_num("abc"))
except Exception as e:
    print(e)
print("프로그램 종료")