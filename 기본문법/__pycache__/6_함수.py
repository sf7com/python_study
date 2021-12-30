# (1) 원의 넓이를 구하는 함수
def getCircleArea(r):
    return r*r*3.14

# (2) 자기 소개를 출력하는 함수
# 반환 데이터가 없는 함수


def say_myself(name, old, man):
    print(f'나의 이름은 {name}입니다.')
    print(f'나이는 {old}입니다.')
    if man:
        print("남자입니다.")
    else:
        print("여자입니다.")


r = 10
print(f"반지름 {r}의 넓이 : {getCircleArea(r)}")
say_myself("홍길동", 30, True)
# -------------------------------------------------------
# 함수의 매개변수 디폴트 값 설정


def say_myself(name, old=20, man=True):
    print(f'나의 이름은 {name}입니다.')
    print(f'나이는 {old}입니다.')
    if man:
        print("남자입니다.")
    else:
        print("여자입니다.")


say_myself("홍길동", 25, True)
say_myself("홍길동")
# 파라미터의 이름을 통한 값 대입
say_myself(old=30, name="임꺽정", man=False)
# ------------------------------------------------
# 함수의 호출방식
# (1) 값에 의한 호출 (call by value)
def testFunc(num):
    num = 3

num = 10
testFunc(num)
print(num)

# (2) 참조에 의한 호출 (call by reference)
def testFunc2(numList):
    numList.append(10)
    numList[0] = -100
    numList = [1, 2, 3]

numList = [1, 2, 3]
testFunc2(numList)
print(numList)

# (3) 함수내에서 함수밖의 변수값 변경
def testFunc3():
    global num
    num = -100


num = 10
testFunc3()
print(num)
# -----------------------------------------------------
# 함수의 가변인수
# 가변인수 : 매개변수의 개수가 정해져 있지 않고 사용하는 인수
def testFunc4(*arg) :
    print(arg)
    #(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    return sum(arg)

#일반 매개변수와 가변인수가 같이 사용될 경우,
#가변인수를 마지막 인수로 넣는다.
def testFunc4(num1,num2,*arg) :
    print(num1, num2, arg)
    #(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    return sum(arg)+num1+num2
print(testFunc4(1,2,1,2,3))

#리스트 언팩킹
numList = [1,2,3,4,5]
#언패킹되는 변수의 개수와 리스트 요소수는 일치해야한다.
n1,n2,n3,n4,n5 = numList
print(n1,n2,n3,n4,n5)

# 언패킹되는 변수의 개수와 리스트 요소수가 다를 경우
# *변수 => 리스트로 패킹됨
n1,n2,*ns= numList
print(n1,n2,ns)

print(*numList) #언패킹
print(numList) 

def testFunc6(*arg) :
    x,y,*z = arg
    print(x,y,z)
    return x,y,z
print(testFunc6(1,2,3,4,5))

#키워드 가변인수
def testFunc7(**kwargs) :
    print(kwargs)
    print(kwargs.keys())
    print(kwargs.values())
    print(f"첫 요소 : {kwargs['first']}")
    print(f"두번째 요소 : {kwargs['second']}")
    print(f"세번째 요소 : {kwargs['third']}")
testFunc7(first=3, second=4, third=5)

#----------------------------------------------
#재귀함수 : 자기자신을 다시 호출하는 함수

def recursive_f() :
    print("재귀 함수를 호출합니다.")
    recursive_f()
recursive_f()
#재귀함수는 반드시 종료조건을 명시해야 한다.

def recursive_f2(i) :
    # 100번째 호출 했을 때 종료되도록 조건
    if i==100 :
        return
    print(i, '번째 재귀함수에서', i+1, '번째 재귀함수 호출합니다.')
    recursive_f2(i+1)
    print(i, '번째 재귀함수를 종료합니다.')

recursive_f2(1)

#팩토리얼 예제
#n! = 1*2*3*4*....*(n-1)*n
#5! = 1*2*3*4*5
#수학적으로 0!, 1! 값이 1이다.

#반복문을 활용한 팩토리얼 함수
def factorial(n) :
    result = 1
    for i in range(1, n+1) :
        result *=i
    return result
print(factorial(5))

#재귀함수로 구현한 팩토리얼 함수
def factorial2(n) :
    if n <= 1:
        return 1
    return n*factorial2(n-1)
result = factorial2(10000)
#5*factorial2(4)
#5*4*factorial2(3)
#5*4*3*factorial2(2)
#5*4*3*2*factorial2(1)
#5*4*3*2*1
print("\n", result)

#재귀 함수를 잘 활용하면 복잡한 알고리즘을 간결하게 작성할 수 있다.
#모든 재귀함수 반복문을 이용해서 동일한 기능을 구현할 수 있다.
#재귀함수가 반복문 보다 유리한 경우도 있고 불리한 경우도 있다.

#피보나치 수열
def fibo(n) :
    if n==1 or n==2 :
        return 1
    else :
        return fibo(n-1) + fibo(n-2)
print(fibo(3))
print(fibo(4))
#fibo(3)+fibo(2)
#(fibo(2)+fibo(1)) + fibo(2)
#(1+1) + 1
#3