#(1) 원의 넓이를 구하는 함수
def getCircleArea(r):
    return r*r*3.14

#(2) 자기 소개를 출력하는 함수
#반환 데이터가 없는 함수
def say_myself(name,old,man):
    print(f'나의 이름은 {name} 입니다.')
    print(f'나의 나이는 {old} 입니다.')
    if man :
        print('남자입니다.')
    else :
        print('여자입니다.')

r=10
print(f"반지름 {r}의 넓이 : {getCircleArea(r)}")
say_myself("홍길동",30,True)

