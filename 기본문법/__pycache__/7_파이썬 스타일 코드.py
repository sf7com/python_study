#파이썬의 독특한 문법
#(1) 리스트 컴프리헨션(List comprehension)
#기존 리스트를 이용해 새로운 리스트를 만든다.
#리스트와 for문을 한 줄에 사용할 수 있는 장점이 있다.

#반복문을 활용하여 0~9까지 숫자 List만들기
result = []
for i in range(10) :
    result.append(i)
print(result)

#리스트 컴프리헨션을 이용하여 만들기
result = [i for i in range(10)]
print(result)

#0~9까지 짝수만 담는 리스트
result = [i for i in range(0,10,2)]
print(result)

#조건문을 통한 값 필터링
result = [i for i in range(10) if i%2==0]
print(result)

result = [] 
for i in range(10) :
    if i%2==0 :
        result.append(i)
print(result)    

#3항 연산자 활용
#짝수는 1, 홀수는 0
result = [1 if i%2==0 else 0 for i in range(10)]
print(result)

result = []
for i in range(10) :
    num = 1 if i%2==0 else 0
    result.append(num)
print(result)

#알파벳 리스트 a~z
alphaList = [chr(ord('a')+i) for i in range(26)]
print(alphaList)
#ord() : 문자를 아스키코드로 반환
#chr() : 아스키 코드를 문자로 반환

#숫자리스트의 값을 제곱으로 해서 새로운 리스트 만들기
numList = [1,2,3,4,5]
numList = [num*num for num in numList]
print(numList)

#과일명 및 과일 개수를 튜플 형태로 리스트로 만들기
fruits = ['사과','사과','딸기','포도','사과','딸기']
fruitsNCnt = [(f, fruits.count(f)) for f in set(fruits)]
print(fruitsNCnt)

fruitsNCnt = [] 
for f in set(fruits) :
    fruitsNCnt.append((f, fruits.count(f)))

#zip함수를 이용한 집계 결과 리스트 만들기
maList = [80,20,50,90] #학생별 수학점수
koList = [90,10,50,77] #학생별 국어점수
enList = [66,31,87,40] #학생별 영어점수
stuStatics = [(sum(scores),sum(scores)/len(scores))
    for scores in zip(maList, koList, enList)]
print(stuStatics) #학생별 총점과 평균값 튜플값이 리스트에 저장

