#자료구조 (Collection, 자바기준)
#데이터를 관리하고 저장하는 방식
#대용량일수록 메모리아 빨리 저장하고 빠르게 검색하여 효율적으로 
#사용하고 실행시간을 줄인다.

#1.스택 - 나중에 들어온 값이 먼저 나가는 자료구조(LIFO)
#리스트로 구현 가능
numList = [1,2,3,4,5]
numList.append(10)
numList.append(2)
print(numList) 
print(numList.pop()) #2가 출력
print(numList) #2가 없어짐
print(numList.pop()) #10이 없어짐
print(numList)#10이 없어짐

#2.큐 - 먼저 들어온 값이 먼저 나가는 자료구조(FIFO)
numList = [1,2,3,4,5]
numList.append(10)
numList.append(2)
print(numList) 
print(numList.pop(0)) #1가 출력
print(numList) #1가 없어짐
print(numList.pop(0)) #2이 없어짐
print(numList)#2이 없어짐

#3 튜플 - 데이터의 변경을 허용하지 않는 자료구조
#상수로 사용, 리스트와 유사하다.
#기호()

#3-1) 튜플 선언
t1 = (1,2,3)
print(t1)
fruits = ["사과","망고","딸기"]
print(fruits)
t2 = tuple(fruits) #리스트를 튜플로 전환
print(t2)

#3-2) 인덱스를 통한 값 참조
print(t1[0], t2[-1])

#3-3) 덧셈, 곱셈 연산
t3=t1+t2 #튜플끼리 합친다.
print(t3)
#3-4) 튜플 크기
print(len(t1))
#3-5) 튜플 값 수정 - 오류남
t1[0] = 10 #오류
#3-6) 값을 우회적으로 바꾸는 방법
t4 = (0,[1,2,3],"hello")
t4[1] = [5,6,7]
t4[1][0]=5
t4[1][1]=6
t4[1][2]=7
print(t4)

#4 셋 : 데이터의 중복을 허용하지 않고, 순서가 없는 자료구조
#4-1)선언
s1 = {1,2,3}
s1.add(1) #셋에 값 추가
s1.add(2)
s1.add(5)
print(s1) #중복 없이 저장

#리스트를 셋으로 변경
numList = [1,2,3,3,3]
s2 = set(numList)
print(s2)

#문자열을 셋으로 변경
s3 = set("hello")
print(s3)

#4-2) set 관련 메서드
s4={1,2,3}
s4.add(4) #값 추가
s4.update([4,5,6]) #값 여러개 추가
s4.update({7,8})#값 여러개 추가
print(s4)
s4.remove(4) #값 제거
print(s4)

#4-3) 집합연산
fruit1 = {'사과','딸기','포도'}
fruit2 = {'배','딸기','귤'}

#합집합
fruit = fruit1 | fruit2
print(fruit)

#교집합
fruit = fruit1 & fruit2
print(fruit)

#차집합
fruit = fruit1 - fruit2
print(fruit)
fruit = fruit2 - fruit1
print(fruit)

#5.딕셔너리
#key와 value로 이루어진 자료구조
#기호 {}
#5-1) 선언
person = {'name':'홍길동', 'age':31,'birth':'0208'}
print(person)
print(person['name'])#키값을 통한 값 참조
print(person['age'])
print(person['birth'])

person2 = dict() #dict 객체 선언
person2['name'] = '임꺽정' #값추가
person2['age'] = 50
person2['birth'] = '0817'
print(person2)

#5-2) 값 얻기
name = person['name']
print(name)
name = person.get('name')
print(name)

addr = person['addr'] #키값이 없어서 오류
addr = person.get('addr') #키값이 없어도 오류가 안남
print(addr) #None객체

#키값이 없을때 default 값을 넣을 수 있음
addr = person.get('addr','대한민국')
print(addr)
#딕셔너리는 get을 통해 값을 가져오면 안정적으로 값을 가져온다.

#5-3) 값 수정
person['age'] = 40
print(person)

#5-4) 데이터 삭제
del person['birth']
print(person)

#모든 요소 삭제
person.clear()
print(person)

#5-5) 딕셔너리 메서드
#dict_keys 객체로 키값들 반환
print(person2.keys()) #키값 가져옴
#dict_values 객체로 값들 반환
print(person2.values()) #값 가져옴

#위 객체들을 List로 변경시
print(list(person2.keys()))
print(list(person2.values()))

#5-6) 키 및 값 검색
#키값 검색
'name' in person2
'addr' in person2
#값 검색
'임꺽정' in person.values()
'홍길동' in person.values()
