#from 패키지명 import 불러올 클래스 이름
from 기본문법.person import Person, Employee
#from 패키지명 import * 해당 패키지에 있는 모든 클래스 불러옴
#from 기본문법.person import *

p1 = Person('홍길동', 30, '남자')
p2 = Person('임꺽정', 55, '남자')
p1.introduce()
p2.introduce()
print(p1) #__str__정의를 안해서 객체명 및 주소 출력

e1=Employee(1,"IT", 300, "홍길동", 30,"남자")
e2=Employee(1,"IT", 400, "홍길동", 35,"여자")
e3=Employee(1,"경영", 280, "김하나", 33,"여자")
print(e1)
print(e2)
print(e3) # __str__()에 있는 메서드
e1.introduce()

print(e1==e2) #이름과 ID같아야 동일한 객체라고 판단
print(e2==e3) #__eq__에 정의된 리턴

employSet = set()
employSet.add(e1)
employSet.add(e2)
employSet.add(e3)
print(employSet) #출력되는 것은 __repr__()에 있는 메서드
#{(아이디:1, 이름:홍길동), (아이디:1, 이름:김하나)}

print(e1>e2) # 300>400
print(e2>e3) # 400>280

print(len(e1))