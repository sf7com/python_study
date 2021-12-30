class Person :
    def __init__(self, name, age, gender) :
        self.name = name
        self.age = age
        self.gender = gender
    def introduce(self) :
        print(f"이름 : {self.name}")
        print(f"나이 : {self.age}")
        print(f"성별 : {self.gender}")

class Employee(Person) :
    def __init__(self, id, dept, salary, name, age, gender) :
        #부모의 생성자 메서드 호출
        super().__init__(name, age, gender)
        self.id = id
        self.dept = dept
        self.salary = salary
    def introduce(self):
        print(f"ID : {self.id}")
        #부모의 introduce() 메서드 호출
        super().introduce()
        print(f"부서 : {self.dept}")
        print(f"봉급 : {self.salary}")
    #파이썬 특수 메서드들
    #해시코드 정의
    def __hash__(self) :
        #이름과 id가 같으면 같은객체라고 인식하기 위함
        return hash(self.id) + hash(self.name)
    #항등연산자 ==에 대한 동작 정의
    def __eq__(self, other) :
        return (self.id==other.id) and (self.name==other.name)
    #부등호 연산자 !=에 대한 동작정의
    def __ne__(self, other) :
        return not self.__eq__()
    #보다 큼 연산자, > 에 대한 동작 정의
    def __gt__(self, other) :
        return self.salary > other.salary
    #보다 크거나 같음 연산자, >= 에 대한 동작 정의
    def __ge__(self, other) :
        return self.salary >= other.salary
    #보다 작음 연산자, < 에 대한 동작 정의
    def __lt__(self, other) :
        return not self.__ge__()
    #보다 작거나 같음 연산자, <= 에 대한 동작 정의
    def __le__(self, other) :
        return not self.__gt__(other)
    #len 함수 정의
    def __len__(self) :
        return len(self.name)
    #print 문자열 출력 정의    
    def __str__(self) :
        return f"아이디:{self.id}, 이름:{self.name}"
    #객체가 컬렉션 안에 있을 때 출력될 내용
    def __repr__(self) :
        return f"({self.__str__()})"

    
