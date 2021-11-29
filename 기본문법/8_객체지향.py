class SoccerPlayer :
    
    #생성자 함수
    def __init__(self,name,position,backNum) :
        #필드값 정의
        self.name = name
        self.__position = position
        self.__backNum = backNum
        #필드변수명 앞에 __을 붙이면 private
        #필드변수명 앞에 _을 붙이면 protected
    def changeBackNum(self, newNum):
        self.__backNum = newNum

#print에 객체를 넣을 때 출력되는 문자열
#Java 기준 toString() 메서드와 같다
    def __str__(self) :
        return f"이름 :{self.name}, 포지션 :{self.__position}, 번호 :{self.__backNum}"

# 객체생성
player1 = SoccerPlayer("손흥민","MF",10)
print("player1의 등번호 : ", player1.__backNum)
player1.changeBackNum(5)
print("player1의 등번호 : ", player1.__backNum)
print(player1)

player1.name = "메시"
print(player1)

#private 필드의 값을 참조하고 싶은 경우
#클래스명__변수명
player1.__SoccerPlayer__backNum = 3
print(player1)
print(player1.__SoccerPlayer__backNum)

#---------------------------------------
# 여러 객체를 List로 할당
names = ['메시','박지성','손흥민','호날두']
posList = ["MF","DF","CF","WF"]
backNums = [10,4,6,3]

playerList = [SoccerPlayer(name,pos,backNum) for name,pos,backNum in zip(names,posList,backNums)]
print(playerList)
for p in playerList :
    print(p)

# 모든 선수들 번호 1씩 증가
for p in playerList :
    p.changeBackNum(p._SoccerPlayer__backNum+1)
    print(p)