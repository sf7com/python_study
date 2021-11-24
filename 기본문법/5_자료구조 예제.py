#1. List 예제
numList = [1,2,3,1,2,1,1,3,3,3]
#(1) 위 리스트의 합계
sum1=0
for num in numList:
    sum1 += num
print(f"리스트의 합계 {sum1}")
print(f"리스트의 합계 {sum(numList)}")
#(2) 위 리스트의 평균
print(f"리스트의 평균 {sum1/len(numList)}")
print(f"리스트의 평균 {sum(numList)/len(numList)}")
#(3) 위 리스트의 최빈값 (가장 많이 등장하는 값)
from collections import Counter
numCounter = Counter(numList)
print(numCounter)
print(numCounter.most_common(1)) #value값이 가장 큰 1개
print(numCounter.most_common(3)) #value값이 가장 큰 3개

setnumList = set(numList)
maxcnt = 0
maxNum = 0
for i in setnumList :
    cnt = numList.count(i)
    if cnt > maxcnt:
        maxcnt = cnt
        maxNum = i
print(maxNum, maxcnt)

#(4) 위 리스트의 중앙값
   # 위 리스트를 오름차순으로 정렬 후 
   # N은 리스트의 크기라고 할때 다음과 같이 중앙값을 구한다.
   # 리스트의 개수가 홀수인경우 : 중앙값 인덱스 int(n/2)
   # 리스트의 개수가 짝수인경우 : 중앙값 인덱스 int(n/2),int(n/2)-1
   # 두 인덱스 값들의 평균
numList.sort()
ctnum = 0
if len(numList)%2 != 0 :
    print(f"중앙값 {numList[len(numList)/2]}")
else :
    ctnum = (numList[int(len(numList)/2)]+numList[int(len(numList)/2)-1])/2 
    print(f"중앙값 : {ctnum}")

#2. set예제
#다음 9~11월 까지 헬스장에 나간 날짜들이다.
dataList = ['09/10','09/11','09/12','10/01','10/03','11/20']
#달별 이용 횟수를 구하시오.
monList = []
for mon in dataList :
    monList.append(mon[0:2])
print(monList)
moncnt = set(monList)
print(monList)
mondic = dict()
for val in moncnt :
    mondic[val] = monList.count(val)
mon_sort = dict(sorted(mondic.items(), key=lambda x: x[0]))
print(mon_sort)
print(Counter(monList))

#3.dict 예제
memberDic = {}
memberDic['id01'] = ['홍길동',30,'수원시']
memberDic['id02'] = ['임꺽정',40,'전주']
memberDic['id03'] = ['김하나',25,'서울']
memberDic['id04'] = ['김두한',60,'서울']
#3-1) 서울에 사는 사람들을 출력하시오.
for key,val in memberDic.items() :
    if '서울' in val[2]:
        print(key,val)
#3-2) 등록된 회원들의 나이 평균값을 구하시오.
avgage = 0
for val in memberDic.values():
    avgage += val[1]
print(f"평균나이 : {avgage/len(memberDic)}")
