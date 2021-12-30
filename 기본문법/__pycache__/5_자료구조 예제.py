#1. list 예제
numList = [1,2,3,1,2,1,1,3,3,3]

#(1) 위 리스트의 합계
total = 0
for val in numList :
    total += val
print(f"리스트의 합계 {total}")
print(f"리스트의 합계 {sum(numList)}")
#(2) 위 리스트의 평균
print(f"리스트의 합계 {total/len(numList)}")
print(f"리스트의 합계 {sum(numList)/len(numList)}")
#(3) 위 리스트의 최빈값 (가장 많이 등장하는 값)
from collections import Counter
import collections
numCounter = Counter(numList)
print(numCounter)
print(numCounter.most_common(1)) #value값이 가장 큰 1개
print(numCounter.most_common(3)) #value값이 가장 큰 순서대로 3개

numSet = set(numList)
maxCnt = 0
maxNum = 0
for num in numSet :
    cnt = numList.count(num)
    if cnt > maxCnt :
        maxCnt = cnt
        maxNum = num
print(maxNum, maxCnt)

for num in numSet :
    cnt = numList.count(num)
    if cnt == maxCnt :
        print(num, cnt)

#(4) 위 리스트의 중앙값
    #위 리스트를 오름차순으로 정렬후
    #N은 리스트의 크기라고 할때 다음과 같이 중앙값을 구한다.
    #리스트의 개수가 홀수인경우 : 중앙값 인덱스 int(N/2)
    #리스트의 개수가 짝수인경우 : 
    #   중앙값 인덱스 int(N/2), int(N/2)-1
    #   두 인덱스 값들의 평균
numList.sort() #오름차순정렬
size = len(numList) #리스트 크기
if size % 2 == 0 : #리스트 개수가 짝수인 경우
    midVal = (numList[int(size/2)]+numList[int(size/2)-1])/2
    print(f"중앙값 : {midVal}")
else :
    print(f"중앙값 : {numList[int(size/2)]}")

#2. set 예제
#다음은 9~11월 까지 헬스장에 나간 날짜들이다.
dateList = ['09/10','09/11','09/12','10/01','10/03','11/20']
#달별 이용 횟수를 출력하시오.
monList = []
for date in dateList :
    monList.append(date[0:2])
print(monList) #['09', '09', '09', '10', '10', '11']
monSet = set(monList)
print(monSet) #{'10', '09', '11'}
mondic = dict()
for val in monSet :
    mondic[val] = monList.count(val)
print(mondic) #{'10': 2, '09': 3, '11': 1}
print(Counter(monList)) #Counter({'09': 3, '10': 2, '11': 1})

#3. dict 예제
memeberDic = {}
memeberDic['id01'] = ['홍길동', 30, '수원시']
memeberDic['id02'] = ['임꺽정', 40, '전주시']
memeberDic['id03'] = ['김하나', 25, '서울']
memeberDic['id04'] = ['김두한', 60, '서울']
#3-1) 서울에 사는 사람들을 출력하시오.
for key,val in memeberDic.items() :
    if '서울' in val[2] :
        print(key, val)
#3-2) 등록된 회원들의 나이 평균값을 구하시오
totalAge = 0
for val in memeberDic.values() :
    totalAge += val[1]
print(f'평균나이 : {totalAge/len(memeberDic)}')






