#(1) 파일 읽기
f = open('./기본문법/data/회원정보.txt', 'r',encoding='utf-8')
text = f.read() #read() 파일내용을 모두 읽어 문자열로 반환
print(text)
f.close()

#with 키워드를 통한 파일 열기
#파일 닫기를 안해도 된다.
with open('./기본문법/data/회원정보.txt', 'r', encoding='utf-8') as f :
    lines = f.readlines() #한줄씩 읽어 리스트로 반환

for i,line in enumerate(lines):
    print(i,line.strip('\n'))
print(lines)

#id키값, value 나머지 정보를 list에서 저장
#[{'ID':'id01','PW':'1234','NAME':'김동현'},
# {'ID':'id02','PW':'1234','NAME':'이현수'},
# {'ID':'id03','PW':'1444','NAME':'강동원'},]

memberList = [] #모든 회원정보를 담는 리스트
with open('./기본문법/data/회원정보.txt', 'r', encoding='utf-8') as f :
    lines = f.readlines() #한줄씩 읽어 리스트로 반환
    keyList = lines[0].strip('\n').split(',')
    print(keyList) #['ID', 'PW', 'Name', 'Phone', 'Address']
    for line in lines[1:]:
        valList = line.strip('\n').split(',')
        memberDic = {}
        for key,val in zip(keyList,valList) :
            memberDic[key] = val
        memberList.append(memberDic)
print(memberList)

#수원시에 거주하는 사람 정보 출력
for member in memberList:
    if '수원시' in member['Address']:
        print(member)

#이름ㄹ이 강동원인 사람 정보 출력
for member in memberList:
    if '강동원'==member['Name']:
        print(member)

#id가 id02사람의 주소를 부산광역시 해운대구 해운대동으로 수정
for member in memberList:
    if 'id02'==member['ID']:
        member['Address'] = '부산광역시 해운대구 해운대동'
print(memberList)

#------------------------------------------------------------
#파일 쓰기
with open('./기본문법/data/파일쓰기.txt','w',encoding='utf-8') as f:
    for i in range(1,11):
        f.write(f'{i}번째 줄 입니다.\n') #한줄씩 파일 쓰기