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

#이름이 강동원인 사람 정보 출력
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

with open('./기본문법/data/memberAlter.txt','w',encoding='utf-8') as f:
    keyList = memberList[0].keys()
    text = ",".join(keyList)+"\n"
    f.write(text)
    for member in memberList :
        # valList = [member[key] for key in keyList]
        valList = member.values()
        text = ','.join(valList)+"\n"
        print(text)
        f.write(text)

#a모드(append모드) 파일 끝에 내용들ㄹ을 추가하여 파일쓰기
#w모드(write 모드) 파일 처음에 내용들을 쓴다.(기존내용 삭제)
with open('./기본문법/data/memberAlter.txt','a',encoding='utf-8') as f: 
    keyList = memberList[0].keys()
    text = ",".join(keyList)+"\n"
    f.write(text)
    for member in memberList :
        # valList = [member[key] for key in keyList]
        valList = member.values()
        text = ','.join(valList)+"\n"
        print(text)
        f.write(text)

#---------------------------------------------------------------------
#pickle - 객체 저장 및 불러오기
import pickle
memberList
#객체저장
with open('./기본문법/data/memberList.pickle','wb') as f:
    pickle.dump(memberList,f)
#저장된 객체 불러오기
with open('./기본문법/data/memberList.pickle','rb') as f:
    loadData = pickle.load(f)

for val in loadData :
    print(val)    

#--------------------------------------------------
#수원시 행정구역 폴더 만들기
import os
path = './기본문법/data/수원'
# os.makedirs(path+'/수원')

#이미 폴더가 있는 경우 폴더만들기 하면 오류발생

folderPath = path+'/수원'
if not os.path.exists(folderPath) :
    os.makedirs(folderPath)

with open('./기본문법/data/수원시행정구역.csv', 'r', encoding='euc-kr') as f :
    lines = f.readlines() #한줄씩 읽어 리스트로 반환
    guList = lines[0].strip('\n').split(',')
    print(guList) #['장안구', '권선구', '팔달구', '영통구']
    for line in lines[1:]:
        dongList = line.strip('\n').split(',')
        for gu,dong in zip(guList,dongList) :
            dongPath = folderPath + "/" + gu + "/" +dong
            if not os.path.isdir(dongPath) and dong !="":
                os.makedirs(dongPath)