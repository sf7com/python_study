# 1) 표준 입력
print("Enter Your Name : ")
name = input()
print("My Name is", name)

# 1-1) 입력 -> 실수
# input()은 문자열로 입력받는다.
temp = float(input("온도를 입력하세요."))
print("온도 :", temp * 10)
#########################################################################################################################################################################################################
# 2) 표준 출력
print("hello") # 자동 개행
print("world")

print("hello", end='') # 개행 안함
print("world")

print("hello", end='------') # 개행 안함
print("world")

print('hello','world','python')
print('hello'+'world'+'python')

num=10
num2= 3.14
print("hello", num, num2) # 잘 출력
# print("hello" + num + num2) # 출력 안됨
print("hello" + str(num) + str(num2))

#########################################################################################################################################################################################################
# 문자열에 변수를 넣는 3가지 방법
#(1) % 포매팅
name = "홍길동"
age = 30
height = 185.3
print("이름 : %s, 나이 : %d" %(name, age))
print("이름 : %s, 나이 : %d, 키 : %f" %(name, age, height))
#(1-1) % 포맷팅 자릿수 지정
print("이름 : %10s, 나이 : %4d, 키 : %2.1f" %(name, age, height))
print("이름 : %-10s, 나이 : %-04d, 키 : %08.1f" %(name, age, height))
#(2) {} 포맷팅
print("이름 : {}, 나이 : {}, 키 : {}".format(name, age, height))
print("이름 : {0}{0}, 나이 : {1}{0}, 키 : {2}".format(name, age, height))
#(2-1) 자릿수 지정
#^ : 가운데 정렬, '<' : 왼쪽 정렬, '>' : 오른쪽 정렬
print("이름 : {0:>10}, 나이 : {1:<5}, 키 : {2:>10.2f}".format(name, age, height))
#(2-2) 자릿수 문자 채우기
print("이름 : {0:->10}, 나이 : {1:0<5}, 키 : {2:0>10.2f}".format(name, age, height))
print("이름 : {0:>a^10}, 나이 : {1:b<5}, 키 : {2:c>10.2f}".format(name, age, height))

#(3) f-포맷팅 방법 (파이썬 3.6버전 이후)
print(f"이름 : {name}, 나이 : {age}, 키 : {height}")
#(3-1) 자릿수 지정
print(f"이름 : {name:-^10}, 나이 : {age:a<5}, 키 : {height:b>8.2f}")
print(f"이름 : {name:^10}, 나이 : {age:<5}, 키 : {height:>8.2f}")






























































































































































































































