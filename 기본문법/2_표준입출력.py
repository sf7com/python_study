#) 표준 입력
print("Enter Your Name:")
name = input()
print("My Name is", name)

#1-1) 입력->실수
#input()은 문자열로 입력받는다.
temp = float(input("온도를 입력하세요."))
print("온도:", temp*10)

#2) 표준출력
print("hello") #자동 개행
print("world")

print("hello",end='')#개행안함
print("world")

print("hello",end='------') #개행안함
print("world")

print('hello','world','python')
print('hello'+'world'+'python')