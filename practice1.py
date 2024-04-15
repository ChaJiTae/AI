# 파이썬 복습(함수)
def sayHello():
  print("Hello!")

sayHello()

def sayHello(name):
  print("Hello! "+name)

sayHello("Cha")

# 클래스
class Person:
  def __init__(self,name,age):
    self.name = name
    self.age = age

  def sayHello(self):
    print("나의 이름은 "+ self.name)
  
p1 = Person("Jitae",24)
p1.sayHello()

lst = [10,20,30,40,50]
lst
lst[2]
lst[2]=90
lst

len(lst)