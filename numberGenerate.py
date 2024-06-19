import random as r

f = open("oneMillionNum.txt", "w")
for i in range(0,1000000):
    num = r.randint(0,100000000)
    f.write("%d\n"%num)

f = open("twoMillionNum.txt", "w")
for i in range(0,2000000):
    num = r.randint(0,100000000)
    f.write("%d\n"%num)


f = open("fourMillionNum.txt", "w")
for i in range(0,4000000):
    num = r.randint(0,100000000)
    f.write("%d\n"%num)


f = open("eightMillionNum.txt", "w")
for i in range(0,8000000):
    num = r.randint(0,100000000)
    f.write("%d\n"%num)

f = open("sxtnMillionNum.txt", "w")
for i in range(0,16000000):
    num = r.randint(0,100000000)
    f.write("%d\n"%num)

f = open("thrtytwMillionNum.txt", "w")
for i in range(0,32000000):
    num = r.randint(0,100000000)
    f.write("%d\n"%num)
