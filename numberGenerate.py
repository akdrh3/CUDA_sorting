import random as r

f = open("oneMillionNum.txt", "w")
for i in range(0,1000000):
    num = r.randint(0,100000)
    f.write("%d\n"%num)
