import random as r

input_a = int(input("Enter the size of numbers: "))
print("generating ... ")


f = open("numbers.txt", "w")
for i in range(0,input_a * 10):
    num = r.randint(0,100)
    f.write("%d\n"%num)

# f = open("oteMillionNum.txt", "w")
# for i in range(0,128000000):
#     num = r.randint(0,100000000)
#     f.write("%d\n"%num)


# f = open("tfsMillionNum.txt", "w")
# for i in range(0,256000000):
#     num = r.randint(0,100000000)
#     f.write("%d\n"%num)


# f = open("fotMillionNum.txt", "w")
# for i in range(0,512000000):
#     num = r.randint(0,100000000)
#     f.write("%d\n"%num)

# f = open("oztfMillionNum.txt", "w")
# for i in range(0,1024000000):
#     num = r.randint(0,100000000)
#     f.write("%d\n"%num)

# f = open("tzfeMillionNum.txt", "w")
# for i in range(0,2048000000):
#     num = r.randint(0,100000000)
#     f.write("%d\n"%num)
