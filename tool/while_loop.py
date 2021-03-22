from sklearn.metrics import accuracy_score
import random

list1 = [0,1,0,2,1,0]
list2 = [0,1,0,3,1,0]
list4 = [0,0,0,0,0,0]

length = len(list1)
list3 = []
acc = accuracy_score(list1,list2,normalize=True)

print("Accuracy: {:.3f}%".format(acc*100))

acc1 = sum(1 for x,y in zip(list1,list2) if x == y) / len(list1)

print(acc1)

i = 0
while i < length:
    a = list1[i]
    b = list2[i]
    c = list4[i]
    # if a == 0:
    #     if b == 0:
    #         if c ==0:
    #             print('hello')
    # print(a)
    # print(b)
    if a == b:
        print('a vs b')
        list3.append('a vs b')

    elif a == 0 and b == 1:
        print('a0 vs b1')
        list3.append('a0 vs b1')
    elif a == 0 and b == 2:
        print('a0 vs b2')
        list3.append('a0 vs b2')
    elif a == 0 and b == 3:
        print('a0')
        list3.append('a0')

    elif b == 0 and a == 1:
        print('b0 vs a1')
        list3.append('b0 vs a1')
    elif b == 0 and a == 2:
        print('b0 vs a2')
        list3.append('b0 vs a2')
    elif b == 0 and a == 3:
        print('b0')
        list3.append('b0')

    elif a == 1 and b == 0:
        print('a1 vs b0')
        list3.append('a1 vs b0')
    elif a == 1 and b == 2:
        print('a1 vs b2')
        list3.append('a1 vs b2')
    elif a == 1 and b == 3:
        print('a1 vs b3')
        list3.append('a1 vs b3')

    elif b == 1 and a == 0:
        print('b1 vs a0')
        list3.append('b1 vs a0')
    elif b == 1 and a == 2:
        print('b1 vs a2')
        list3.append('b1 vs a2')
    elif b == 1 and a == 3:
        print('b1')
        list3.append('b1')

    elif a == 2 and b == 0:
        print('a2 vs b0')
        list3.append('a2 vs b0')
    elif a == 2 and b == 1:
        print('a2 vs b1')
        list3.append('a2 vs b1')
    elif a == 2 and b == 3:
        print('a2')
        list3.append(3)

    elif b == 2 and a == 0:
        print('b2 vs a0')
        list3.append('b2 vs a0')
    elif b == 2 and a == 1:
        print('b2 vs a1')
        list3.append('b2 vs a1')
    elif b == 2 and a == 3:
        print('b2')
        list3.append(3)

    i += 1

print(list3)


num1 = random.randint(0, 8)
print("Random integer: ", num1)
