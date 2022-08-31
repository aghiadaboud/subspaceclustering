from pyclustering.utils import read_sample

from pyclustering.samples.definitions import SIMPLE_SAMPLES
import numpy as num
import itertools
import more_itertools
import math
import matplotlib.pyplot as plt
from itertools import chain
from collections import Counter
from functools import reduce




class temp:

    def __init__(self):
      self.x = 1
    def test():
        a = [[1, 2, 3], [4, 5, 6]]
        print(a)
        print(num.shape(a))
        print(a[0])

    def test2(self):
        sample = read_sample(SIMPLE_SAMPLES.SAMPLE_SIMPLE1)
        print(sample)
        print(num.shape(sample))
        print([row[0] for row in sample])
        print(num.shape([row[0] for row in sample]))
        print(sample[0][1])
        print(num.shape(sample[0]))

    def test3():
        a = [1, 2, 3]
        b = []
        while a:
            print("aaaaa  ")
            break
        while b:
            print("bbbb  ")

    def test4():
        h = {}
        g = {}
        a = [1, 2, 3]
        b = [2, 3, 4]
        c = [5, 6, 7]
        h[1] = a
        h[2] = b
        h[3] = c
        print(h)
        print(h.get(1))
        while g:
            print("zw")
            break
        while h:
            print("xy")
            break

    def test5():
        a = [1, 2, 3]
        for i1 in a:
            for i2 in a:
                print(i1, i2)

    def test6():
        a = [["a", "b", "c"], [1, 2, 3], [4, 5, 6], [7, 8, 9]]
        print(a)
        print(num.shape(a))
        print([row[0:2] for row in a])
        print(num.shape([row[0] for row in a]))
        print(a[0][1])
        print(num.shape(a[0]))

    def test7():
        a = [1, 2, 3]
        for i1 in range(len(a)):
            print(i1)

    def test8():
        a = [[1, 2, 3], [4, 5, 6]]
        for l in a:
            print(l)

    def test9():
        h = {}
        a = [1, 2, 3, 4]
        b = [2, 3, 4]
        c = [5, 6]
        h[1] = a
        h[2] = b
        h[3] = c
        print(h)
        print([h.get(key) for key in [1, 3]])

    def test10():
        h = [[1, 2, 3, 4], [2, 3, 4], [5, 6]]
        clusters_len = [len(cluster) for cluster in h]
        print(clusters_len)
        print(min(clusters_len))
        # somecluster = [c for c in h if len(c) == min(clusters_len)]
        for somecluster in h:
            if len(somecluster) == min(clusters_len):
                c = somecluster
        print(c)

    def test11():
        h = {}
        a = [1, 2, 3, 4]
        b = [2, 3, 4]
        c = [5, 6]
        h[1] = a
        h[2] = b
        h[7] = c
        print(h)
        print(list(h.keys())[list(h.values()).index([5, 6])])

    def test12():
        a = [
            ["a", "b", "c", "d"],
            [1, 2, 3, 4],
            [4, 5, 6, 7],
            [7, 8, 9, 10],
            [1, 11, 12, 13],
        ]
        print(a)
        # b = a for ([row[0] for row in a] ==)
        print(len(a))
        result = []
        featuers = [1, 3]
        for i in range(len(a)):
            for t in range(len(a[0])):
                if a[i][t] == 1:
                    l = [a[i][t]]
                    for f in featuers:
                        l.append(a[i][f])
                    # result.append([a[i][t], a[i][featuer]])
                    result.append(l)
        print(result)

    def test13(self):
        Y = num.arange(25).reshape(5, 5)
        print(Y)
        print(Y[num.ix_([1, 2, 3], [1, 2, 3])])
        print(Y[num.ix_([1, 3], [1, 3])])
        Y[num.ix_([1, 3], [1, 3])] = num.arange(4).reshape(2, 2)
        print(Y, num.shape(Y))
        x = num.zeros((5,2)).reshape(5, 2)
        Y[num.ix_(list(range(5)), [0, 1])] = x
        print(Y, )


    def test14():
        a = [["a", "b", "c"], [1, 2, 3], [4, 5, 6], [7, 8, 9]]
        print(a)
        b = [row[0] for row in a]
        print(b)
        b.pop(0)
        print(b, b.index(1))

    def test16():
        h = {}
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        b = [[10, 11], [12, 13, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        h[1] = a
        h[2] = b
        print(a)
        print(h)
        minl = 0
        bestsubspace = 0
        for key in [1, 2]:
            count_objects = 0
            liste = h.get(key)
            for cluster in range(len(liste)):
                count_objects = count_objects + len(liste[cluster])
            if minl == 0:
                minl = count_objects
                bestsubspace = key
            elif count_objects < minl:
                minl = count_objects
                bestsubspace = key
        print(bestsubspace)

    def test17():
        a = [[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10], [1, 2, 3, 4], [4, 5, 6, 7]]
        print(a)
        b = num.array(a)
        print(b)
        print(b[num.ix_([0, 3, 4], [0, 3])])
        c = b.tolist()
        print(c)

    def test18(self):
        a = [1, 2, 3]
        b = [1, 2, 4]
        set_a = set(a)
        set_b = set(b)
        print(set(a) & set(b))
        print(list(set(a) & set(b)))
        print(len(set(a) & set(b)) == len(a) - 1 and a[-1] < b[-1])

    def test19():
        a = [[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10], [1, 2, 3, 4], [4, 5, 6, 7]]
        b = num.array(a)
        print(b)
        new_b = num.delete(b, 0, 0)
        print(new_b)

    def test20():
        a = [1, 2, 3]
        b = [1, 2, 3]
        m = [1, 2, 4]
        if a == b:
            print(a[:-1])
        if a == m:
            print("zzz")

    def test21(self):
        a = [0, 2, 3]
        b = itertools.combinations(a, r=2)
        print([list(y) for y in b])
        print(list(b))
        for x in itertools.combinations(a, r=2):
          print(list(x))


    def test22():
        h = {}
        a = [1, 2, 3, 4]
        b = [2, 3, 4]
        h[1] = a
        h[2] = b
        g = {}
        t = [2, 3, 4]
        s = [5, 6]
        g[77] = t
        g[34] = s
        f = h | g
        print(h, g)
        print(f)




    def test23(self):
        h = {}
        a = [[1, 2, 3, 4]]
        b = [[2, 3, 4], [7,8,9,7]]
        h[1] = a
        h[2] = b
        print(h, list(h.keys()))
        g = list(h.values())
        print(g)



    def test24():
      b = [[2, 3, 4], [7,8,9,7]]
      b.remove([7,8,9,7])
      print(b)


    def test25():
        h = {}
        a = [[1, 2, 3, 4]]
        b = [[2, 3, 4], [7,8,9,7]]
        h[1] = a
        h[2] = b
        h.pop(2)
        print(h)


    def test26():
      a = [1, 2]
      b = [1, 2]
      d = [3, 4]
      c= [a,d, b]
      print(c)
      c.remove([1,2])
      print(c)


    def test27(self):
      a = [1,2,3,4,5, 5]
      b = [2,3,4]
      print(num.max(a))
      print(a.index(5))
      print(num.where(num.array(a) == 5)[0])
      print(set(a).difference(set(b)))


    def test28(self):
      a = [1,77, 5, 2,3, 78, 4,100, 5]
      b = sorted(a)[-4:]
      c = [a.index(i) for i in b]
      print(b)
      print(c)

    def test29(self):
      a = [1,2,3]
      b = [4, 5, 2]
      print(list(set(a + b)))


    def test30(self):
      d = [[1,2,3,4], [2,3,4], [3,4,5,6,7], [5,77,9,10]]
      print(set.intersection(*map(set,d)))


    def test31(self):
      a = [1,3,2]
      b = 5
      c = b
      self.test32(a)
      b = 9
      print(a, b, c)


    def test32(self, b):
      b.remove(2)


    def test33(self):
      a = [1,3,2, 5 , 7 ,8]
      for i in a:
        if i == 2:
          a.remove(2)
        print(a)
        print(i)

    def test34(self):
      a = [1,3,2, 5 , 7 ,8]
      for i in range(len(a)):
        print(i)
        a.pop(i)
        print(a[i])

    def test35(self):
      lol = [[1,2,3,4], [2,3,4], [3,4,5,6,7], [5,77,9,10]]
      for i, l in enumerate(lol):
        print(i, l)
        a = [0,0]
        lol[i] = a.copy()
      print(lol)

    def test36(self):
      lol = [[1, 1,2,3,4], [2,3,4], [3,4,5,6,7], [5,77,9,10]]
      union = set().union(*lol)
      print(union)
      print(list(union))


    def test37(self):
      l = [3, 77]
      for i in range(5):
        l.append(i)
        print(len(l) - 1)
      print(l)

    def test38(self):
        h = {}
        a = [[1, 2, 3, 4]]
        b = [[2, 3, 4], [7,8,9,7]]
        h[1] = a
        h[2] = b
        print(h)
        h[77] = h.pop(1)
        print(h)


    def test39(self):
      a = [1, 2, 3, 4]
      b = [[7,8,9,7]]
      b.append(a)
      print(b)
      a.pop(1)
      print(b)

    def test40(self):
      b = {}
      a = [1, 2, 2, 4]
      c = [2, 3, 4]
      d = [7,8,9,7]
      b[tuple(a)] = 1
      b[tuple(c)] = 2
      b[tuple(d)] = 3
      print(b)
      for t in b.keys():
        print(t)
      print(b.keys())
      print(list(b.keys()))
      print(a)


    def test41(self):
      lol = [[1,2,3,4], [2,3,4, 4], [3,4,5,6,7], [5,77,9,10]]
      a = [1,3]
      b = [lol[i] for i in a]
      b[0].pop(0)
      print(b)
      print(lol)

    def test42(self):
      a = [0, 2, 3]
      b = itertools.combinations(a, r=2)
      print([list(y) for y in b])
      print(b)

    def test43(self):
      b = {}
      a = [1, 2, 2, 4]
      b[1] = a
      for i in range(len(b.get(2, []))):
        print('sdfsdf')
      print(b.get(2, []))


    def test44(self):
      a = [0, 2, 3]
      b = list(itertools.combinations(a, r=2))
      c = [list(x) for x in b]
      print(c)
      for y in b:
        print(y)
      for y in b:
        print('yo')
      print(b[2])

    def test45(self):
      a = [0, 2]
      b = [0, 5]
      c = a.copy() + [b[-1]]
      print(a, b, c)

    def test46(self):
      h = {}
      a = [1, 2, 3, 4]
      b = [2, 3, 4]
      h[1] = a
      h[2] = b
      print(list(h))
      print(h)

    def test47(self):
      s = [5,77,9,10, 444]
      t = sorted(range(len(s)), key=lambda k: s[k])
      print(t)
      print([s[i] for i in t[-3:]])

    def test48(self):
      s = [5,77,9,10, 444, 5, 77]
      print(list(more_itertools.locate(s, lambda a: a == num.min(s))))

    def test49(self):
      a = [0, 2, 3]
      b = list(itertools.combinations(a, r=2))
      c = [b[i] for i in [0, 2]]
      print(c)

    def test50(self):
      pass

c = temp()
c.test50()





