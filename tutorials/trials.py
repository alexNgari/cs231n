import numpy as np
import statistics

a = np.array([[1,2,3],[4,5,6],[7,8,9]])

# print(a)
# print(a[0,:])

b = a[:,0].T

# print(b)

# a = np.array([1,2,3,1,2,1,1,1,3,2,2,1])
# counts = np.bincount(a)
# print(counts)
# print(np.argmax(counts))

# print(np.argsort(a[:,0])[:2])

listA = [1,2,3,1,2,1,1,1,3,2,2,1]
# print(statistics.mode(listA))
# print(max(set(listA), key=listA.count))
# print(listA.count(1))

# print(int(1.0))

aAll = np.split(a, 3, axis=0)
# print(a)
# print(aAll.pop(0))

b = np.stack((aAll[:]), axis=0)
# print(b)
# print(int(b.shape[0]/2))
# c = b.reshape(3,3)
# print(c)

# d = ['Alex', 'Ngari', 'Maina', 'Yvonne', 'Mickel', 'Belashi']
# e = [x for x in d if x!=d[3]]
# print(e)

# randInt = 5
# k_to_accuracies = {}
# k_to_accuracies[randInt] = []
# k_to_accuracies[randInt].append(1)
# print(k_to_accuracies)

# a = list(range(10))
# b = a
# b.append(0)
# print(a)
# print(b)

# print(a)
# a[:,0] = a[:,0] + a[:,0]
# print(a[0])

c = np.array([[2,3,4], [6,7,8], [10,11,12]])

# print(a)
# print(c)
# a[:,0] += c[0,:].T
# print(a)

# print(np.maximum(1,2))

# print(np.where(c>5, 1, 0))

d = c[0,:]

# d[c>5] = 1

# print(d)
# print(np.shape(np.sum(d, axis=1).T))

# print(d.shape)


a = np.matrix([[1,2,3], [4,5,6],[7,8,9]])
# b = np.zeros((4,3))
# b[np.arange(3), :] = a
# b[-1, :] = [1,1,1]
# print(b)
# c = np.zeros(np.shape(a))
# print(c)

b = a[:,1]
print(b)

b += 1

print(a>3)

