import numpy as np

a = np.array([[1,2,3],[4,5,6]])
# print(f'type: {type(a)}')
# print(f'shape: {a.shape}')
# print(f'a[0,0] = {a[0,0]}')

a = np.zeros((0,0))           # Array of zeros

b = np.ones((1,2))            # Array of ones

c = np.full((2,2), 7)       # Array of constants (7's)

d = np.eye(2)               # Identity matrix

e = np.random.rand(2,2) # Array of normalised random numbers

# print(e)

# ARRAY INDEXING
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
b = a[:2, 1:3]  # slice out row 0&1, column 1&2
# print(b)

b[0,0] = 77     # changes both b and a, since b is a view into a
# print(a[0,1])

row2_1 = a[1, :]    # Rank 1 view of row 2: [5,6,7,8]
row2_2 = a[1:2, :]  # Rank 2 view of row 2: [[5,6,7,8]]
print(f'row2_1: {row2_1} -- shape: {row2_1.shape}')
print(f'row2_2: {row2_2} -- shape: {row2_2.shape}')

# Integer array indexing
# print(a[[0, 1, 2], [0, 1, 0]])      # same as printing a[0,1], a[1,1], a[2,0]
# print(np.array([a[0, 0], a[1, 1], a[2, 0]]))

# Selecting one element from each row:
# print(np.arange(4))    # prints 0,1,2,3
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
# b = np.array([0,2,0,1]) # Indices into each row
# print(a[np.arange(4), b])
# a[np.arange(4), b] += 10    #Mutate those elements
# print(a)

bool_idx = (a>2)
# print(bool_idx)
# print(a[bool_idx])


## OPERATIONS
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# print(v.dot(w))
# print(np.dot(v,w))

# print(x.dot(v))     # inner product
# print(v.dot(x))

# print(np.sum(x))    # sum all elements
# print(np.sum(x, axis=0))    # sum each column
# print(np.sum(x, axis=1))    # sum each row

# print(x.T)      # Transpose

# Broadcasting
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))   # Stack 4 copies of v on top of each other
print(vv)
y = x + vv
print(y)
y = x+v     # Works because of broadcasting
print(y)

# Broadcasting two arrays together follows these rules:

# If the arrays do not have the same rank, prepend the shape of the lower rank array with 1s until both shapes have the same length.
# The two arrays are said to be compatible in a dimension if they have the same size in the dimension, or if one of the arrays has size 1 in that dimension.
# The arrays can be broadcast together if they are compatible in all dimensions.
# After broadcasting, each array behaves as if it had shape equal to the elementwise maximum of shapes of the two input arrays.
# In any dimension where one array had size 1 and the other array had size greater than 1, the first array behaves as if it were copied along that dimension