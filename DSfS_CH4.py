# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 23:33:54 2021

@author: Alejandro Herrera
"""
#### CHAPTER 4: Linear Algebra]

##VECTORS
#Vecotrs are objects that can be added together to form new vetcors:
    #and can be nultiplied by scalars(numbers) to form new vectors
# Simplest from-scratch  approach is to represent vectors as lists of numbers:
    # List of three numbers corresponds to a vector in three-dimensional space,
    # vice versa 
# Vector is just a list of floats:
from typing import List
Vector = List[float]

height_wieght_age = [70,       #inches,
                      120,      #pounds,
                      40 ]      #years

grades = [95,   # exam1
          80,   # exam2
          75,   # exam3
          62 ]  # exam4

# Can easily implement adding vectors by zip-ing the vectors together and using 
# a list comprehension to add corresponding elements:

def add(v: Vector, w:Vector) -> Vector:
    """ADDS CORRESPONDING ELEMENTS"""
    assert len(v) == len(w), "Vectors must be the same length"
    return [v_i + w_i for v_i, w_i in zip(v,w)]
assert add([1, 2, 3], [4, 5, 6]) == [5, 6, 7]

def subtract(v: Vector, w: Vector) -> Vector:
    """SUBTRACTS CORRESPONDING ELEMENTS"""
    assert len(v) == len(w), "Vectors must be the same length"
    return [v_i - w_i for v_i, w_i in zip(v,w)]
assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]

# Sometimes want to componentwise sum a list of vectors -> Create a vetcor
# whoes first element is the sum of all the first elements, 
# second element is the sum of all the second elements
def vector_sum(vectors: List[Vector]) -> Vector:
    """SUM ALL CORRESPONDING ELEMENSTS"""
    assert vectors, "no vectors provided!"
    # Check the vectors are all the same size
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different size"
    
    # the i-th element of the result is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors)
        for i in range (num_elements)]
assert vector_sum([[1, 2], [3, 4], [5, 6], [7,8]]) == [16, 20]

# Need to be able to multiply a vector by a scalar
# Simply by multiplying each element of the vector
def scalar_multiply(c: float, v: Vector) -> Vector:
    """MULTIPLIES EVERY ELEMENT BY C"""
    return [c* v_i for v_i in v]
assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]

# Allow us to compute the componentwise means of a list of(same-sized) vectors:
def vector_mean(vectors: List[Vector]) -> Vector:
    """COMPUTES THE ELEMENT WISE AVERAGE"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))
assert vector_mean([[1, 2],[3, 4], [5, 6]]) == [3, 4]

# Less obvious tool dot product. Dot product of two vectors is the sum of their componentwise products
def dot(v: Vector, w: Vector) -> float:
    """COMPUTES v_i * w_i + ... v_n + w_n"""
    assert v(len) == len(w), "vectors must be same length"
    return sum(v_i * w_i for v_i, w_i in zip(v, w))
assert dot([1, 2, 3], [4, 5, 6]) == 32      # 1 * 4 + 2 * 5 + 3 * 6

# Easy to compute a vectors sum of squares:
def sum_of_squares(v: Vector) -> float:
    """RETURNS v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)
assert sum_of_squares([1, 2, 3]) == 14  #1 * 1 + 2 * 2 + 3 * 3
# We can compute its magnitude (length)

import math 

def magnitude(v: Vector) -> float:
    """Returns the magnitude (or length) of v"""
    return math.sqrt(sum_of_squares(v))          #math.sqrt is square root function

assert magnitude([3, 4]) == 5
# All the pices we need to compute the distances between 2 vectors
# IN CODE:
def squared_distance(v:Vector, w: Vector) -> float:
    """COMPUTES (v_1 - w_1)**2 + ... + (v_n -w_n) ** 2"""
    return sum_of_squares(subtract(v,w))

def distance(v: Vector, w: Vector) -> float:
    """COMPUTES THE DISTANCE BETWEEN v and w"""
    return math.sqrt(subtract(v, w))

#Using lists as vectors is great for exposition but terrible for performance:
#For production code you wuld want to us teh NumPy library, which includes a high performance
#array class with all sorts of arithmetic operations

##MATRICES

#Matrix - two dimensional collection of numbers
#Will represent matricies as list of list, with each inner list having the same size and representing
# a row of the matrix
# If A is a matrix, then A[i][j] is the element in the ith row anth the jth row column.
#Frequently us capital letters to represent matricies

#Another type of alias
Matrix = List[List[float]]

A = [[1, 2, 3],         #A has 2 rowsand 3 columns
     [4, 5, 6]]

B = [[1, 2],         #B has 3 rows and 2 columns 
    [3,4],
    [5,6]]

#In mathematics, you would usually name the first row of the matrix "row1" and the first column "column1"
#We call the first row of a matrix "row0" and the first column "column 0"

#Matrix A has len(A) rows and len([0]) columns -> We consider its shape:
from typying import tuple
def shape(A: Matrix) -> tuple[int, int]:
    """RETURNS (# of rows of A, # of columns of A)"""
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0        # number of elements in firts row
    return num_rows, num_cols

assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)  #2 rows, #3 columns

#If matrix has n rows and k columns - will be refered to as an nxk matrix 
def get_row(A: Matrix, i: int) -> Vector:
    """RETURNS THE iTH ROW OF A  (AS A VECTOR)"""
    return A[i]     #A[i] is already the ith row
def get_column(A: Matrix, j: int) -> Vector:
    """RETURNS THE jTH COLUMN OF A (AS A VECTOR)"""
    return [A_i[j]                  #jth element of row A_i
            for A_i in A]           #for each row A_i

#Also want to be able to create matrix given its shape and a function for generating 
#its elements. Can do this by using a nested list comprehension:
from typing import Callable
def make_matrix (num_rows: int, 
                 num_cols: int,
                 entry_fn: Callable[[int, int], float]) -> Matrix:
        """
        RETURNS A num_rows x num_cols MATRIX
        WHOSE (i,j) th ENTRY IS entry_fn(i, j)
        """
        return [[entry_fn(i, j)             # given, create a list
                 for j in range(num_cols)]  # [entry_fn (i, 0), ...]
                for i in range(num_rows)]   # create one list for each i
#This function - you could make a 5x5 IDENTITY MATRIX(with 1s on the diagnoal and 0s elsewhere)
def identity_matrix(n: int) -> Matrix:
    """RETURNS THE n x n IDENTITY MATRIX"""
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)

assert identity_matrix(5) == [[1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0], 
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1]]
#FIRST: Can use matrix to represent a dataset concicting of multiple vectors, simply
#by considering each vector as a row of the matrix
#-- If you had the heights, weights, and ages of 1,000 people, you could
#-- put them in a 1,000 * 3 matrix:
data = [[70, 170, 40],
        [65, 120, 26],
        [77, 250, 19]]
#SECOND: We can use an n x k matrix to represent a linear function that maps k-dimensional 
# vectors to n-dimensional vectors.
#THRID: Matricies can be used to represent binary relationships. Relationship would be 
# to create a matrix A such that A[i][j] is 1 if nodes i and j are connected and 0 otherwise
#   user 0 1 2 3 4 5 6 7 8 9 
#
friend_matrix = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],    #user 0
                 [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],    #user 1
                 [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],    #user 2
                 [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],    #user 3
                 [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],    #user 4
                 [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],    #user 5
                 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],    #user 6
                 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],    #user 7
                 [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],    #user 8
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]    #user 9
#Matrix representation -it much quicker to check whether two nodes are connected
assert friend_matrix[0][2] == 1, "0 and 2 are friends"
assert friend_matrix[0][8] == 0, "0 and 8 are friends"
#To find a node's connections, you only need to inspect the column corresponding to that node
#only need to look at one row
friends_of_five = [i
                   for i, is_friend in enumerate(friend_matrix[5])
                   if is_friend]
#With a small group you could just add a list of connections to each node object to
#speed up this process; but for large, evolving graph that would too much to maintain
