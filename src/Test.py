import math
import random
import numpy as np

arr1 = np.array([[1,2,3],[4,5,6]])

arr2 = np.array([[1,2,3],[4,7,6]])

equal = arr1.flatten() == arr2.flatten()

print(equal)