import time
from python.xavier import Array
from python.metal.graph import MTLGraph
from python.metal.kernels import sparse_copy
from python.config import config
import numpy as np

x1 = Array.constant([1, 3, 2], 7.0)
x2 = Array.arange([1, 3, 2], 2, 2)
x3 = x1 + x2
x4 = x1 + x2
x5 = x3 * x4
x6 = x5 * x5
g = MTLGraph(x1)
g.compile()
start = time.time()
g.forward()
end = time.time()
print(end - start)
# n1 = np.frombuffer(x1, dtype=np.float32)
# n2 = np.frombuffer(x2, dtype=np.float32)
# start = time.time()
# n3 = n1 * n2
# n4 = n1 + n2
# n5 = n3 * n4
# n6 = n5 * n5
# end = time.time()
# print(end - start)
