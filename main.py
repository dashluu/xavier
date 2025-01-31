import time
from python.xavier import Array
from python.metal.graph import MTLGraph
from python.config import config

x1 = Array.constant([32, 64, 30, 30], 2.0)
x2 = Array.arange([32, 64, 30, 30], 2, 2)
x3 = x1 * x2
x4 = x1 + x2
x5 = x3 * x4
x6 = x5 * x5
g = MTLGraph(x6)
start = time.time()
g.forward()
end = time.time()
print(end - start)
g.compile()
start = time.time()
g.forward()
end = time.time()
print(end - start)
# print(x1)
# print()
# print(x2)
# print()
# print(x3)
# print()
# print(x4)
# print()
# print(x5)
