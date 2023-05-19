from simpful import *
import numpy as np
import matplotlib.pyplot as plt

FS = FuzzySystem()

TLV = AutoTriangle(3, terms=['poor', 'average', 'good'], universe_of_discourse=[0,10])
FS.add_linguistic_variable("service", TLV)
FS.add_linguistic_variable("quality", TLV)

O1 = TriangleFuzzySet(0,0,13,   term="low")
O2 = TriangleFuzzySet(0,13,25,  term="medium")
O3 = TriangleFuzzySet(13,25,25, term="high")
FS.add_linguistic_variable("tip", LinguisticVariable([O1, O2, O3], universe_of_discourse=[0,25]))

FS.add_rules([
    "IF (quality IS poor) OR (service IS poor) THEN (tip IS low)",
    "IF (service IS average) THEN (tip IS medium)",
    "IF (quality IS good) OR (service IS good) THEN (tip IS high)"
    ])

x = []
y = []
z = []

for a in range(10):
    for b in range(10):
        x.append(a)
        y.append(b)
        FS.set_variable("quality", a)
        FS.set_variable("service", b)
        tip = FS.inference()['tip']
        z.append(tip)

X = x
Y = y
Z = z

x = np.reshape(X, (10, 10))
y = np.reshape(Y, (10, 10))
z = np.reshape(Z, (10, 10))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
quit()

