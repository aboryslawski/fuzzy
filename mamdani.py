from simpful import *
import numpy as np
import matplotlib.pyplot as plt

FS = FuzzySystem()

TLV = AutoTriangle(3, terms=['niska', 'srednia', 'wysoka'], universe_of_discourse=[0,10])
FS.add_linguistic_variable("jakosc_serwisu", TLV)
FS.add_linguistic_variable("jakosc_jedzenia", TLV)

O1 = TriangleFuzzySet(0,0,13,   term="niski")
O2 = TriangleFuzzySet(0,13,25,  term="sredni")
O3 = TriangleFuzzySet(13,25,25, term="wysoki")
FS.add_linguistic_variable("napiwek", LinguisticVariable([O1, O2, O3], universe_of_discourse=[0,25]))

FS.add_rules([
    "IF (jakosc_jedzenia IS niska) OR (jakosc_serwisu IS niska) THEN (napiwek IS niski)",
    "IF (jakosc_serwisu IS srednia) THEN (napiwek IS sredni)",
    "IF (jakosc_jedzenia IS wysoka) OR (jakosc_serwisu IS wysoka) THEN (napiwek IS wysoki)"
    ])

x = []
y = []
z = []
print("jakosc_jedzenia;jakosc_serwisu;napiwek")
for a in range(10):
    for b in range(10):
        x.append(a)
        y.append(b)
        FS.set_variable("jakosc_jedzenia", a)
        FS.set_variable("jakosc_serwisu", b)
        napiwek = FS.inference()['napiwek'] - 4.324999667
        if napiwek < 0:
            napiwek = 0
        z.append(napiwek)

x = np.reshape(x, (10, 10))
y = np.reshape(y, (10, 10))
z = np.reshape(z, (10, 10))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, z)

ax.set_xlabel('jakosc_jedzenia')
ax.set_ylabel('jakosc_serwisu')
ax.set_zlabel('napiwek')
plt.show()

