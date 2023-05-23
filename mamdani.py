from simpful import *
import numpy as np
import matplotlib.pyplot as plt

FS = FuzzySystem()

triangle_1 = TriangleFuzzySet(0,5,10,   term="short")
triangle_2 = TriangleFuzzySet(7,12,15,  term="medium")
triangle_3 = TriangleFuzzySet(13,20,25, term="long")
FS.add_linguistic_variable("brewing_time",
        LinguisticVariable([triangle_1, triangle_2, triangle_3], universe_of_discourse=[0,25]))

triangle_4 = TriangleFuzzySet(0,5,12.5,  term="low")
triangle_5 = TriangleFuzzySet(7.5,10,17.5,  term="medium")
triangle_6 = TriangleFuzzySet(15,20,25, term="high")
FS.add_linguistic_variable("coffee_strength",
        LinguisticVariable([triangle_4, triangle_5, triangle_6], universe_of_discourse=[0,25]))

triangle_7 = TriangleFuzzySet(0,0,13,   term="low")
triangle_8 = TriangleFuzzySet(0,13,25,  term="medium")
triangle_9 = TriangleFuzzySet(13,25,25, term="low")
FS.add_linguistic_variable("enjoyment_level",
        LinguisticVariable([triangle_7, triangle_8, triangle_9], universe_of_discourse=[0,25]))

FS.add_rules([
        "IF (coffee_strength IS low) AND (brewing_time IS short) THEN (enjoyment_level IS low)",
        "IF (coffee_strength IS low) AND (brewing_time IS medium) THEN (enjoyment_level IS medium)",
        "IF (coffee_strength IS low) AND (brewing_time IS long) THEN (enjoyment_level IS medium)",
        "IF (coffee_strength IS medium) AND (brewing_time IS short) THEN (enjoyment_level IS medium)",
        "IF (coffee_strength IS medium) AND (brewing_time IS medium) THEN (enjoyment_level IS high)",
        "IF (coffee_strength IS medium) AND (brewing_time IS long) THEN (enjoyment_level IS medium)",
        "IF (coffee_strength IS high) AND (brewing_time IS short) THEN (enjoyment_level IS medium)",
        "IF (coffee_strength IS high) AND (brewing_time IS medium) THEN (enjoyment_level IS medium)",
        "IF (coffee_strength IS high) AND (brewing_time IS long) THEN (enjoyment_level IS low)"
    ])

x = []
y = []
z = []

for a in range(25):
    for b in range(25):
        x.append(a)
        y.append(b)
        FS.set_variable("coffee_strength", a)
        FS.set_variable("brewing_time", b)
        enjoyment_level = FS.inference()['enjoyment_level']
        z.append(enjoyment_level)

x = np.reshape(x, (25, 25))
y = np.reshape(y, (25, 25))
z = np.reshape(z, (25, 25))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, z)

ax.set_xlabel('coffee_strength')
ax.set_ylabel('brewing_time')
ax.set_zlabel('enjoyment_level')
plt.show()

