from simpful import *
import numpy as np
import matplotlib.pyplot as plt

FS = FuzzySystem()

def draw_triangles(values):
    if len(values) != 3:
        raise ValueError("Invalid values passed to draw_triangles function")
    x = np.linspace(0, 25, 100)  # X-axis range
    for p in values:
        if len(p) != 3:
            raise ValueError("Invalid values passed to draw_triangles function")
        a = p[0]  # Left boundary
        b = p[1]  # Peak
        c = p[2]  # Right boundary
        y = np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))
        plt.plot(x, y)

    plt.title('Triangular Membership Function')
    plt.xlabel('X')
    plt.ylabel('Membership Value')
    plt.grid(True)
    plt.show()

t1 = [0, 5, 10]
t2 = [7, 11, 15]
t3 = [13, 19, 25]
triangle_1 = TriangleFuzzySet(t1[0], t1[1], t1[2], term="short")
triangle_2 = TriangleFuzzySet(t2[0], t2[1], t2[2], term="medium")
triangle_3 = TriangleFuzzySet(t3[0], t3[1], t3[2], term="long")
draw_triangles([t1, t2, t3])
FS.add_linguistic_variable("brewing_time",
        LinguisticVariable([triangle_1, triangle_2, triangle_3], universe_of_discourse=[0,25]))

t1 = [0, 6.25, 12.5]
t2 = [7.5, 12.5, 17.5]
t3 = [15, 20, 25]
triangle_4 = TriangleFuzzySet(t1[0], t1[1], t1[2], term="low")
triangle_5 = TriangleFuzzySet(t2[0], t2[1], t2[2], term="medium")
triangle_6 = TriangleFuzzySet(t3[0], t3[1], t3[2], term="high")
draw_triangles([t1, t2, t3])
FS.add_linguistic_variable("coffee_strength",
        LinguisticVariable([triangle_4, triangle_5, triangle_6], universe_of_discourse=[0,25]))

t1 = [0, 0, 13]
t2 = [0, 13, 25]
t3 = [13, 25, 25]
triangle_7 = TriangleFuzzySet(t1[0], t1[1], t1[2], term="low")
triangle_8 = TriangleFuzzySet(t2[0], t2[1], t2[2], term="medium")
triangle_9 = TriangleFuzzySet(t3[0], t3[1], t3[2], term="low")
FS.add_linguistic_variable("enjoyment_level",
        LinguisticVariable([triangle_7, triangle_8, triangle_9], universe_of_discourse=[0,25]), verbose=False)
draw_triangles([t1, t2, t3])

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
        enjoyment_level = FS.inference(verbose=False)['enjoyment_level']
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

