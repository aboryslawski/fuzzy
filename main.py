# installing dependencies:
# pip3 install -r requirements.txt

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Input variable: x
x = np.linspace(0, 10, 100)

# Fuzzy membership functions for input variable x
x_low = fuzz.trimf(x, [0, 0, 5])
x_medium = fuzz.trimf(x, [0, 5, 10])
x_high = fuzz.trimf(x, [5, 10, 10])

# Output variable: y
y = np.linspace(0, 20, 100)

# Fuzzy membership functions for output variable y
y_low = fuzz.trimf(y, [0, 0, 10])
y_medium = fuzz.trimf(y, [0, 10, 20])
y_high = fuzz.trimf(y, [10, 20, 20])

# Rule 1: IF x is low THEN y is low
rule1 = np.fmin(x_low, y_low)

# Rule 2: IF x is medium THEN y is medium
rule2 = np.fmin(x_medium, y_medium)

# Rule 3: IF x is high THEN y is high
rule3 = np.fmin(x_high, y_high)

# Aggregate all output membership functions
aggregated = np.fmax(rule1, np.fmax(rule2, rule3))

# Defuzzify the aggregated membership function
output = fuzz.defuzz(y, aggregated, 'centroid')

# Visualize the membership functions and the aggregated result
plt.figure(figsize=(8, 6))

plt.plot(x, x_low, 'b', linewidth=1.5, label='x_low')
plt.plot(x, x_medium, 'g', linewidth=1.5, label='x_medium')
plt.plot(x, x_high, 'r', linewidth=1.5, label='x_high')

plt.plot(y, y_low, 'c', linewidth=1.5, label='y_low')
plt.plot(y, y_medium, 'm', linewidth=1.5, label='y_medium')
plt.plot(y, y_high, 'y', linewidth=1.5, label='y_high')

plt.plot(y, aggregated, 'k', linewidth=2, label='aggregated')

plt.fill_between(y, 0, aggregated, facecolor='Orange', alpha=0.7)
plt.vlines(output, 0, 1, colors='r', linestyle='dashed')

plt.title('Takagi-Sugeno Fuzzy Logic System')
plt.xlabel('Input (x) / Output (y)')
plt.ylabel('Membership Degree')
plt.legend()
plt.grid(True)
plt.show()

