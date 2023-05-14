# installing dependencies:
# pip3 install -r requirements.txt

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# zmienna wejsciowa: x
x = np.linspace(0, 10, 100)

# funkcje przynaleznosci fuzzy dla danej wejsciowej x
inside_low = fuzz.trimf(x, [0, 0, 5])
inside_medium = fuzz.trimf(x, [0, 5, 10])
inside_high = fuzz.trimf(x, [5, 10, 10])

# zmienna wyjsciowa: y
y = np.linspace(0, 20, 100)

# funkcje przynaleznosci fuzzy dla danej wyjsciowej y
outside_low = fuzz.trimf(y, [0, 0, 10])
outside_medium = fuzz.trimf(y, [0, 10, 20])
outside_high = fuzz.trimf(y, [10, 20, 20])

# zasada 1: jesli x jest niskie THEN y is niskie
rule1 = np.fmin(inside_low, outside_low)

# zasada 2: jesli x jest srednie THEN y is srednie
rule2 = np.fmin(inside_medium, outside_medium)

# zasada 3: jesli x jest wysokie THEN y is wysokie
rule3 = np.fmin(inside_high, outside_high)

# agregacja wszystkich wyjsciowych funkcji przynaleznosci
aggregated = np.fmax(rule1, np.fmax(rule2, rule3))

# defuzyfikacja zagregowanych funkcji przynaleznosci metoda centroidu
output = fuzz.defuzz(y, aggregated, 'centroid')

# wizualizacja funkcji przynaleznosci oraz zagregowanych wynikow
plt.figure(figsize=(8, 6))

plt.plot(x, inside_low, 'b', linewidth=1.5, label='inside_low')
plt.plot(x, inside_medium, 'g', linewidth=1.5, label='inside_medium')
plt.plot(x, inside_high, 'r', linewidth=1.5, label='inside_high')

plt.plot(y, outside_low, 'c', linewidth=1.5, label='outside_low')
plt.plot(y, outside_medium, 'm', linewidth=1.5, label='outside_medium')
plt.plot(y, outside_high, 'y', linewidth=1.5, label='outside_high')

plt.plot(y, aggregated, 'k', linewidth=2, label='aggregated')

plt.fill_between(y, 0, aggregated, facecolor='Orange', alpha=0.7)
plt.vlines(output, 0, 1, colors='r', linestyle='dashed')

plt.title('kontrola temperatury')
plt.xlabel('Input (x) / Output (y)')
plt.ylabel('stopien przynaleznosci')
plt.legend()
plt.grid(True)
plt.show()

