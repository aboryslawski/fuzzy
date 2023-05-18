# installing dependencies:
# pip3 install -r requirements.txt

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# zmienna wejsciowa: x
x = np.linspace(0, 15, 150)

# funkcje przynaleznosci fuzzy dla danej wejsciowej x
quality_low = fuzz.trimf(x, [0, 0, 5])
quality_medium = fuzz.trimf(x, [0, 5, 15])
quality_high = fuzz.trimf(x, [5, 15, 15])

# zmienna wyjsciowa: y
y = np.linspace(0, 20, 150)

# funkcje przynaleznosci fuzzy dla danej wyjsciowej y
speed_low = fuzz.trimf(y, [0, 0, 15])
speed_medium = fuzz.trimf(y, [0, 15, 20])
speed_high = fuzz.trimf(y, [15, 20, 20])

# zasada 1: jesli x jest niskie to y is niskie
rule1 = np.fmin(quality_low, speed_low)

# zasada 2: jesli x jest srednie to y is srednie
rule2 = np.fmin(quality_medium, speed_medium)

# zasada 3: jesli x jest wysokie to y is wysokie
rule3 = np.fmin(quality_high, speed_high)

# agregacja wszystkich wyjsciowych funkcji przynaleznosci
aggregated = np.fmax(rule1, np.fmax(rule2, rule3))

# defuzyfikacja zagregowanych funkcji przynaleznosci metoda centroidu
output = fuzz.defuzz(y, aggregated, 'centroid')

# wizualizacja funkcji przynaleznosci oraz zagregowanych wynikow
plt.figure(figsize=(8, 6))

plt.plot(x, quality_low, 'b', linewidth=1.5, label='quality_low')
plt.plot(x, quality_medium, 'g', linewidth=1.5, label='quality_medium')
plt.plot(x, quality_high, 'r', linewidth=1.5, label='quality_high')

plt.plot(y, speed_low, 'c', linewidth=1.5, label='speed_low')
plt.plot(y, speed_medium, 'm', linewidth=1.5, label='speed_medium')
plt.plot(y, speed_high, 'y', linewidth=1.5, label='speed_high')

plt.plot(y, aggregated, 'k', linewidth=2, label='aggregated')

plt.fill_between(y, 0, aggregated, facecolor='Orange', alpha=0.7)
plt.vlines(output, 0, 1, colors='r', linestyle='dashed')

plt.title('napiwek dla obslugi')
plt.xlabel('procent rachunku')
plt.ylabel('stopien przynaleznosci')
plt.legend()
plt.grid(True)
plt.show()

