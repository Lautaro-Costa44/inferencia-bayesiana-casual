# ### Apuesta individual
#
# Una casa de apuestas paga 3 por Cara y 1.2 por Sello. La moneda tiene 0.5 de probabilidad de que salga Cara o Sello.
# Nos ofrecen jugar 10000 veces, apostando en cada ocasión todos nuestros recursos, 50% a Cara y 50% a Sello.
# ¿Nos conviene jugar?
# Notar que cuando sale Cara, crecen nuestros recursos 50% (Si teníamos 100, pusimos 50 en Cara y nos pagaron 3*50=150).
# Notar que cuando sale Sello, crecen nuestros recursos -40% (Si teníamos 100, pusimos 50 en Cara y nos pagaron 1.2*50=60).
#
# 0. No
# 1. Sí

import random
import statistics
import math
# Utilizo log para manejar mejor estos numeros muy bajos
def jugar_log(veces):
    log_plata = math.log(1000) 
    for _ in range(veces):
        resultado = random.choices(["Cara", "Sello"], weights=[0.5, 0.5], k=1)[0]
        if resultado == "Cara":
            log_plata += math.log(1.5)   # +50%
        else:
            log_plata += math.log(0.6)   # -40%
    return log_plata

n_jugadas = 10000
n_simulaciones = 2000

log_resultados = [jugar_log(n_jugadas) for _ in range(n_simulaciones)]
mediana_log = statistics.median(log_resultados)

print(f"mediana de log(plata): {mediana_log:.2f}")
print(f"equivalente a plata = e^{mediana_log:.2f} = {math.exp(mediana_log):.3e}")

#Conlcucion: no conviene