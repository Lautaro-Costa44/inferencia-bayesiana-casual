# ### Diversificación
#
# Una casa de apuestas nos paga 3 por Cara y 1.2 por Sello por el lanzamiento de moneda.
# La moneda es normal, con 0.5 de probabilidad de que salga Cara o Sello.
# Supongamos que nos ofrecen jugar 10000 veces, pero apostando absolutamente todos los recursos en cada paso temporal.
# Apostamos todo, nos devuelven actualizado y volvemos a apostar.
# ¿Qué proporción conviene apostar a Cara?
# Notar que el resto se asigna a Sello.
# Notar además que si apostamos todo a Cara y sale Sello perdemos todos los recursos y no podemos volver a jugar (solo nos pagan en el lado donde sale la moneda).
#
# 0. Recursos asignados a Cara: 0.0
# 1. Recursos asignados a Cara: 0.1
# 2. Recursos asignados a Cara: 0.2    
# 3. Recursos asignados a Cara: 0.3
# 4. Recursos asignados a Cara: 0.4
# 5. Recursos asignados a Cara: 0.5
# 6. Recursos asignados a Cara: 0.6
# 7. Recursos asignados a Cara: 0.7
# 8. Recursos asignados a Cara: 0.8
# 9. Recursos asignados a Cara: 0.9
# 10. Recursos asignados a Cara: 1.0


import random
import statistics

def jugar(recursos_cara, veces):
    mult_cara = 3
    mult_sello = 1.2
    prob_cara = 0.5
    
    plata = 1000
    for _ in range(veces):
        resultado = random.choices(["Cara", "Sello"], weights=[prob_cara, 1 - prob_cara], k=1)[0]
        
        if resultado == "Cara":
            plata = plata * recursos_cara * mult_cara
        else:
            plata = plata * (1 - recursos_cara) * mult_sello
    
    return plata

n_jugadas = 100
n_simulaciones = 2000  # repeticiones por cada recursos_cara, para promediar

for i in range(11):
    recursos_cara = i / 10
    resultados = [jugar(recursos_cara, n_jugadas) for _ in range(n_simulaciones)]
    mediana = statistics.median(resultados)
    print(f"recursos_cara={recursos_cara:.1f} -> plata promedio final: {mediana:.75f}")


