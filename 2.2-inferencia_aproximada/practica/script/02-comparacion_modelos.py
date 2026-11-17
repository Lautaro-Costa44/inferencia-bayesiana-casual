"""
Practica 2.2 - Ejercicio 2: experimento con datos escasos y corrompidos.
Genera en figuras-02/: 2-4a_operadores.png, 2-4b_desempenos.png y la tabla del punto 4c en consola.
"""
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

MUESTRAS = 20000  # con 5000 el r_hat daba mayor a 1.01, por eso lo subi

# Datos: para cada operador (fila) y contexto (columna), cuantas de las 3 muestras dieron positivo
AQUI = Path(__file__).resolve().parent
(AQUI / "figuras-02").mkdir(exist_ok=True)  # carpeta de las figuras, se crea si no existe
data = pd.read_csv(AQUI.parent / "datos" / "02-positivos_por_cepas_verdaderos.csv")
contextos = ["20par-DTU1", "20par-DTU2", "20par-DTU6", "50par-DTU1", "50par-DTU2", "50par-DTU6"]
cepas = ["DTU1", "DTU2", "DTU6"]
N = 3
# el csv tiene el porcentaje de positivos, lo paso a cantidad (0, 1, 2 o 3)
n = np.round(data[contextos].to_numpy() / 100 * N).astype(int)  # 14 operadores x 6 contextos
n_op = n.shape[0]


# 1) Modelo propuesto
#   p_c ~ Beta(2, 2)            desempeño de cada contexto (igual que en los otros modelos)
#   p50_k = max(p20_k, p50_k)   en concentracion alta se detecta al menos lo mismo que en baja
#   q ~ Beta(1, 9)              probabilidad de que un operador trabaje mal
#   r_i ~ Beta(1, 1)            cuanto baja el desempeño el operador i si trabaja mal
#   C_i ~ Bernoulli(q)          1 si el operador i trabajo mal
#   n_ci ~ Binomial(3, p_c)        si C_i = 0
#   n_ci ~ Binomial(3, p_c * r_i)  si C_i = 1
# Supuestos: (a) cada operador falla en distinta medida (un r_i para cada uno)
#            (b) mas parasito en la muestra no puede bajar la deteccion


# 2) Modelos
# Cada funcion recibe los 6 desempeños p y devuelve log P(Datos | parametros, modelo)

def base(p):
    # todos trabajan bien
    return pm.logp(pm.Binomial.dist(n=N, p=p), n).sum()


def log_mezcla(p, q, r):
    # No sabemos si cada operador trabajo bien o mal, asi que sumamos los dos casos:
    #   P(n_i) = (1 - q) * P(n_i | bien) + q * P(n_i | mal)
    # Todo en logaritmos. sum(axis=1) suma los 6 contextos de cada operador.
    log_bien = pt.log(1 - q) + pm.logp(pm.Binomial.dist(n=N, p=p), n).sum(axis=1)
    log_mal = pt.log(q) + pm.logp(pm.Binomial.dist(n=N, p=p * r), n).sum(axis=1)
    log_total = pt.logaddexp(log_bien, log_mal)  # logaddexp(a, b) = log(e^a + e^b)
    prob_mal = pt.exp(log_mal - log_total)       # P(C_i = 1 | parametros, datos), para el 4a
    return log_total.sum(), prob_mal


def operadores(p):
    # modelo del enunciado: un solo r para todos los que trabajan mal
    q = pm.Beta("q", alpha=1, beta=9)
    r = pm.Beta("r", alpha=1, beta=1)
    log_vero, prob_mal = log_mezcla(p, q, r)
    pm.Deterministic("mal_op", prob_mal)  # se guarda para poder leerlo despues de muestrear
    return log_vero


def propuesto(p):
    q = pm.Beta("q_prop", alpha=1, beta=9)
    r = pm.Beta("r_prop", alpha=1, beta=1, shape=n_op)  # (a) un r_i por operador
    # (b) los primeros 3 desempeños son a 20 par.Eq/mL y los ultimos 3 a 50 par.Eq/mL
    p20, p50 = p[:3], p[3:]
    p_c = pm.Deterministic("p_prop", pt.concatenate([p20, pt.maximum(p20, p50)]))
    # r tiene 14 valores y p_c tiene 6. Con r[:, None] r pasa a ser una columna (14 x 1)
    # y cada fila (operador) queda multiplicada por su propio r_i
    log_vero, prob_mal = log_mezcla(p_c, q, r[:, None])
    pm.Deterministic("mal_prop", prob_mal)
    return log_vero


modelos = {"Base": base, "Operadores": operadores, "Propuesto": propuesto}
# nombre de las variables que usa el punto 4 en cada modelo: (P(C_i = 1), los 6 desempeños)
variables = {"Base": (None, "p"), "Operadores": ("mal_op", "p"), "Propuesto": ("mal_prop", "p_prop")}


# 2 y 3) Comparacion con un selector M y diagnosticos de convergencia

def diagnostico(traza):
    resumen = az.summary(traza)
    divergencias = int(traza.sample_stats.diverging.sum())
    print("  divergencias:", divergencias, " r_hat max:", round(resumen["r_hat"].max(), 4))


def selector(prior_M):
    with pm.Model():
        p = pm.Beta("p", alpha=2, beta=2, shape=len(contextos))  # compartido por los 3 modelos
        log_vero = pt.stack([modelo(p) for modelo in modelos.values()])
        M = pm.Categorical("M", p=prior_M)          # M elige que modelo genero los datos
        pm.Potential("verosimilitud", log_vero[M])  # solo cuenta la verosimilitud del modelo elegido
        # cores=1 porque en windows el muestreo en paralelo falla
        traza = pm.sample(MUESTRAS, tune=2000, chains=4, cores=1, target_accept=0.95)

    # la proporcion de muestras en cada modelo es P(M | Datos)
    M = traza.posterior["M"].values.ravel()
    visitas = np.bincount(M, minlength=len(modelos))
    diagnostico(traza)
    print("  visitas:", dict(zip(modelos, visitas.tolist())))
    return visitas / visitas.sum()


# Si un modelo casi no se visita no se puede estimar bien su evidencia. Entonces se repite
# con mas prior para los modelos poco visitados, y despues se descuenta ese prior:
#   P(M | D) proporcional a P(D | M) P(M)  =>  log P(D | M) = log P(M | D) - log P(M) + cte
prior_M = np.ones(len(modelos)) / len(modelos)  # primero prior uniforme
for intento in range(5):
    print("\nSelector, intento", intento + 1)
    posterior_M = selector(prior_M)
    # 1e-4 para no hacer log(0) si un modelo no se visito nunca
    log_evidencia = np.log(np.maximum(posterior_M, 1e-4)) - np.log(prior_M)
    print("  prior M:", np.round(prior_M, 4), " posterior M:", np.round(posterior_M, 4))
    if posterior_M.min() > 0.05:  # todos los modelos bien visitados
        break
    prior_M = np.exp(-log_evidencia) / np.exp(-log_evidencia).sum()

log_evidencia -= log_evidencia.max()  # el mejor modelo queda en 0
prob_M = np.exp(log_evidencia) / np.exp(log_evidencia).sum()  # P(M | Datos) con prior uniforme
print("\nmodelo, log-evidencia relativa, P(M | Datos)")
for nombre, le, pr in zip(modelos, log_evidencia, prob_M):
    print(nombre, round(le, 2), round(pr, 4))


# 4) Analisis con el modelo de mayor evidencia
# Se corre ese modelo solo (sin el selector) para tener su posterior
mejor = list(modelos)[int(np.argmax(log_evidencia))]
print("\n4) modelo de mayor evidencia:", mejor)
with pm.Model():
    p = pm.Beta("p", alpha=2, beta=2, shape=len(contextos))
    pm.Potential("verosimilitud", modelos[mejor](p))
    traza = pm.sample(MUESTRAS, tune=2000, chains=4, cores=1, target_accept=0.95)
diagnostico(traza)

var_mal, var_p = variables[mejor]
post = traza.posterior
# traza.posterior tiene forma (cadenas, muestras, 6). Juntamos cadenas y muestras:
# cada fila de P es una muestra de los 6 desempeños juntos
P = post[var_p].values.reshape(-1, len(contextos))

# 4a) Probabilidad de que cada operador haya trabajado mal
if var_mal is not None:  # el modelo base no tiene operadores que trabajen mal
    prob_mal = post[var_mal].mean(("chain", "draw")).values
    plt.figure(figsize=(8, 3))
    plt.bar(data["Laboratorio"], prob_mal)
    plt.ylim(0, 1)
    plt.ylabel("P(C_i = 1 | Datos)")
    plt.title(f"4a) Probabilidad de que cada operador trabajo mal ({mejor})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(AQUI / "figuras-02" / "2-4a_operadores.png", dpi=150)

# 4b) Posterior de cada desempeño, un grafico por concentracion
fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
for j, (ax, conc) in enumerate(zip(axs, ["20 par.Eq/mL", "50 par.Eq/mL"])):
    for k, cepa in enumerate(cepas):
        x = P[:, 3 * j + k]  # columnas 0 a 2: 20 par.Eq/mL, columnas 3 a 5: 50 par.Eq/mL
        lo, hi = np.quantile(x, [0.05, 0.95])
        ax.hist(x, bins=100, range=(0, 1), density=True, histtype="step",
                label=f"{cepa} [{lo:.3f}, {hi:.3f}]")
    ax.set_title(f"4b) {conc}")
    ax.set_xlabel("PPA")
    ax.legend(title="IC 5%-95%")
axs[0].set_ylabel("Densidad")
plt.tight_layout()
plt.savefig(AQUI / "figuras-02" / "2-4b_desempenos.png", dpi=150)

# 4c) P(p_i > p_j | Datos) en concentracion baja
# Se compara dentro de la misma fila de P (misma muestra) y no las marginales por separado,
# porque los desempeños no son independientes en el posterior
tabla = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        tabla[i, j] = np.mean(P[:, i] > P[:, j])
print("\n4c) P(p_fila > p_columna | Datos), 20 par.Eq/mL:")
print(pd.DataFrame(tabla, index=cepas, columns=cepas).round(4))

plt.show()
