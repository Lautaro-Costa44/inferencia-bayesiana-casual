"""
Practica 2.2 - Ejercicio 1: pasaje de mensajes con descuento lambda.
Genera en figuras-01/: 1-2_log_evidencia.png, 1-3_posteriors.png y 1-4_p_q_mayor_p.png
"""
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import beta, betabinom

np.random.seed(0)
AQUI = Path(__file__).resolve().parent  # carpeta de este archivo, para que ande desde cualquier lado
(AQUI / "figuras-01").mkdir(exist_ok=True)  # carpeta de las figuras, se crea si no existe

# Datos: por cada mes, n = conversiones y N = usuarios de cada grupo
data = pd.read_csv(AQUI.parent / "datos" / "01-Incrementalidad.csv", encoding="utf-8-sig")

def serie(tipo):
    return data[data["Tipo"] == tipo]["online_schedules"].to_numpy()

meses = data["Mes"].unique()
T = len(meses)
n_C, N_C = serie("Eventos CG"), serie("Usuarios CG")                    # control
n_T, N_T = serie("Eventos Tratamiento"), serie("Usuarios Tratamiento")  # tratamiento

PRIOR = np.array([1, 20])  # prior Beta(1, 20) del primer mes


# 1) Pasaje de mensajes
# Cada mensaje es una Beta, la guardamos como sus dos conteos (a, b).
# Multiplicar dos Betas = sumar los conteos. Descontar = multiplicar los conteos por lambda.
# use _lambda porque "lambda" es una palabra reservada de python.

def pasaje_de_mensajes(n, N, _lambda):
    likelihood = np.column_stack([n, N - n]).astype(float)  # (exitos, fracasos) de cada mes
    forwardPrior = np.zeros((T, 2))
    backwardPrior = np.zeros((T, 2))  # el ultimo mes no recibe nada del futuro: (0, 0)

    # ida: lo que sabemos del mes t lo pasamos (descontado) al mes t+1
    forwardPrior[0] = PRIOR
    for t in range(T - 1):
        forwardPosterior = forwardPrior[t] + likelihood[t]
        forwardPrior[t + 1] = _lambda * forwardPosterior

    # vuelta: lo mismo pero del mes t al mes t-1
    for t in range(T - 1, 0, -1):
        backwardPosterior = backwardPrior[t] + likelihood[t]
        backwardPrior[t - 1] = _lambda * backwardPosterior

    withinPrior = forwardPrior + backwardPrior  # todo lo que se sabe del mes t menos su propio dato
    posterior = withinPrior + likelihood
    return withinPrior, posterior


def log_evidencia(n, N, _lambda):
    # Prediccion de cada mes con todos los demas meses (ecuacion 6).
    # Beta-Binomial = probabilidad de n exitos en N intentos con p ~ Beta(withinPrior)
    withinPrior, _ = pasaje_de_mensajes(n, N, _lambda)
    return betabinom.logpmf(n, N, withinPrior[:, 0], withinPrior[:, 1]).sum()


# 2) Elegir lambda: el que mejor predice los datos, por separado para control y tratamiento
lambdas = np.linspace(0.001, 1, 1000)
cadenas = {"Control (p)": (n_C, N_C), "Tratamiento (q)": (n_T, N_T)}

log_ev = {}
lambda_opt = {}
for nombre, (n, N) in cadenas.items():
    log_ev[nombre] = np.array([log_evidencia(n, N, l) for l in lambdas])
    lambda_opt[nombre] = lambdas[np.argmax(log_ev[nombre])]
    print(nombre, "lambda =", round(lambda_opt[nombre], 3),
          "log-evidencia =", round(log_ev[nombre].max(), 2))

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for c, (ax, nombre) in enumerate(zip(axs, cadenas)):
    ax.plot(lambdas, log_ev[nombre], color=f"C{c}")
    ax.axvline(lambda_opt[nombre], color="k", ls=":", label=f"lambda = {lambda_opt[nombre]:.3f}")
    # recorto el eje y cerca del maximo, si no con lambda chico la curva cae tanto que no se ve nada
    ax.set_ylim(log_ev[nombre].max() - 50, log_ev[nombre].max() + 5)
    ax.set_xlabel("lambda")
    ax.set_ylabel("log P(Datos | lambda)")
    ax.set_title(nombre)
    ax.legend()
plt.tight_layout()
plt.savefig(AQUI / "figuras-01" / "1-2_log_evidencia.png", dpi=150)


# 3) Posterior de cada mes con el lambda elegido
posteriors = {}
for nombre, (n, N) in cadenas.items():
    _, posteriors[nombre] = pasaje_de_mensajes(n, N, lambda_opt[nombre])

x = np.arange(T)
plt.figure(figsize=(10, 5))
for c, (nombre, (n, N)) in enumerate(cadenas.items()):
    a, b = posteriors[nombre][:, 0], posteriors[nombre][:, 1]
    plt.plot(x, beta.ppf(0.5, a, b), color=f"C{c}", lw=2, label=f"{nombre}: mediana posterior")
    plt.fill_between(x, beta.ppf(0.025, a, b), beta.ppf(0.975, a, b), color=f"C{c}", alpha=0.2,
                     label=f"{nombre}: IC 95%")
    plt.plot(x, n / N, "o", color=f"C{c}", label=f"{nombre}: frecuencia observada")
plt.xticks(x, meses)
plt.ylabel("Tasa de conversion")
plt.title("Posterior mes a mes (pasaje de mensajes)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(AQUI / "figuras-01" / "1-3_posteriors.png", dpi=150)


# 4) P(q_t > p_t) de cada mes, conectando los meses vs cada mes por separado
# Control y tratamiento son independientes, asi que sacamos muestras de cada Beta
# y contamos en que proporcion la de tratamiento es mayor.
M = 200000  # cantidad de muestras

def prob_q_mayor_p(post_p, post_q):
    probs = []
    for t in range(T):
        muestras_q = np.random.beta(post_q[t, 0], post_q[t, 1], M)
        muestras_p = np.random.beta(post_p[t, 0], post_p[t, 1], M)
        probs.append(np.mean(muestras_q > muestras_p))
    return np.array(probs)

prob_mensajes = prob_q_mayor_p(posteriors["Control (p)"], posteriors["Tratamiento (q)"])

# Modelo base: cada mes solo, posterior Beta(1 + n, 20 + N - n)
base_p = np.column_stack([PRIOR[0] + n_C, PRIOR[1] + N_C - n_C])
base_q = np.column_stack([PRIOR[0] + n_T, PRIOR[1] + N_T - n_T])
prob_base = prob_q_mayor_p(base_p, base_q)

print("\nmes, P(q>p) pasaje de mensajes, P(q>p) mes por separado")
for t, mes in enumerate(meses):
    print(mes, round(prob_mensajes[t], 4), round(prob_base[t], 4))

plt.figure(figsize=(10, 4))
plt.plot(x, prob_mensajes, "k-o", label="Pasaje de mensajes (meses conectados)")
plt.plot(x, prob_base, "o--", color="gray", label="Modelo base (cada mes por separado)")
plt.axhline(0.95, color="k", ls=":", lw=1)
plt.xticks(x, meses)
plt.ylim(0, 1.02)
plt.ylabel("P(q_t > p_t | Datos)")
plt.legend()
plt.tight_layout()
plt.savefig(AQUI / "figuras-01" / "1-4_p_q_mayor_p.png", dpi=150)

plt.show()
