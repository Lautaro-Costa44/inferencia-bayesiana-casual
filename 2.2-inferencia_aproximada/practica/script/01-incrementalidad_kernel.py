"""
Practica 2.2 - Ejercicio 1, opcional: modelo temporal (figura 3 del enunciado) en PyMC,
comparado con el pasaje de mensajes de 01-pasaje_mensajes.py
Genera en figuras-01/: 1-5_modelo_temporal.png
"""
import importlib
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from matplotlib import pyplot as plt
from scipy.stats import beta

AQUI = Path(__file__).resolve().parent
(AQUI / "figuras-01").mkdir(exist_ok=True)  # carpeta de las figuras, se crea si no existe

# Datos: por cada mes, n = conversiones y N = usuarios de cada grupo
data = pd.read_csv(AQUI.parent / "datos" / "01-Incrementalidad.csv", encoding="utf-8-sig")

def serie(tipo):
    return data[data["Tipo"] == tipo]["online_schedules"].to_numpy()

meses = data["Mes"].unique()
T = len(meses)
n_C, N_C = serie("Eventos CG"), serie("Usuarios CG")
n_T, N_T = serie("Eventos Tratamiento"), serie("Usuarios Tratamiento")


# Modelo (una cadena para control y otra para tratamiento, cada una con su kappa):
#   log10(kappa) ~ U[0, 7]
#   p_0 ~ Beta(1, 20)
#   p_t ~ Beta(kappa * p_{t-1}, kappa * (1 - p_{t-1}))   el mes t se parece al anterior
#   n_t ~ Binomial(N_t, p_t)
# kappa grande = los meses casi no cambian, kappa chico = cada mes puede ser muy distinto

def cadena(nombre, n, N):
    log10_kappa = pm.Uniform(f"log10_kappa_{nombre}", 0, 7)
    kappa = 10 ** log10_kappa
    p = [pm.Beta(f"{nombre}_0", alpha=1, beta=20)]
    for t in range(1, T):
        p.append(pm.Beta(f"{nombre}_{t}", alpha=kappa * p[t - 1], beta=kappa * (1 - p[t - 1])))
    p = pm.Deterministic(nombre, pt.stack(p))  # junta los T meses en un solo vector
    pm.Binomial(f"n_{nombre}", n=N, p=p, observed=n)

with pm.Model():
    cadena("p", n_C, N_C)  # control
    cadena("q", n_T, N_T)  # tratamiento
    # cores=1 porque en windows el muestreo en paralelo falla
    traza = pm.sample(2000, tune=2000, chains=4, cores=1, target_accept=0.99)


# Diagnosticos: divergencias (tienen que ser 0), r_hat (<= 1.01) y ESS (cantidad de muestras efectivas)
resumen = az.summary(traza)
print("divergencias:", int(traza.sample_stats["diverging"].sum()))
print("r_hat maximo:", round(resumen["r_hat"].max(), 3))
print("ESS minimo:", int(resumen["ess_bulk"].min()))
print(resumen.loc[["log10_kappa_p", "log10_kappa_q"], ["mean", "sd", "ess_bulk", "r_hat"]])


# P(q_t > p_t): en cada muestra del posterior comparamos q_t con p_t del mismo mes
p = traza.posterior["p"].values.reshape(-1, T)  # junta cadenas y muestras: (muestras, meses)
q = traza.posterior["q"].values.reshape(-1, T)
p_kernel = (q > p).mean(axis=0)

# Para comparar, cada mes por separado: posterior Beta(1 + n, 20 + N - n)
q_mes = np.random.beta(1 + n_T, 20 + N_T - n_T, (100000, T))
p_mes_sep = np.random.beta(1 + n_C, 20 + N_C - n_C, (100000, T))
p_mes = (q_mes > p_mes_sep).mean(axis=0)

print("\nmes, P(q>p) modelo temporal, P(q>p) mes por separado")
for t, mes in enumerate(meses):
    print(mes, round(p_kernel[t], 3), round(p_mes[t], 3))

# Traemos los posteriors del pasaje de mensajes. Se usa importlib porque el nombre del
# archivo empieza con un numero y no se puede escribir "import 01-pasaje_mensajes".
# Ojo: al importarlo se corre ese script entero (y se regeneran sus figuras).
mensajes = importlib.import_module("01-pasaje_mensajes")
post_msj = {"Control": mensajes.posteriors["Control (p)"],
            "Tratamiento": mensajes.posteriors["Tratamiento (q)"]}
# media de una Beta(a, b) = a / (a + b)
media_msj = {k: v[:, 0] / v.sum(axis=1) for k, v in post_msj.items()}

print("\nmedia posterior por mes: p PyMC, p mensajes, q PyMC, q mensajes")
for t, mes in enumerate(meses):
    print(mes, round(p[:, t].mean(), 5), round(media_msj["Control"][t], 5),
          round(q[:, t].mean(), 5), round(media_msj["Tratamiento"][t], 5))


# Graficos: arriba los posteriors, en el medio P(q > p), abajo el posterior de kappa
x = np.arange(T)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 11))
grupos = [("Control", p, n_C, N_C), ("Tratamiento", q, n_T, N_T)]
for c, (nombre, muestras, n, N) in enumerate(grupos):
    ax1.plot(x, muestras.mean(axis=0), color=f"C{c}", lw=2, label=f"{nombre} (media posterior)")
    ax1.fill_between(x, *np.percentile(muestras, [2.5, 97.5], axis=0), color=f"C{c}", alpha=0.2)
    ax1.plot(x, n / N, "o", color=f"C{c}", label=f"{nombre} (frecuencia observada)")
    a, b = post_msj[nombre][:, 0], post_msj[nombre][:, 1]
    ax1.plot(x, a / (a + b), "--", color=f"C{c}", label=f"{nombre} (pasaje de mensajes)")
    ax1.plot(x, beta.ppf(0.025, a, b), ":", color=f"C{c}", lw=1)
    ax1.plot(x, beta.ppf(0.975, a, b), ":", color=f"C{c}", lw=1)
ax1.set_xticks(x, meses)
ax1.set_ylabel("Desempeño")
ax1.legend(fontsize=8)

ax2.plot(x, p_kernel, "k-o", label="Modelo temporal")
ax2.plot(x, p_mes, "o--", color="gray", label="Mes por separado")
ax2.axhline(0.95, color="k", ls=":", lw=1)
ax2.set_xticks(x, meses)
ax2.set_ylim(0, 1.02)
ax2.set_ylabel("P(q > p | datos)")
ax2.legend(fontsize=8)

for c, nombre in enumerate(["p", "q"]):
    ax3.hist(traza.posterior[f"log10_kappa_{nombre}"].values.ravel(), bins=50, range=(0, 7),
             density=True, color=f"C{c}", alpha=0.5, label=nombre)
ax3.set_xlabel("log10(kappa)")
ax3.set_ylabel("P(log10 kappa | datos)")
ax3.legend(fontsize=8)

plt.tight_layout()
plt.savefig(AQUI / "figuras-01" / "1-5_modelo_temporal.png", dpi=150)
