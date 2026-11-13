# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 21:06:19 2026

@author: Joaco
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate
import scipy.io as sio

sio.whosmat('ECG_TP4.mat')
mat_struct=sio.loadmat('./ECG_TP4.mat')

ecg_one_lead =mat_struct ['ecg_lead'] * 1.0
ecg_one_lead = ecg_one_lead.flatten()
s = ecg_one_lead

ecg_QRS = mat_struct ['qrs_detections']
ecg_QRS  = ecg_QRS.flatten()

#%% Mediana

b_intermedio = signal.medfilt(s,201)
x_sombrero_int = s - b_intermedio

b_sombrero = signal.medfilt(b_intermedio,601)
x_sombrero = s - b_sombrero
"""
plt.figure(1)
plt.plot(s[8000:13000],label="Señal original (8000 a 130000)")
plt.plot(x_sombrero_int[8000:13000],label="Señal intermedia (8000 a 130000)")
plt.plot(x_sombrero[8000:13000],label="Señal Filtrada (8000 a 130000)")
plt.plot(b_sombrero[8000:13000], label="B final")
plt.legend()
plt.grid()
plt.title("FIltración de señal paso a paso (8000 a 130000)")
plt.xlabel("Muestras [#]")
plt.ylabel("Amplitud [uV]")


plt.figure(2)
plt.plot(s[:5000],label="Señal original (0 a 5000)")
plt.plot(x_sombrero_int[:5000],label="Señal intermedia (0 a 5000)")
plt.plot(x_sombrero[:5000],label="Señal Filtrada (0 a 5000)")
plt.plot(b_sombrero[:5000], label="B final")
plt.legend()
plt.grid()
plt.title("FIltración de señal paso a paso (0 a 5000)")
plt.xlabel("Muestras [#]")
plt.ylabel("Amplitud [uV]")
"""


#%% Spline cúbica
"""
zonasmuertas_start = ecg_QRS-60
zonasmuertas_end = ecg_QRS-40

zonasmuertas = np.array([x for par in zip(zonasmuertas_start, zonasmuertas_end) for x in par])

pq_hat = np.array([np.mean(ecg_one_lead[a:b]) for a,b in zip(zonasmuertas_start,zonasmuertas_end)]).astype(int)
pq_n_hat = np.array([np.mean([a,b]) for a,b in zip(zonasmuertas_start,zonasmuertas_end)])

plt.figure(4)
plt.plot(ecg_one_lead)
plt.plot(pq_n_hat, pq_hat, marker = ".", linestyle = " ")

prueba = zonasmuertas[zonasmuertas<6000]

plt.figure(5)
plt.plot(ecg_one_lead[:6000])
plt.vlines(x=prueba, ymin=np.min(ecg_one_lead), ymax=np.max(ecg_one_lead), color='red', linestyle='--', linewidth=0.8, alpha=0.7, label='QRS')

inter = interpolate.CubicSpline(pq_n_hat, pq_hat)

x = np.arange(0,len(ecg_one_lead),1)
linea_base = inter(x)

ecg_filtrado_spline = ecg_one_lead -linea_base

plt.figure(6)
plt.plot(ecg_one_lead, color="gold", label="Original")
plt.plot(ecg_filtrado_spline,color="red", label = "Filtrado por spline")
plt.plot(linea_base, color="green", linestyle="--")
plt.title("ECG original y filtrado mediante Spline")
plt.legend()
plt.grid()
plt.xlabel("Muestras [#]")
plt.ylabel("Amplitud [uV]")

plt.figure(7)
plt.subplot(3,1,1)
plt.plot(ecg_one_lead[8000: 13000],color="gold", label="Original (8000 a 13000)")
plt.plot(ecg_filtrado_spline[8000: 13000], color="red",label = "Filtrado por spline (8000 a 13000)")
plt.plot(linea_base[8000: 13000], color="green", linestyle="--")
plt.legend()
plt.title("ECG original y filtrado mediante Spline (8000 a 13000)")
plt.grid()
plt.xlabel("Muestras [#]")
plt.ylabel("Amplitud [uV]")

plt.subplot(3,1,2)
plt.plot(ecg_one_lead[3000: 6000],color="gold", label="original")
plt.plot(ecg_filtrado_spline[3000: 6000],color="red", label = "filtrado")
plt.plot(linea_base[3000: 6000], color="green", linestyle="--")
plt.legend()
plt.title("ECG original y filtrado mediante Spline (3000 a 6000)")
plt.xlabel("Muestras [#]")
plt.ylabel("Amplitud [uV]")
plt.grid()

plt.subplot(3,1,3)
plt.plot(ecg_one_lead[100000: 120000],color="gold", label="Original (100000 a 120000)")
plt.plot(ecg_filtrado_spline[100000: 120000], color="red",label = "Filtrado por spline (100000 a 120000)")
plt.plot(linea_base[100000: 120000], color="green", linestyle="--")
plt.legend()
plt.title("ECG original y filtrado mediante Spline (100000 a 120000)")
plt.grid()
plt.xlabel("Muestras [#]")
plt.ylabel("Amplitud [uV]")

plt.show()
"""
#%% Filtro adaptado (matched filter)

ecg_pat = mat_struct ['qrs_pattern1']
ecg_pat  = ecg_pat.flatten()*1.0

corr = signal.lfilter(ecg_pat[::-1], 1, x_sombrero)

x_sombrero = (x_sombrero-np.mean(x_sombrero))/(np.std(x_sombrero))
corr = (corr-np.mean(corr))/(np.std(corr))

plt.figure(10)
plt.plot(x_sombrero[:15000], color="red", label="ECG sin ruido")
plt.plot(np.abs(-corr)[70:15000], color="blue", label="Correlación (invertida + adelantada + abs)")
plt.grid()
plt.title("Correlación y ECG sin ruido")
plt.legend()


fs=1000

corr=np.abs(-corr)[70:]

y_b=signal.order_filter(corr, np.ones(31), 29)
peaks, prop=signal.find_peaks(corr, distance=400, height=2)
alt=prop['peak_heights']

plt.figure(12)
plt.plot(x_sombrero, color="red", label="ECG sin ruido")
plt.plot(corr, color="blue", label="Correlación (invertida + corrida + abs)")
plt.plot(y_b, color="green", label="Filtrado")
#plt.vlines(x=peaks[:30], ymin=np.min(x_sombrero), ymax=np.max(x_sombrero), color='hotpink', linestyle='--', linewidth=0.8, alpha=0.7, label='QRS')
plt.scatter(peaks, alt, color="hotpink", marker="D")
plt.grid()
plt.title("Correlación y ECG sin ruido")
plt.legend()
plt.show()

# Convertir los vectores a arrays
referencia = np.array(ecg_QRS).flatten()
detecciones = np.array(peaks).flatten()

# Ordenar las posiciones
referencia = np.sort(referencia)
detecciones = np.sort(detecciones)

# Frecuencia de muestreo
fs = 1000  # Reemplazar por la real
tolerancia = int(0.1 * fs)

# Inicializar contadores
TP = 0
FP = 0
FN = 0

# Registrar qué detecciones ya fueron utilizadas
usados = np.zeros(len(detecciones), dtype=bool)

for r in referencia:
    distancias = np.abs(detecciones - r)
    # Buscar detecciones dentro de la tolerancia
    candidatos = np.where((distancias <= tolerancia) & (~usados))[0]
    if len(candidatos) > 0:
        # Elegir la detección más cercana
        mejor = candidatos[np.argmin(distancias[candidatos])]
        TP += 1
        usados[mejor] = True
    else:
        FN += 1
# Las detecciones que no se utilizaron son falsos positivos
FP = np.sum(~usados)

print("TP:", TP)
print("FP:", FP)
print("FN:", FN)

sensibilidad = TP / (TP + FN)
VPP = TP / (TP + FP)

print("Sensibilidad:", sensibilidad)
print("Valor predictivo positivo:", VPP)
print("Sensibilidad: {:.2f}%".format(sensibilidad * 100))
print("VPP: {:.2f}%".format(VPP * 100))

