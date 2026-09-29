# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 16:43:59 2026

@author: Zsofi
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import signal as sig


#%% ECG con ruido

fs_ECG = 1000

# para listar las variables que hay en el archivo
sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead_ruido = mat_struct['ecg_lead']
N = len(ecg_one_lead_ruido)

hb_1 = mat_struct['heartbeat_pattern1']
hb_2 = mat_struct['heartbeat_pattern2']

plt.figure(1, figsize=(15,5))
plt.subplot(1,3,1)
plt.plot(ecg_one_lead_ruido[5000:12000], color="red")
plt.title("ECG con ruido")
plt.axhline(0,color="black")
plt.axvline(0,color="black")
plt.grid()
plt.xlabel("Muestras")
plt.ylabel("Amplitud [mV]")

plt.subplot(1,3,2)
plt.plot(hb_1, color="red")
plt.title("Latido (1er patrón)")
plt.axhline(0,color="black")
plt.axvline(0,color="black")
plt.grid()
plt.xlabel("Muestras")
plt.ylabel("Amplitud [mV]")

plt.subplot(1,3,3)
plt.plot(hb_2, color="red")
plt.title("Latido (2do patrón)")
plt.axhline(0,color="black")
plt.axvline(0,color="black")
plt.grid()
plt.xlabel("Muestras")
plt.ylabel("Amplitud [mV]")
#%% ECG sin ruido

ecg_one_lead = np.load('ecg_sin_ruido.npy')

plt.figure(4)
plt.plot(ecg_one_lead[5000:12000], color="purple")
plt.axhline(0,color="black")
plt.axvline(0,color="black")
plt.grid()
plt.title("ECG sin ruido")
#%% Pletismografía con ruido

fs_ppg = 400

# # # # Cargar el archivo CSV como un array de NumPy
ppg_ruido = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe

plt.figure(5, figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(ppg_ruido, color="gold")
plt.title("PPG con ruido")
plt.axvline(0,color="black")
plt.grid()

#%% Pletismografía sin ruido

ppg = np.load('ppg_sin_ruido.npy')
plt.subplot(1,2,2)
plt.plot(ppg, color="peachpuff")
plt.title("PPG sin ruido")
plt.axhline(0,color="black")
plt.axvline(0,color="black")
plt.grid()


#%% Lectura de audios

# # Cargar el archivo CSV como un array de NumPy
fs_audio1, cucaracha = sio.wavfile.read('la cucaracha.wav')
fs_audio2, prueba = sio.wavfile.read('prueba psd.wav')
fs_audio3, silbido = sio.wavfile.read('silbido.wav')

plt.figure(6, figsize=(15,5))
plt.subplot(1,3,1)
plt.plot(cucaracha, color="coral")
plt.title("Cucharacha")
plt.axhline(0,color="black")
plt.axvline(0,color="black")
plt.grid()
plt.ticklabel_format(axis='x', style='sci', scilimits=(4, 4))

plt.subplot(1,3,2)
plt.plot(prueba, color="crimson")
plt.title("Prueba PSD")
plt.axhline(0,color="black")
plt.axvline(0,color="black")
plt.grid()
plt.ticklabel_format(axis='x', style='sci', scilimits=(4, 4))

plt.subplot(1,3,3)
plt.plot(silbido, "lightcoral")
plt.title("Silbido")
plt.axhline(0,color="black")
plt.axvline(0,color="black")
plt.grid()
plt.ticklabel_format(axis='x', style='sci', scilimits=(4, 4))

# sd.play(wav_data1, fs_audio1)

#%% Welch 
def welch_señal(señal, minimo, maximo, fs, fig, nombre,e):
    N=len(señal)
    L= np.array([N//33, N//35, N//37])
    K=N/L
    arrwelch_señal=[]
    for l in L:
        welch_señal = sig.welch(señal,fs,'hann',l,l/2,N)   
        arrwelch_señal.append(welch_señal)

    plt.figure(fig, figsize=(8, 4))
    plt.subplot(1,2,1)
    for i in range(len(arrwelch_señal)):
        plt.plot(arrwelch_señal[i][0],arrwelch_señal[i][1],label=f'K = {K[i]:.0f}')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('PSD')
    plt.title(nombre)
    plt.legend()
    plt.grid()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(e, e))
    plt.subplot(1,2,2)
    for i in range(len(arrwelch_señal)):
        plt.plot(arrwelch_señal[i][0][minimo:maximo],arrwelch_señal[i][1][minimo:maximo],label=f'K = {K[i]:.0f}')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('PSD')
    plt.title(nombre)
    plt.legend()
    plt.grid()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(e, e))
    return arrwelch_señal, K

welch_ECG, K=welch_señal(ecg_one_lead, 0, 1000, 1000, 7, "Welch: ECG sin ruido", 5) 
ecg_one_lead_ruido = ecg_one_lead_ruido.flatten()
welch_ECG_ruido, K=welch_señal(ecg_one_lead_ruido, 0, 500, 1000, 8, "Welch: ECG con ruido", 8)
welch_ppg, K=welch_señal(ppg, 0, 1000, 400, 9, "Welch: PPG sin ruido", 5)
welch_ppg_ruido, K=welch_señal(ppg_ruido, 0, 1000, 400, 10, "Welch: PPG con ruido",5)
welch_cucaracha, K=welch_señal(cucaracha, 0, 500, 1000, 11, "Welch: Sonido de cucaracha",-6)
welch_prueba, K=welch_señal(prueba, 0, 4000, 1000, 12, "Welch: Sonido de prueba",-5)
welch_silbido, K=welch_señal(silbido, 7500, 20000, 1000, 13, "Welch: Sonido de silbido", -4)

#%% Ancho de bandas

def ancho_banda(welch, K):
    suma=0
    sum_prim=0
    sum_ult=0
    for i in range(len(welch)):
        freq = welch[i][0]
        psd = welch[i][1]
        resol = freq[1] - freq[0]
        potencia = np.cumsum(psd) * resol
        prim_indice = np.searchsorted(potencia,0.005 * potencia[-1])
        ult_indice = np.searchsorted(potencia, 0.995 * potencia[-1])
        prim_freq = freq[prim_indice]
        ult_freq = freq[ult_indice]
        bw = ult_freq - prim_freq
        sum_prim+=prim_freq
        sum_ult+=ult_freq
        suma+=bw
    prom=suma/len(welch)
    prom_prim=sum_prim/3
    prom_ult=sum_ult/3
    return prom_prim, prom_ult, prom
        

prim_ECG, ult_ECG, bw_ECG=ancho_banda(welch_ECG, K)
prim_ECG_ruido, ult_ECG_ruido, bw_ECG_ruido=ancho_banda(welch_ECG_ruido, K)
prim_ppg, ult_ppg, bw_ppg=ancho_banda(welch_ppg, K)
prim_ppg_ruido, ult_ppg_ruido, bw_ppg_ruido=ancho_banda(welch_ppg_ruido, K)
prim_cucaracha, ult_cucaracha, bw_cucaracha=ancho_banda(welch_cucaracha, K)
prim_prueba, ult_prueba, bw_prueba=ancho_banda(welch_prueba, K)
prim_silbido, ult_silbido, bw_silbido=ancho_banda(welch_silbido, K)
    

    
    
    