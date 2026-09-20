# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 19:24:09 2026

@author: ECyT
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as win 

N=1000
fs=1000

def mi_fft(x, fs=fs):
    x_fft = np.fft.fft(x, axis=1)
    x_fft = x_fft[:,:N//2]
    freq = np.fft.fftfreq(N, 1/fs)[:N//2]
    x_fft_abs = np.abs(x_fft)
    
    return freq, x_fft_abs

omega_cero = np.pi/2
f_r = np.random.uniform(-2,2,size=200)

omega_uno = omega_cero + f_r * 2 * np.pi/N
nn = np.arange(0,N,1)
s = np.sqrt(2)*np.sin(np.outer(omega_uno,nn))
Ps = np.var(s)
SNR = 3
sigma = np.sqrt(Ps/np.power(10,SNR/10)) 
n_a = np.random.normal(scale=sigma,size=s.shape)

x = s + n_a

#%% Rectangular 3dB
freq, x_fft_abs = mi_fft(x)
x_fft_abs = 2*x_fft_abs/N
x_fft_abs_traspuesta = np.transpose(x_fft_abs)

plt.figure(1)
plt.plot(freq,x_fft_abs_traspuesta,":.")
plt.xlim(240,260)
plt.title("F1: Senoidales + ruido con ventana implícita (3dB)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral de potencia [V]")

a_1_rect = x_fft_abs[:,freq==(omega_cero/(2*np.pi))*N]

indice_maximo = np.argmax(x_fft_abs, axis=1)
omega_rect = freq[indice_maximo]*2*np.pi/fs

esp_a_1 = np.sum(a_1_rect)/200  #Lo mismo que usar np.mean(a_1_rect)
s_a_1=esp_a_1-np.sqrt(2)
var_a_1 = np.sum((a_1_rect-esp_a_1)**2)/200

esp_ohm_1 = np.sum(omega_rect)/200 
s_ohm_1=np.mean(esp_ohm_1-omega_uno)
var_ohm_1 = np.sum((omega_rect-esp_ohm_1)**2)/200


#%% Flattop 3dB

flat=win.flattop(N)
x_flat=flat*x

freq, x_fft_abs = mi_fft(x_flat)
x_fft_abs = 2*x_fft_abs/N
x_fft_abs_traspuesta = np.transpose(x_fft_abs)
plt.figure(2)
plt.plot(freq,x_fft_abs_traspuesta,":.")
plt.xlim(240,260)
plt.title("F2: Senoidales + ruido con ventana Flattop (3dB)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral de potencia [V]")


a_1_flat = x_fft_abs[:,freq==(omega_cero/(2*np.pi))*N]

indice_maximo = np.argmax(x_fft_abs, axis=1)
omega_flat = freq[indice_maximo]*2*np.pi/fs

esp_a_1 = np.sum(a_1_flat)/200  
s_a_1=esp_a_1-np.sqrt(2)
var_a_1 = np.sum((a_1_flat-esp_a_1)**2)/200

esp_ohm_1 = np.sum(omega_flat)/200 
s_ohm_1=np.mean(esp_ohm_1-omega_uno)
var_ohm_1 = np.sum((omega_flat-esp_ohm_1)**2)/200

#%% Blackmanharris 3dB

blackh=win.blackmanharris(N)
x_blackh=x*blackh

freq, x_fft_abs = mi_fft(x_blackh)
x_fft_abs_norm = 2*x_fft_abs/N
x_fft_abs_traspuesta = np.transpose(x_fft_abs_norm)
plt.figure(3)
plt.plot(freq,x_fft_abs_traspuesta,":.")
plt.xlim(240,260)
plt.title("F3: Senoidales + ruido con ventana Blackmanharris (3dB)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral de potencia [V]")

a_1_blackh = x_fft_abs_norm[:,freq==(omega_cero/(2*np.pi))*N]

indice_maximo = np.argmax(x_fft_abs_norm, axis=1)
omega_blackh = freq[indice_maximo]*2*np.pi/fs

esp_a_1 = np.sum(a_1_blackh)/200 
s_a_1=esp_a_1-np.sqrt(2)
var_a_1 = np.sum((a_1_blackh-esp_a_1)**2)/200

esp_ohm_1 = np.sum(omega_blackh)/200 
s_ohm_1=np.mean(esp_ohm_1-omega_uno)
var_ohm_1 = np.sum((omega_blackh-esp_ohm_1)**2)/200


#%% Hann 3dB
hann=win.hann(N)
x_hann=x*hann

freq, x_fft_abs = mi_fft(x_hann)
x_fft_abs_norm = 2*x_fft_abs/N
x_fft_abs_traspuesta = np.transpose(x_fft_abs_norm)
plt.figure(4)
plt.plot(freq,x_fft_abs_traspuesta,":.")
plt.xlim(240,260)
plt.title("F4: Senoidales + ruido con ventana Hann (3dB)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral de potencia [V]")

a_1_hann = x_fft_abs_norm[:,freq==(omega_cero/(2*np.pi))*N]

indice_maximo = np.argmax(x_fft_abs_norm, axis=1)
omega_hann = freq[indice_maximo]*2*np.pi/fs

esp_a_1 = np.sum(a_1_hann)/200
s_a_1=esp_a_1-np.sqrt(2)
var_a_1 = np.sum((a_1_hann-esp_a_1)**2)/200

esp_ohm_1 = np.sum(omega_hann)/200 
s_ohm_1=np.mean(esp_ohm_1-omega_uno)
var_ohm_1 = np.sum((omega_hann-esp_ohm_1)**2)/200

#%% Histogramas 3dB

plt.figure(5)
plt.hist(a_1_rect, bins=20, color="red", label="Rectangular")
plt.hist(a_1_hann, bins=20, color="green", label="Hann")
plt.hist(a_1_blackh, bins=20, color="gold", label="Blackmanharris")
plt.hist(a_1_flat, bins=20, color="blue", label="Flattop")
plt.title("F5: Histograma del estimador $\widehat{a}_1^i$ (3dB)")
plt.xlabel("Amplitud [V]")
plt.ylabel("Número de resoluciones")
plt.legend()

plt.figure(6)
plt.hist(omega_rect, bins=20, color="red", histtype="step", linewidth=2, label="Rectangular")
plt.hist(omega_flat, bins=20, color="blue", histtype="step", linewidth=2, label="Flattop")
plt.hist(omega_blackh, bins=20, color="gold", histtype="step", linewidth=2, label="Blackmanharris")
plt.hist(omega_hann, bins=20, color="green", histtype="step", linewidth=2, label="Hann")
plt.title("F6: Histograma del estimador $\widehat{Ω}_1^i$ (3dB)")
plt.xlabel(" [Hz]")
plt.ylabel("Número de resoluciones")
plt.legend()


#%% Para SNR=10
SNR = 3
sigma = np.sqrt(Ps/np.power(10,SNR/10)) 

n_a = np.random.normal(scale=sigma,size=s.shape)
    
x = s + n_a
#%% Rectangular 10dB

freq, x_fft_abs = mi_fft(x)
x_fft_abs = 2*x_fft_abs/N
x_fft_abs_traspuesta = np.transpose(x_fft_abs)
plt.figure(7)
plt.plot(freq,x_fft_abs_traspuesta,":.")
plt.xlim(240,260)
plt.title("F7: Senoidales + ruido con ventana implícita (10dB)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral de potencia [V]")

a_1_rect = x_fft_abs[:,freq==(omega_cero/(2*np.pi))*N]

indice_maximo = np.argmax(x_fft_abs, axis=1)
omega_rect = freq[indice_maximo]*2*np.pi/fs

esp_a_1 = np.sum(a_1_rect)/200
s_a_1=esp_a_1-np.sqrt(2)
var_a_1 = np.sum((a_1_rect-esp_a_1)**2)/200

esp_ohm_1 = np.sum(omega_rect)/200 
s_ohm_1=np.mean(esp_ohm_1-omega_uno)
var_ohm_1 = np.sum((omega_rect-esp_ohm_1)**2)/200
#%% Flattop 10dB

flat=win.flattop(N)
x_flat=flat*x

freq, x_fft_abs = mi_fft(x_flat)
x_fft_abs = 2*x_fft_abs/N
x_fft_abs_traspuesta = np.transpose(x_fft_abs)
plt.figure(8)
plt.plot(freq,x_fft_abs_traspuesta,":.")
plt.xlim(240,260)
plt.title("F8: Senoidales + ruido con ventana Flattop (10dB)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral de potencia [V]")

a_1_flat = x_fft_abs[:,freq==(omega_cero/(2*np.pi))*N]

indice_maximo = np.argmax(x_fft_abs, axis=1)
omega_flat = freq[indice_maximo]*2*np.pi/fs

esp_a_1 = np.sum(a_1_flat)/200 
s_a_1=esp_a_1-np.sqrt(2)
var_a_1 = np.sum((a_1_flat-esp_a_1)**2)/200

esp_ohm_1 = np.sum(omega_flat)/200 
s_ohm_1=np.mean(esp_ohm_1-omega_uno)
var_ohm_1 = np.sum((omega_flat-esp_ohm_1)**2)/200
#%% Blackmanharris 10dB

blackh=win.blackmanharris(N)
x_blackh=x*blackh

freq, x_fft_abs = mi_fft(x_blackh)
x_fft_abs_norm = 2*x_fft_abs/N
x_fft_abs_traspuesta = np.transpose(x_fft_abs_norm)
plt.figure(9)
plt.plot(freq,x_fft_abs_traspuesta,":.")
plt.xlim(240,260)
plt.title("F9: Senoidales + ruido con ventana Blackmanharris (3dB)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral de potencia [V]")

a_1_blackh = x_fft_abs_norm[:,freq==(omega_cero/(2*np.pi))*N]

indice_maximo = np.argmax(x_fft_abs_norm, axis=1)
omega_blackh = freq[indice_maximo]*2*np.pi/fs

esp_a_1 = np.sum(a_1_blackh)/200 
s_a_1=esp_a_1-np.sqrt(2)
var_a_1 = np.sum((a_1_blackh-esp_a_1)**2)/200

esp_ohm_1 = np.sum(omega_blackh)/200 
s_ohm_1=np.mean(esp_ohm_1-omega_uno)
var_ohm_1 = np.sum((omega_blackh-esp_ohm_1)**2)/200

#%% Hann 10dB
hann=win.hann(N)
x_hann=x*hann

freq, x_fft_abs = mi_fft(x_hann)
x_fft_abs_norm = 2*x_fft_abs/N
x_fft_abs_traspuesta = np.transpose(x_fft_abs_norm)
plt.figure(10)
plt.plot(freq,x_fft_abs_traspuesta,":.")
plt.xlim(240,260)
plt.title("F3: Senoidales + ruido con ventana Hann (10dB)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral de potencia [V]")

a_1_hann = x_fft_abs_norm[:,freq==(omega_cero/(2*np.pi))*N]

indice_maximo = np.argmax(x_fft_abs_norm, axis=1)
omega_hann = freq[indice_maximo]*2*np.pi/fs

esp_a_1 = np.sum(a_1_hann)/200
s_a_1=esp_a_1-np.sqrt(2)
var_a_1 = np.sum((a_1_hann-esp_a_1)**2)/200

esp_ohm_1 = np.sum(omega_blackh)/200 
s_ohm_1=np.mean(esp_ohm_1-omega_uno)
var_ohm_1 = np.sum((omega_blackh-esp_ohm_1)**2)/200

#%% Histogramas 10dB

plt.figure(11)
plt.hist(a_1_rect, bins=20, color="red", label="Rectangular")
plt.hist(a_1_hann, bins=20, color="green", label="Hann")
plt.hist(a_1_blackh, bins=20, color="gold", label="Blackmanharris")
plt.hist(a_1_flat, bins=20, color="blue", label="Flattop")
plt.title("F5: Histograma del estimador $\widehat{a}_1^i$ (10dB)")
plt.xlabel("Amplitud [V]")
plt.ylabel("Número de resoluciones")
plt.legend()

plt.figure(12)
plt.hist(omega_rect, bins=20, color="red", histtype="step", linewidth=2, label="Rectangular")
plt.hist(omega_flat, bins=20, color="blue", histtype="step", linewidth=2, label="Flattop")
plt.hist(omega_blackh, bins=20, color="gold", histtype="step", linewidth=2, label="Blackmanharris")
plt.hist(omega_hann, bins=20, color="green", histtype="step", linewidth=2, label="Hann")
plt.title("F6: Histograma del estimador $\widehat{Ω}_1^i$ (10dB)")
plt.xlabel(" [Hz]")
plt.ylabel("Número de resoluciones")
plt.legend()

#%% Sesgo y varianza

