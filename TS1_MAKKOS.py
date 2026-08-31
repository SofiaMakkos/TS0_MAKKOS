# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 18:36:14 2026

@author: Zsofi
"""
#%% imports
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import square
import matplotlib.pyplot as plt

N=1000
#%% Señal sinusoidal de 2 KHz que tenga al menos 10 puntos por período.
"""
fs=20000

def mi_funcion_sen( vmax = 1, dc = 0, ff = 1, ph=0, nn = N, fs = fs):
    tt=np.arange(0.0, N/fs, 1/fs)
    xx=vmax*np.sin(2*np.pi*ff*tt +ph)+dc
    return(tt, xx)
tt, xx= mi_funcion_sen(ff=2000)

fftxx = fft(xx)
fftxx = fftxx[:N//2]
#print(fftxx[k])

freq = fftfreq(N,1/fs)
freq = freq[:N//2]

fftxx_abs = np.abs(fftxx)

plt.figure(1)
plt.plot(tt[:10],xx[:10], ":.")
plt.title("Función seno (2Khz)")
   
plt.figure(2)
plt.plot(freq,fftxx_abs, ":.")
plt.title("Modulo FFT seno (2Khz)")

"""
#%% Misma señal con 2 W de potencia media y desfasada en π/2.
"""
fs=20000
N=1000

def mi_funcion_sen( vmax = 2, dc = 0, ff = 1, ph=np.pi/2, nn = N, fs = fs):
    tt=np.arange(0.0, N/fs, 1/fs)
    xx=vmax*np.sin(2*np.pi*ff*tt +ph)
    return(tt, xx)
tt, xx= mi_funcion_sen(ff=2000)

fftxx = fft(xx)
fftxx = fftxx[:N//2]
#print(fftxx[k])

freq = fftfreq(N,1/fs)
freq = freq[:N//2]

fftxx_abs = np.abs(fftxx)

plt.figure(3)
plt.plot(tt[:10],xx[:10], ":.")
plt.title("Función seno (2W)")
   
plt.figure(4)
plt.plot(freq,fftxx_abs, ":.")
plt.title("Modulo FFT seno (2W)")
"""


#%% Una secuencia aleatoria de ruido normalmente distribuido con DC (valor medio) 0V y varianza 0.1 W.
"""
tt=np.arange(N)
fs=1000
norm=np.random.normal(scale=np.sqrt(0.1),size=len(tt))
pnorm=np.var(norm)

print(pnorm)
plt.figure(5)
plt.plot(tt, norm)
plt.title("Distribución normal con 0,1W")

fftnorm = fft(norm)
fftnorm = fftnorm[:N//2]
fftnorm_abs = np.abs(fftnorm)
freq = fftfreq(N,1/fs)
freq = freq[:N//2]

plt.figure(6)
plt.plot(freq,fftnorm_abs, ":.")
plt.title("Módulo FFT Distribución normal con 0,1W")
"""
#%% Una secuencia aleatoria de ruido uniformemente distribuido con DC (valor medio) 0V y varianza 0.1 W. 
"""
fs=1000
tt=np.arange(N)
#media=(a+b)/2=0
#a=-b
#var=(b-a)**2/12=0,1W
#var=(2b)**2/12=0,1W

b=(np.sqrt(1.2))/2
a=-b
unif=np.random.uniform(low=a,high=b,size=len(tt))
varuniform=np.var(unif)
print(a,b,varuniform)

plt.figure(7)
plt.plot(unif)
plt.title("Distribución uniforme con 0,1W")

fftunif = fft(unif)
fftunif = fftunif[:N//2]
fftunif_abs = np.abs(fftunif)
freq = fftfreq(N,1/fs)
freq = freq[:N//2]

plt.figure(8)
plt.plot(freq,fftunif_abs, ":.")
plt.title("Módulo FFT Distribución uniforme con 0,1W")
"""
#%% Un pulso rectangular de la misma frecuencia, 1 W de potencia y ciclo de actividad del 50% (Ver scipy.signal apartado Waveforms).
"""
fs=20000
tt=np.arange(0,N/fs, 1/fs)
cuadr=((square(tt*2*np.pi*2000, duty=0.5)+1)/2)*np.sqrt((2))
pot=np.mean(cuadr**2)

plt.figure(9)
plt.plot(tt[:50], cuadr[:50])
plt.title("Señal cuadrada 1W")

fftcuadr = fft(cuadr)
fftcuadr = fftcuadr[:N//2]
fftcuadr_abs = np.abs(fftcuadr)
freq = fftfreq(N,1/fs)
freq = freq[:N//2]

plt.figure(8)
plt.plot(freq,fftcuadr_abs, ":.")
plt.title("Módulo FFT Señal cuadrada con 1W")
"""

plt.show()
