# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 19:36:25 2026

@author: ECyT
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

#%%
N=1000
k0=N/4
fs=1000
f0=k0*fs/N

#%%

def mi_funcion_sen( vmax = np.sqrt(2), dc = 0, ff = 1, ph=0, nn = N, fs = fs):
    tt=np.arange(0.0, N/fs, 1/fs)
    xx=vmax*np.sin(2*np.pi*ff*tt +ph)+dc
    return tt, xx

def fastfouriertrans (xx, N, fs):
    fftxx = fft(xx)
    #fftxx = fftxx[:N//2]
    freq = fftfreq(N,1/fs)
    freq = freq[:N//2]
    fftxx_abs = np.abs(fftxx)
    return freq, fftxx_abs
#%%
tt1, sen1=mi_funcion_sen(ff=f0)

plt.figure(1)
plt.plot(tt1[:24],sen1[:24], ":.",color="red")
plt.title("Seno k0=N/4")
plt.ylabel("Amplitud [V]")
plt.xlabel("Tiempo [s]")

k0=N/4+0.25
f0=k0*fs/N
tt2, sen2=mi_funcion_sen(ff=f0)

plt.figure(2)
plt.plot(tt2[:24],sen2[:24], ":.",color="green")
plt.title("Seno k0=N/4+0.25")
plt.ylabel("Amplitud [V]")
plt.xlabel("Tiempo [s]")

k0=N/4+0.5
f0=k0*fs/N
tt3, sen3=mi_funcion_sen(ff=f0)

plt.figure(3)
plt.plot(tt3[:24],sen3[:24], ":.",color="blue")
plt.title("Seno k0=N/4+0.5")
plt.ylabel("Amplitud [V]")
plt.xlabel("Tiempo [s]")
#%%

freq1, fftsen1_abs=fastfouriertrans(sen1, N, fs)
freq2, fftsen2_abs=fastfouriertrans(sen2, N, fs)
freq3, fftsen3_abs=fastfouriertrans(sen3, N, fs)

ref=np.max(fftsen1_abs)
fftsen1_abs_dB=20*np.log10(fftsen1_abs/ref)
fftsen2_abs_dB=20*np.log10(fftsen2_abs/ref)
fftsen3_abs_dB=20*np.log10(fftsen3_abs/ref)

plt.figure(4)
plt.plot(freq1, fftsen1_abs_dB[:N//2],":.", color="red")
plt.title("FFT normalizada Seno k0=N/4")
plt.ylabel("Densidad espectral de potencia [dB]")
plt.xlabel("Frecuencia [Hz]")
plt.figure(5)
plt.plot(freq2, fftsen2_abs_dB[:N//2],":.", color="green")
plt.title("FFT normalizada Seno k0=N/4+0.25")
plt.ylabel("Densidad espectral de potencia [dB]")
plt.xlabel("Frecuencia [Hz]")
plt.figure(6)
plt.plot(freq3, fftsen3_abs_dB[:N//2],":.", color="blue")
plt.title("FFT normalizada Seno k0=N/4+0.5")
plt.ylabel("Densidad espectral de potencia [dB]")
plt.xlabel("Frecuencia [Hz]")

plt.figure(7)
plt.plot(freq1, fftsen1_abs[:N//2]/N, ":.", color="red")
plt.title("FFT normalizada Seno k0=N/4")
plt.ylabel("Densidad espectral de potencia [V]")
plt.xlabel("Frecuencia [Hz]")

plt.figure(8)
plt.plot(freq2, fftsen2_abs[:N//2]/N,":.", color="green")
plt.title("FFT normalizada Seno k0=N/4+0.25")
plt.ylabel("Densidad espectral de potencia [V]")
plt.xlabel("Frecuencia [Hz]")

plt.figure(9)
plt.plot(freq3, fftsen3_abs[:N//2]/N, ":.", color="blue")
plt.title("FFT normalizada Seno k0=N/4+0.5")
plt.ylabel("Densidad espectral de potencia [V]")
plt.xlabel("Frecuencia [Hz]")

Psen1=np.sum(sen1**2)/N
Pfft_sen1=np.sum(fftsen1_abs**2)/N**2
print(Psen1, Pfft_sen1)

Psen2=np.sum(sen2**2)/N
Pfft_sen2=np.sum(fftsen2_abs**2)/N**2
print(Psen2, Pfft_sen2)

Psen3=np.sum(sen3**2)/N
Pfft_sen3=np.sum(fftsen3_abs**2)/N**2
print(Psen3, Pfft_sen3)

#%% Zero padding

ceros=np.zeros(9*N)
sen1_0=np.concatenate((sen1,ceros))
sen2_0=np.concatenate((sen2,ceros))
sen3_0=np.concatenate((sen3,ceros))

freq1, fftsen1_0_abs=fastfouriertrans(sen1_0, 10*N, fs)
freq2, fftsen2_0_abs=fastfouriertrans(sen2_0, 10*N, fs)
freq3, fftsen3_0_abs=fastfouriertrans(sen3_0, 10*N, fs)

ref=np.max(fftsen1_0_abs)
fftsen1_0_abs_dB=20*np.log10(fftsen1_0_abs/ref)
fftsen2_0_abs_dB=20*np.log10(fftsen2_0_abs/ref)
fftsen3_0_abs_dB=20*np.log10(fftsen3_0_abs/ref)

plt.figure(10)
plt.plot(freq1, fftsen1_0_abs_dB[:10*N//2], ":.", color="red")
plt.title("FFT normalizada Seno k0=N/4 (zero padding)")
plt.ylabel("Densidad espectral de potencia [dB]")
plt.xlabel("Frecuencia [Hz]")
plt.figure(11)
plt.plot(freq2, fftsen2_0_abs_dB[:10*N//2],":.", color="green")
plt.title("FFT normalizada Seno k0=N/4+0.25 (zero padding)")
plt.ylabel("Densidad espectral de potencia [dB]")
plt.xlabel("Frecuencia [Hz]")
plt.figure(12)
plt.plot(freq3, fftsen3_0_abs_dB[:10*N//2], ":.", color="blue")
plt.title("FFT normalizada Seno k0=N/4+0.5 (zero padding)")
plt.ylabel("Densidad espectral de potencia [dB]")
plt.xlabel("Frecuencia [Hz]")

plt.figure(13)
plt.plot(freq1, fftsen1_0_abs[:10*N//2]/N, ":.", color="red")
plt.title("FFT normalizada Seno k0=N/4 (zero padding)")
plt.ylabel("Densidad espectral de potencia [V]")
plt.xlabel("Frecuencia [Hz]")

plt.figure(14)
plt.plot(freq2, fftsen2_0_abs[:10*N//2]/N,":.", color="green")
plt.title("FFT normalizada Seno k0=N/4+0.25 (zero padding)")
plt.ylabel("Densidad espectral de potencia [V]")
plt.xlabel("Frecuencia [Hz]")

plt.figure(15)
plt.plot(freq3, fftsen3_0_abs[:10*N//2]/N, ":.", color="blue")
plt.title("FFT normalizada Seno k0=N/4+0.5 (zero padding)")
plt.ylabel("Densidad espectral de potencia [V]")
plt.xlabel("Frecuencia [Hz]")

Psen1=np.sum(sen1_0**2)/(10*N)
Pfft_sen1=np.sum(fftsen1_0_abs**2)/(10*N)**2
print(Psen1, Pfft_sen1)

Psen2=np.sum(sen2_0**2)/(10*N)
Pfft_sen2=np.sum(fftsen2_0_abs**2)/(10*N)**2
print(Psen2, Pfft_sen2)

Psen3=np.sum(sen3_0**2)/(10*N)
Pfft_sen3=np.sum(fftsen3_0_abs**2)/(10*N)**2
print(Psen3, Pfft_sen3)



