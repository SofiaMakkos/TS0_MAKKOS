# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 20:21:49 2026

@author: Zsofi
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
#Definiciones
fs=1000 #Hz
N=1000 #muestras

#%%
#Funciones

def mi_funcion_sen( vmax = 1, dc = 0, ff = 1, ph=0, nn = N, fs = fs):
    tt=np.arange(0.0, N/fs, 1/fs)
    xx=vmax*np.sin(2*np.pi*ff*tt +ph)
    return(tt, xx)

#%%

vmax=2
dc=0
ff=5
ph=0


tt,xx=mi_funcion_sen(ff=500)
plt.plot(tt, xx, color="blue", linestyle=" ",marker=".",  label="Funcion seno")
plt.legend(loc="upper right")
plt.xlabel("Tiempo (t)")
plt.show()

tt,xx=mi_funcion_sen(ff=999)
plt.plot(tt, xx, color="green", linestyle=" ",marker=".", label="Funcion seno")
plt.legend(loc="upper right")
plt.xlabel("Tiempo (t)")
plt.show()

tt,xx=mi_funcion_sen(ff=1001)
plt.plot(tt, xx, color="purple", linestyle=" ",marker=".", label="Funcion seno")
plt.legend(loc="upper right")
plt.xlabel("Tiempo (t)")
plt.show()

tt,xx=mi_funcion_sen(ff=2001)
plt.plot(tt, xx, color="cyan", linestyle=" ",marker=".", label="Funcion seno")
plt.legend(loc="upper right")
plt.xlabel("Tiempo (t)")
plt.show()

#tt, xx = mi_funcion_sen( vmax = 1, dc = 0, ff = 1, ph=0, nn = N, fs = fs)


#%%