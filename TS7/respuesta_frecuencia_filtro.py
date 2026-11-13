"""
Respuesta en frecuencia de un filtro digital a partir de h[n].
Ejecutar en Spyder. Requiere scipy, numpy y matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.patches import Rectangle


wp = 4/100
ws = 3/ 100
G_p = 1
G_s = 80

x = np.arange(0,np.pi,0.001)
"""
b, a = signal.iirdesign(wp, ws, G_p, G_s, ftype='butter')

print(len(b))
w, H = signal.freqz(b, a, worN=2048)
modulo = np.abs(H)                 # módulo en veces (escala lineal)
fase = np.unwrap(np.angle(H))      # fase en radianes

fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax[0].plot(w, modulo)
ax[0].set_ylabel('Módulo (veces)')
ax[0].set_title('Respuesta en frecuencia')
ax[0].grid(True)

# Bordes de paso y rechazo. freqz expresa w en radianes/muestra;
# iirdesign recibe wp y ws normalizados respecto de Nyquist.
wp_rad = wp * np.pi
ws_rad = ws * np.pi
for axes in ax:
    axes.axvline(wp_rad, color='orange', linestyle='--', label='wp')
    axes.axvline(ws_rad, color='red', linestyle='--', label='ws')

# Límites de módulo correspondientes a Gp y Gs expresados en dB.
ax[0].axhline(10 ** (-G_p / 20), color='orange', linestyle=':', label='Límite Gp')
ax[0].axhline(10 ** (-G_s / 20), color='red', linestyle=':', label='Límite Gs')
ax[0].legend()

ax[1].plot(w, fase)

ax[1].set_ylabel('Fase (rad)')
ax[1].set_xlabel('Frecuencia angular omega (rad/muestra)')
ax[1].grid(True)
ax[1].legend()

plt.tight_layout()
plt.show()

print('b (numerador) =', b)
print('a (denominador) =', a)
"""
# %%
ws = 60
wp = 40
G_p = 3
G_s = 60
fs = 1000

sos = signal.iirdesign(wp, ws, G_p, G_s, ftype='butter',output='sos',fs=fs)

W, H = signal.freqz_sos(sos,fs=fs)

modulo = np.abs(H)                 
fase = np.unwrap(np.angle(H))
retardo = -np.gradient(H, W*2*np.pi)   


fig, ax = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
ax[0].plot(W, 20*np.log10(modulo))
ax[0].set_ylabel('Módulo (veces)')
ax[0].set_title('Respuesta en frecuencia')
ax[0].grid(True)

# Bordes de paso y rechazo. freqz expresa w en radianes/muestra;
# iirdesign recibe wp y ws normalizados respecto de Nyquist.
wp_rad = wp
ws_rad = ws
for axes in ax:
    axes.axvline(wp_rad, color='orange', linestyle='--', label='wp')
    axes.axvline(ws_rad, color='red', linestyle='--', label='ws')

# Límites de módulo correspondientes a Gp y Gs expresados en dB.
ax[0].axhline( -G_p , color='orange', linestyle=':', label='Límite Gp')
ax[0].axhline( -G_s , color='red', linestyle=':', label='Límite Gs')
ax[0].legend()

ax_mod = ax[0]

#Límites verticales del gráfico en dB
y_min, y_max = ax_mod.get_ylim()

#Banda pasante: zona no permitida por debajo de -G_p
rect_pasante = Rectangle((0, y_min),wp,-G_p - y_min,facecolor="red",alpha=0.12, edgecolor="red", hatch="xxx", linewidth=1.2,label="Plantilla")
ax_mod.add_patch(rect_pasante)

#Banda de rechazo: zona no permitida por encima de -G_s
rect_rechazo = Rectangle((ws, -G_s),fs / 2 - ws, y_max + G_s, facecolor="red",alpha=0.12, edgecolor="red",hatch="xxx",linewidth=1.2)
ax_mod.add_patch(rect_rechazo)

ax_mod.legend()

ax[1].plot(W, fase)

ax[1].set_ylabel('Fase (rad)')
ax[1].set_xlabel('Frecuencia angular omega (rad/muestra)')
ax[1].grid(True)
ax[1].legend()

ax[2].plot(W, retardo)
ax[2].set_ylabel('Retardo (ms)')
ax[2].set_xlabel('Frecuencia ')
ax[2].grid(True)
ax[2].legend()

"""
y = signal.sosfilt(sos, x, zi=None)


fig2, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax[0].plot(x, y)
ax[0].set_ylabel('Módulo (veces)')
ax[0].set_title('Respuesta en frecuencia')
ax[0].grid(True)

# Bordes de paso y rechazo. freqz expresa w en radianes/muestra;
# iirdesign recibe wp y ws normalizados respecto de Nyquist.
wp_rad = wp * np.pi
ws_rad = ws * np.pi
for axes in ax:
    axes.axvline(wp_rad, color='orange', linestyle='--', label='wp')
    axes.axvline(ws_rad, color='red', linestyle='--', label='ws')

# Límites de módulo correspondientes a Gp y Gs expresados en dB.
ax[0].axhline(10 ** (-G_p / 20), color='orange', linestyle=':', label='Límite Gp')
ax[0].axhline(10 ** (-G_s / 20), color='red', linestyle=':', label='Límite Gs')
ax[0].legend()

plt.tight_layout()
plt.show()
"""
print(sos)
import scipy.io as sio
fs_ecg = 1000

sio.whosmat('ECG_TP4.mat')
mat_struct=sio.loadmat('./ECG_TP4.mat')

ecg_one_lead_audio =mat_struct ['ecg_lead'] * 1.0
ecg_one_lead_audio = ecg_one_lead_audio.flatten()
y = signal.sosfilt(sos,ecg_one_lead_audio)

plt.figure(2)
plt.title("Señales")
plt.plot(y[5000:12000],label='Señal filtrada por SOS')
plt.plot(ecg_one_lead_audio[5000:12000],label='Señal de ECG')
plt.grid()
plt.legend()
plt.show()

x=np.zeros(200)
x[0] = 1

t= signal.sosfilt(sos,x)

plt.figure(3)
plt.title("Señales")
plt.plot(t,label='Señal filtrada por SOS')
plt.plot(x,label='Señal de ECG')
plt.grid()
plt.legend()
plt.show()
