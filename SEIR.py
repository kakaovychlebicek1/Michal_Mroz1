import numpy as np
import matplotlib.pyplot as plt

BETA, SIGMA, GAMMA, POPULACJA = 1, 1, 0.1, 1

I0, E0, S0, R0 = 0, 0.01, 0.99, 0

Ro = (BETA * S0) / GAMMA
print(Ro)

DT = 0.01
T0, T13 = 0, 40

CZAS = np.arange(T0, T13, DT)

N = len(CZAS)

Y = np.zeros([N, 4])

Y[0, 0] = S0
Y[0, 1] = E0
Y[0, 2] = I0
Y[0, 3] = R0

def SEIR(STAN, T):
    S = STAN[0]
    E = STAN[1]
    I = STAN[2]
    
    DS = (-BETA * S * I)
    DE = (BETA * S * I) - SIGMA * E
    DI = SIGMA * E - GAMMA * I
    DR = GAMMA * I

    return np.array([DS, DE, DI, DR])

def Runge_Kutta(Y, T, DT, POCHODNA):
    K1 = DT * POCHODNA(Y, T)
    K2 = DT * POCHODNA(Y + K1 / 2, T + 0.5 * DT)
    K3 = DT * POCHODNA(Y + K2 / 2, T + 0.5 * DT)
    K4 = DT * POCHODNA(Y + K3, T + DT)
    Y_STEP = Y + 1 / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
    return Y_STEP

for i in range(N - 1):
    Y[i + 1] = Runge_Kutta(Y[i], CZAS[i], DT, SEIR)
    

plt.figure(figsize=(12, 8))
plt.plot(CZAS, Y[:, 0], label='Podatni (S)')
plt.plot(CZAS, Y[:, 1], label='Wystawieni (E)')
plt.plot(CZAS, Y[:, 2], label='Chorzy (I)')
plt.plot(CZAS, Y[:, 3], label='Ozdrowieńcy (R)')
plt.xlabel('Czas')
plt.ylabel('Liczba populacji')
plt.title('Model SEIR')
plt.legend()
plt.grid(True)
plt.show()