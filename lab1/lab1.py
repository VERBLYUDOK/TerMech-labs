import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def Rot2D(X, Y, Alpha):
    RX = X*np.cos(Alpha) - Y*np.sin(Alpha)
    RY = X*np.sin(Alpha) + Y*np.cos(Alpha)
    return RX, RY


t = sp.Symbol('t')

# Закон движения согласно варианту
r = 1 + sp.sin(t)
phi = t

# Переход к декартовым координатам
x = r * sp.cos(phi)
y = r * sp.sin(phi)

# Вычисление скорости
Vx = sp.diff(x, t)
Vy = sp.diff(y, t)

# Вычисление ускорения
Ax = sp.diff(Vx, t)
Ay = sp.diff(Vy, t)

T = np.linspace(0, 10 * np.pi, 1000)

# Создание функций для вычислений
F_x = sp.lambdify(t, x, 'numpy')
F_y = sp.lambdify(t, y, 'numpy')
F_Vx = sp.lambdify(t, Vx, 'numpy')
F_Vy = sp.lambdify(t, Vy, 'numpy')
F_Ax = sp.lambdify(t, Ax, 'numpy')
F_Ay = sp.lambdify(t, Ay, 'numpy')

# Вычисление значений
X = F_x(T)
Y = F_y(T)
VX = F_Vx(T)
VY = F_Vy(T)
AX = F_Ax(T)
AY = F_Ay(T)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')
ax1.set(xlim=[-3, 3], ylim=[-2, 4])

# Построение траектории
ax1.plot(X, Y, label='Траектория')

# Точка на траектории (инициализируем без данных)
P, = ax1.plot([], [], 'ko', label='Точка')

# Базовые координаты для стрелочек
ArrowX = np.array([-0.2, 0, -0.2])
ArrowY = np.array([0.1, 0, -0.1])

# Линия и стрелка для вектора скорости (инициализируем без данных)
VLine, = ax1.plot([], [], 'r', label='Вектор скорости')
VArrow, = ax1.plot([], [], 'r')

# Линия и стрелка для вектора ускорения (инициализируем без данных)
ALine, = ax1.plot([], [], 'g', label='Вектор ускорения')
AArrow, = ax1.plot([], [], 'g')

# Линия и стрелка для радиус-вектора (инициализируем без данных)
RLine, = ax1.plot([], [], 'b', label='Радиус-вектор')
RArrow, = ax1.plot([], [], 'b')

# Добавление легенды
ax1.legend(loc='upper left')

# Функция инициализации
def init():
    P.set_data([], [])
    VLine.set_data([], [])
    VArrow.set_data([], [])
    ALine.set_data([], [])
    AArrow.set_data([], [])
    RLine.set_data([], [])
    RArrow.set_data([], [])
    return P, VLine, VArrow, ALine, AArrow, RLine, RArrow

# Функция анимации
def anima(i):
    P.set_data([X[i]], [Y[i]])

    # Обновление вектора скорости
    VLine.set_data([X[i], X[i] + VX[i]], [Y[i], Y[i] + VY[i]])
    VArrowX, VArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[i], VX[i]))
    VArrow.set_data(VArrowX+X[i] + VX[i], VArrowY + Y[i] + VY[i])

    # Обновление вектора ускорения
    ALine.set_data([X[i], X[i] + AX[i]], [Y[i], Y[i] + AY[i]])
    AArrowX, AArrowY = Rot2D(ArrowX, ArrowY, math.atan2(AY[i], AX[i]))
    AArrow.set_data(AArrowX + X[i] + AX[i], AArrowY + Y[i] + AY[i])

    # Обновление радиус-вектора
    RLine.set_data([0, X[i]], [0, Y[i]])
    RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(Y[i], X[i]))
    RArrow.set_data(RArrowX + X[i], RArrowY + Y[i])

    return P, VLine, VArrow, ALine, AArrow, RLine, RArrow

# Создание анимации с функцией инициализации
anim = FuncAnimation(fig, anima, frames=len(T), init_func=init, interval=20, blit=True)

plt.show()
