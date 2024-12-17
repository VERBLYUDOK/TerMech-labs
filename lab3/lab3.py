import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

# --- Параметры системы ---
R = 2.0    # Радиус колеса (м)
c = 200.0   # Жесткость пружины (Н/м)
m = 2.0    # Масса шарика (кг)
M = 5.0    # Масса колеса
g = 9.81   # Ускорение свободного падения (м/с^2)
v = 1.0    # Линейная скорость центра колеса (м/с)
phi0 = 0.2   # Начальное отклонение угла фи (рад)
psi0 = 0.0         # Начальный угол пси (рад)
phi_dot0 = 0.0      # Начальная угловая скорость фи (рад/с)
psi_dot0 = 0.0      # Начальная угловая скорость пси (рад/с)
ball_radius = 0.2   # Радиус шарика (м)

# Параметры времени
Steps = 1001
t_fin = 20
t = np.linspace(0, t_fin, Steps)
dt = t[1]-t[0]


def odesys(y, t, M, m, c, g, R):
    # y = [phi, psi, phi_dot, psi_dot]
    phi = y[0]
    psi = y[1]
    phi_dot = y[2]
    psi_dot = y[3]

    alpha = (phi + psi)/2.0

    # Матрица системы:
    # A * [ddphi; ddpsi] = B
    A11 = 1.0
    A12 = np.cos(phi)
    A21 = np.cos(phi)
    A22 = 1.0 + 2.0*(M/m)
    det = A11*A22 - A12*A21

    # Правая часть
    # B1 и B2 из уравнений Лагранжа:
    # phi'' + psi'' cos phi = -[2c/m(1−cos alpha) sin alpha + (g/R) sin phi]
    # [1+2(M/m)] psi'' + phi'' cos phi - phi_dot² sin phi = -2c/m(1−cos alpha) sin alpha

    B1 = - (2*c/m)*(1 - np.cos(alpha))*np.sin(alpha) - (g/R)*np.sin(phi)
    B2 = - (2*c/m)*(1 - np.cos(alpha))*np.sin(alpha) + (phi_dot**2)*np.sin(phi)

    ddphi = (B1*A22 - B2*A12) / det
    ddpsi = (A11*B2 - A21*B1) / det

    return np.array([phi_dot, psi_dot, ddphi, ddpsi])


# Начальные условия
y0 = [phi0, psi0, phi_dot0, psi_dot0]

# Численное интегрирование
Y = odeint(odesys, y0, t, args=(M, m, c, g, R))

phi = Y[:, 0]
psi = Y[:, 1]
phi_dot = Y[:, 2]
psi_dot = Y[:, 3]

# Чтобы найти ddphi и ddpsi на каждом шаге, снова вызовем odesys
ddphi = np.zeros_like(phi)
ddpsi = np.zeros_like(phi)
for i in range(len(t)):
    dydt = odesys(Y[i,:], t[i], M, m, c, g, R)
    ddphi[i] = dydt[2]
    ddpsi[i] = dydt[3]

# Теперь вычисляем силу N
# N = m[gcos phi + R( phi'^2- psi''sin phi )] + 2 R c (1−cos alpha) cos alpha
alpha = (phi + psi)/2.0
N = (m*(g*np.cos(phi) + R*(phi_dot**2 - ddpsi*np.sin(phi))) + 
     2*R*c*(1 - np.cos(alpha))*np.cos(alpha))

# Координаты центра колеса
X_O = v * t
Y_O = R

# Координаты точки A
X_A = X_O + R * np.sin(psi)
Y_A = Y_O - R * np.cos(psi)

# Положение шарика
center_radius = R - ball_radius
X_B = X_O + center_radius * np.sin(phi)
Y_B = Y_O - center_radius * np.cos(phi)

# Параметры пружины
spring_segments = 20

# Координаты внутренней трубки
inner_tube_radius = R - ball_radius*2
X_Tube = inner_tube_radius * np.cos(np.linspace(0, 2*np.pi, 100))
Y_Tube = inner_tube_radius * np.sin(np.linspace(0, 2*np.pi, 100))


def create_spring_segments(x_start, y_start, x_end, y_end, segments):
    X_spring = np.zeros(segments)
    Y_spring = np.zeros(segments)
    for i in range(segments):
        fraction = i/(segments - 1)
        X_spring[i] = x_start + fraction*(x_end - x_start)
        Y_spring[i] = y_start + fraction*(y_end - y_start)

        if 0 < i < segments - 1:
            if i % 2 == 0:
                X_spring[i] += 0.1*(y_end - y_start)
            else:
                X_spring[i] -= 0.1*(y_end - y_start)
    return X_spring, Y_spring


# --- Анимация ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(X_O.min() - R - 1, X_O.max() + R + 1)
ax.set_ylim(-R - 1, Y_O + R + 1)
ax.set_aspect('equal')
ax.set_xlabel('Горизонтальная позиция (м)')
ax.set_ylabel('Вертикальная позиция (м)')
ax.set_title('Анимация катящегося колеса с шариком и пружиной')

# Горизонтальная направляющая
ax.plot([X_O.min() - R - 1, X_O.max() + R + 1], [0, 0], 'k-', linewidth=2)

wheel_outline = 100
psi_circle = np.linspace(0, 2*np.pi, wheel_outline)
X_Wheel = R*np.cos(psi_circle)
Y_Wheel = R*np.sin(psi_circle)
wheel, = ax.plot([], [], 'b-', linewidth=2)
tube, = ax.plot([], [], 'gray', linestyle='--', linewidth=1)
spring, = ax.plot([], [], 'r-', linewidth=2)

ball = plt.Circle((X_B[0], Y_B[0]), ball_radius, color='g')
ax.add_patch(ball)

radius_vector_A, = ax.plot([], [], 'g--', linewidth=1)
radius_vector_B, = ax.plot([], [], 'b--', linewidth=1)


def init():
    wheel.set_data([], [])
    tube.set_data([], [])
    spring.set_data([], [])
    ball.center = (X_B[0], Y_B[0])
    radius_vector_A.set_data([], [])
    radius_vector_B.set_data([], [])
    return wheel, tube, spring, ball, radius_vector_A, radius_vector_B


def anima(i):
    current_X_Wheel = X_Wheel + X_O[i]
    current_Y_Wheel = Y_Wheel + Y_O
    wheel.set_data(current_X_Wheel, current_Y_Wheel)

    current_X_Tube = X_Tube + X_O[i]
    current_Y_Tube = Y_Tube + Y_O
    tube.set_data(current_X_Tube, current_Y_Tube)

    X_spring_, Y_spring_ = create_spring_segments(X_A[i], Y_A[i], X_B[i], Y_B[i], spring_segments)
    spring.set_data(X_spring_, Y_spring_)

    ball.center = (X_B[i], Y_B[i])

    radius_vector_A.set_data([X_O[i], X_A[i]], [Y_O, Y_A[i]])
    radius_vector_B.set_data([X_O[i], X_B[i]], [Y_O, Y_B[i]])

    return wheel, tube, spring, ball, radius_vector_A, radius_vector_B


anim = FuncAnimation(fig, anima, init_func=init,
                     frames=Steps, interval=40, blit=True)

plt.show()

# Построение трех графиков: x(t), phi(t), N(t)
fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))

# x(t) = X_O(t)
ax1.plot(t, X_O, label='x(t)')
ax1.set_ylabel('x (м)')
ax1.set_title('x(t)')
ax1.grid(True)
ax1.legend()

# phi(t)
ax2.plot(t, phi, label='phi(t)', color='r')
ax2.set_ylabel('phi (рад)')
ax2.set_title('phi(t)')
ax2.grid(True)
ax2.legend()

# N(t)
ax3.plot(t, N, label='N(t)', color='g')
ax3.set_ylabel('N (H)')
ax3.set_title('N(t)')
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.show()
