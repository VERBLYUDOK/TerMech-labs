import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

# --- Параметры системы ---
R = 2.0        # Радиус колеса (м)
r = 1.0        # Радиус внутренней трубки (м)
c = 20.0       # Жесткость пружины (Н/м)
m = 1.0        # Масса шарика (кг)
g = 9.81       # Ускорение свободного падения (м/с^2)
v = 1.0        # Скорость движения колеса (м/с)
phi_eq = 0.0       # Равновесное положение угла шарика (рад)
phi0 = np.pi / 6   # Начальное отклонение угла (рад)
phi_dot0 = 0.0     # Начальная угловая скорость (рад/с)
psi0 = 0.0         # Начальный угол точки A
ball_radius = 0.2  # Радиус шарика (м)

# --- Время и дискретизация ---
Steps = 1001               # Количество кадров
t_fin = 20                 # Конечное время (с)
t = np.linspace(0, t_fin, Steps)  # Массив времени

dt = t[1] - t[0]           # Шаг по времени

# --- Связь между угловой и линейной скоростью ---
X_O = v * t                # Горизонтальное положение центра колеса (м)
Y_O = R                    # Вертикальное положение центра колеса (м)

# Угловая скорость вращения колеса
omega_wheel = v / R        # Угловая скорость колеса (рад/с)

# --- Угловое положение точек ---
psi = omega_wheel * t + psi0  # Угол точки A относительно вертикали

# --- Решение уравнения движения шарика ---
phi = np.zeros(Steps)      # Угловое положение шарика
phi_dot = np.zeros(Steps)  # Угловая скорость шарика

phi[0] = phi0
phi_dot[0] = phi_dot0
omega_0 = np.sqrt(c / m)   # Собственная угловая частота пружинного маятника

for i in range(1, Steps):
    phi_ddot = -omega_0**2 * (phi[i-1] - phi_eq)  # Ускорение
    phi_dot[i] = phi_dot[i-1] + phi_ddot * dt     # Скорость
    phi[i] = phi[i-1] + phi_dot[i] * dt           # Положение

# --- Координаты точки A ---
X_A = X_O + R * np.sin(psi)
Y_A = Y_O - R * np.cos(psi)

# --- Координаты шарика B (исправлено для точного положения между окружностями) ---
center_radius = R - ball_radius  # Центр шарика учитывает внешний радиус минус его радиус
X_B = X_O + center_radius * np.sin(phi)
Y_B = Y_O - center_radius * np.cos(phi)

# --- Параметры пружины ---
spring_segments = 20       # Количество сегментов пружины

# --- Координаты внутренней трубки ---
inner_tube_radius = R - ball_radius * 2  # Внутренняя трубка корректируется относительно радиуса шарика
X_Tube = inner_tube_radius * np.cos(np.linspace(0, 2 * np.pi, 100))
Y_Tube = inner_tube_radius * np.sin(np.linspace(0, 2 * np.pi, 100))

# Функция для создания координат пружины

def create_spring_segments(x_start, y_start, x_end, y_end, segments):
    X_spring = np.zeros(segments)
    Y_spring = np.zeros(segments)

    for i in range(segments):
        fraction = i / (segments - 1)
        X_spring[i] = x_start + fraction * (x_end - x_start)
        Y_spring[i] = y_start + fraction * (y_end - y_start)

        if 0 < i < segments - 1:
            if i % 2 == 0:
                X_spring[i] += 0.1 * (y_end - y_start)
            else:
                X_spring[i] -= 0.1 * (y_end - y_start)

    return X_spring, Y_spring

# --- Настройка графика ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(X_O.min() - R - 1, X_O.max() + R + 1)
ax.set_ylim(-R - 1, Y_O + R + 1)
ax.set_aspect('equal')
ax.set_xlabel('Горизонтальная позиция (м)')
ax.set_ylabel('Вертикальная позиция (м)')
ax.set_title('Анимация катящегося колеса с шариком и пружиной')

# --- Рисование статичных элементов ---
# Рисуем горизонтальную направляющую
ax.plot([X_O.min() - R - 1, X_O.max() + R + 1], [0, 0], 'k-', linewidth=2)

# --- Рисование динамических элементов ---
# Рисуем колесо
wheel_outline = 100
psi_circle = np.linspace(0, 2 * np.pi, wheel_outline)
X_Wheel = R * np.cos(psi_circle)
Y_Wheel = R * np.sin(psi_circle)
wheel, = ax.plot([], [], 'b-', linewidth=2)

# Рисуем внутреннюю трубку
tube, = ax.plot([], [], 'gray', linestyle='--', linewidth=1)

# Рисуем пружину
spring, = ax.plot([], [], 'r-', linewidth=2)

# Рисуем шарик
ball = plt.Circle((X_B[0], Y_B[0]), ball_radius, color='g')
ax.add_patch(ball)

# Рисуем радиус-векторы
radius_vector_A, = ax.plot([], [], 'g--', linewidth=1)
radius_vector_B, = ax.plot([], [], 'b--', linewidth=1)

# --- Функция инициализации ---
def init():
    wheel.set_data([], [])
    tube.set_data([], [])
    spring.set_data([], [])
    ball.center = (X_B[0], Y_B[0])
    radius_vector_A.set_data([], [])
    radius_vector_B.set_data([], [])
    return wheel, tube, spring, ball, radius_vector_A, radius_vector_B

# --- Функция анимации ---
def anima(i):
    # Обновление положения колеса
    current_X_Wheel = X_Wheel + X_O[i]
    current_Y_Wheel = Y_Wheel + Y_O
    wheel.set_data(current_X_Wheel, current_Y_Wheel)

    # Обновление положения внутренней трубки
    current_X_Tube = X_Tube + X_O[i]
    current_Y_Tube = Y_Tube + Y_O
    tube.set_data(current_X_Tube, current_Y_Tube)

    # Обновление положения пружины
    X_spring, Y_spring = create_spring_segments(X_A[i], Y_A[i], X_B[i], Y_B[i], spring_segments)
    spring.set_data(X_spring, Y_spring)

    # Обновление положения шарика
    ball.center = (X_B[i], Y_B[i])

    # Обновление радиус-векторов
    radius_vector_A.set_data([X_O[i], X_A[i]], [Y_O, Y_A[i]])
    radius_vector_B.set_data([X_O[i], X_B[i]], [Y_O, Y_B[i]])

    return wheel, tube, spring, ball, radius_vector_A, radius_vector_B

# --- Создание анимации ---
anim = FuncAnimation(fig, anima, init_func=init,
                     frames=Steps, interval=40, blit=True)

plt.show()
