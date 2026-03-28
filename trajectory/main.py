import matplotlib.pyplot as plt
import numpy as np
import skimage.measure as measure

# Ваши функции из первого кода
def area(labeled, label):
    return (labeled == label).sum()

def centroid(labeled, label):
    coords = np.argwhere(labeled == label)
    if coords.size > 0:
        return coords.mean(axis=0)
    else:
        return None

def distance_to_last(last, target_point):
    dy = last[0] - target_point[0]
    dx = last[1] - target_point[1]
    dx_squared = dx ** 2
    dy_squared = dy ** 2
    sum_of_squares = dy_squared + dx_squared
    distance = sum_of_squares ** 0.5
    return distance

# Словарь для хранения траекторий (как во втором коде)
trajectories = {}

# Цикл по всем 100 кадрам
for i in range(100):
    image = np.load(f"trajectory/out/h_{i}.npy")
    labeled = measure.label(image, background=0)
    max_label = labeled.max()
    
    # Получаем центроиды всех объектов в текущем кадре
    centroids = []
    for j in range(1, max_label + 1):
        center = centroid(labeled, j)
        if center is not None:
            centroids.append(center)
    
    if i == 0:
        # Инициализация траекторий в первом кадре
        for j, center in enumerate(centroids):
            trajectories[j] = [center]
    else:
        # Сопоставление объектов с существующими траекториями
        for cur_cord in centroids:
            min_num = 0
            min_dist = float('inf')
            
            for num, cord in trajectories.items():
                last_cord = cord[-1]
                # Используем вашу функцию расстояния
                dist = distance_to_last(last_cord, cur_cord)
                
                if min_dist > dist:
                    min_dist = dist
                    min_num = num
            
            trajectories[min_num].append(cur_cord)

# Отрисовка всех траекторий
for num, obj in trajectories.items():
    cord = np.array(obj)
    plt.plot(cord[:, 1], cord[:, 0], marker='o', label=f'Track {num}')

plt.legend()
plt.show()