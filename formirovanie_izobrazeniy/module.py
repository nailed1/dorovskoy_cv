import numpy as np
import matplotlib.pyplot as plt

class ImagingModel():
    def __init__(self, shape=(256,256)):
        self.h, self.w = shape
        self.y, self.x = np.meshgrid(np.arange(self.h), np.arange(self.w))
    
    def create_reflection(self, background=0.85, foreground=0.03):
        r = np.zeros((self.h, self.w)) + background
        mask = (self.x - self.w/2)**2 + (self.y - self.h/2)**2 < 100**2
        r[mask] = foreground
        return r
    
    def point_light(self, cx, cy):
        d = np.sqrt((self.x - cx)**2 + (self.y - cy)**2)
        return np.exp(-d/50)
    
    def ambient_light(self, constant=0.1):
        return np.ones((self.h, self.w)) * constant
    
    def directional_light(self, angle=np.pi/2, intensity = 0.2):
        x_norm = self.x/self.w
        y_norm = self.y/self.h
        dx = np.cos(angle)
        dy = np.sin(angle)
        projection = dx*x_norm + dy*y_norm
        return intensity * np.clip(projection, 0, 1)
    


model = ImagingModel((512,512))
r = model.create_reflection()           


plt.figure(figsize=(15,7))
plt.ion()

p1 = (model.h//2, model.w//2)
p2 = [10,10]
p3 = [model.h-10, model.w-10]

intense_p1 = []
intense_p2 = []
intense_p3 = []

iters = 4
step = 5
for angle, rad in zip(range(0,360 * iters, step), np.linspace(50, 250, int(360*iters/step))):
    x = model.w//2+rad*np.cos(np.deg2rad(angle))
    y = model.h//2+rad*np.sin(np.deg2rad(angle))
    i = model.point_light(x, y) + model.ambient_light(0.4) + model.directional_light(np.pi/4, 0.3)
    f = r*i

    intense_p1.append(f[*p1])
    intense_p2.append(f[*p2])
    intense_p3.append(f[*p3])

    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(f)
    plt.clim(0, 1)
    plt.scatter(p1[1], p1[0])
    plt.scatter(p2[1], p2[0])
    plt.scatter(p3[1], p3[0])
    plt.subplot(1,2,2)
    plt.plot(intense_p1, label="p1")
    plt.plot(intense_p2, label="p2")
    plt.plot(intense_p3, label="p3")
    plt.legend()
    plt.pause(0.05)

    if not plt.get_fignums():
        break
plt.ioff()
plt.show()