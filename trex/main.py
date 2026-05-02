import mss
import numpy as np
import pyautogui
import time
import cv2

class DinoBot:
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = None
        self.last_jump = 0
        
    def find_game(self):
        """Автоматически находит окно с игрой"""
        print("Ищем игру на экране...")
        
        # Пробуем найти по характерному цвету фона игры (#f7f7f7)
        monitor = self.sct.monitors[1]
        screen = np.array(self.sct.grab(monitor))[:, :, :3]
        
        hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
        
        # Маска для белого/светло-серого фона игры
        lower = np.array([0, 0, 200])
        upper = np.array([180, 20, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Ищем самый большой светлый прямоугольник
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            
            # Проверяем что это похоже на игру (достаточно широкое)
            if w > 400 and h > 150:
                self.monitor = {
                    "top": y,
                    "left": x,
                    "width": w,
                    "height": h
                }
                print(f"Игра найдена: x={x}, y={y}, w={w}, h={h}")
                return True
        
        # Если не нашли автоматически — используем весь экран
        self.monitor = {
            "top": 0,
            "left": 0,
            "width": monitor["width"],
            "height": monitor["height"]
        }
        print("Игра не найдена, использую весь экран")
        return False
    
    def capture(self):
        return np.array(self.sct.grab(self.monitor))[:, :, :3]
    
    def find_dino_ground(self, img):
        """Находит землю под динозавром"""
        h, w = img.shape[:2]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Ищем тёмную линию земли в нижней части
        bottom = gray[h//2:, :]
        
        # Находим строку с самой тёмной полосой
        row_means = np.mean(bottom, axis=1)
        ground_y = np.argmin(row_means) + h//2
        
        return ground_y
    
    def process(self, img):
        h, w = img.shape[:2]
        
        # Находим землю
        ground_y = self.find_dino_ground(img)
        
        # Зона перед динозавром (левая треть) на уровне земли
        y1 = max(0, ground_y - 50)
        y2 = min(h, ground_y + 20)
        x1 = w // 5
        x2 = w // 2
        
        roi = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Адаптивная бинаризация
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Опасная зона - ближайшая к динозавру часть
        danger = thresh[:, :thresh.shape[1]//2]
        pct = cv2.countNonZero(danger) / danger.size * 100
        
        return pct
    
    def run(self):
        self.find_game()
        
        print("Запуск через 2 секунды...")
        time.sleep(2)
        pyautogui.press('space')
        
        score = 0
        
        try:
            while True:
                img = self.capture()
                obstacle_pct = self.process(img)
                
                if 3 < obstacle_pct < 30 and time.time() - self.last_jump > 0.15:
                    pyautogui.press('space')
                    self.last_jump = time.time()
                
                score += 1
                
                # Game over detection
                if obstacle_pct > 40:
                    print(f"Score: ~{score}, restarting...")
                    time.sleep(0.5)
                    pyautogui.press('space')
                    score = 0
                
                if score % 500 == 0:
                    print(f"Score: {score}")
                
                time.sleep(0.005)
                
        except KeyboardInterrupt:
            print(f"\nFinal score: ~{score}")

if __name__ == "__main__":
    DinoBot().run()