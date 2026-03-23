import cv2              
import numpy as np      
import math             
import tkinter as tk    
from tkinter import ttk 
import time              
import csv              
import pandas as pd      
import matplotlib.pyplot as plt  

# HSV aralıkları
color_ranges = {
    "red": [
        (np.array([0, 120, 70]),  np.array([10, 255, 255])),   
        (np.array([170, 120, 70]), np.array([180, 255, 255]))  
    ],
    "green": [(np.array([40, 40, 40]),  np.array([85, 255, 255]))],
    "blue":  [(np.array([100, 100, 50]), np.array([130, 255, 255]))],
    "yellow":[(np.array([20, 100, 100]), np.array([30, 255, 255]))],
    "orange":[(np.array([10, 100, 100]), np.array([20, 255, 255]))],
    "purple":[(np.array([130, 100, 100]), np.array([160, 255, 255]))],
    "cyan":  [(np.array([85, 100, 100]), np.array([95, 255, 255]))],
    "pink":  [(np.array([160, 100, 100]), np.array([170, 255, 255]))],
    "brown": [(np.array([10, 50, 50]), np.array([20, 200, 200]))],
    "white": [(np.array([0, 0, 200]), np.array([180, 30, 255]))],
    "black": [(np.array([0, 0, 0]), np.array([180, 255, 50]))]
}

# Çizim renkleri
draw_colors = {
    "red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0),
    "yellow": (0, 255, 255), "orange": (0, 165, 255), "purple": (255, 0, 255),
    "cyan": (255, 255, 0), "pink": (255, 105, 180), "brown": (42, 42, 165),
    "white": (255, 255, 255), "black": (0, 0, 0)
}

# Parametreler
min_area = 1000          
min_circ = 0.30          
blur_ksize = 5          
morph_iter = 2           
match_thresh = 50        
max_age = 30            
empty_reset_age = 15    

# Yardımcı Fonksiyonlar
def circularity(cnt):
    area = cv2.contourArea(cnt)
    per = cv2.arcLength(cnt, True)
    if per == 0:
        return 0.0
    return 4 * math.pi * area / (per * per)

def make_mask(hsv, ranges):
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blurred = cv2.GaussianBlur(hsv, (k, k), 0)
    mask = None
    for low, high in ranges:
        m = cv2.inRange(blurred, low, high)
        mask = m if mask is None else (mask | m)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None, iterations=morph_iter)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, None, iterations=morph_iter)
    return mask

# Kararlı ID Takibi
class StableTracker:
    def __init__(self, color_name, match_thresh=50, max_age=30, empty_reset_age=15):
        self.color_name = color_name
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.empty_reset_age = empty_reset_age
        self.reset_all()

    def reset_all(self):
        self.next_id = 1
        self.objects = {}
        self.empty_frames = 0

    def update(self, centers):
        if len(centers) == 0:
            self.empty_frames += 1
        else:
            self.empty_frames = 0
        if self.empty_frames >= self.empty_reset_age:
            self.reset_all()

        for oid in list(self.objects.keys()):
            self.objects[oid]["age"] += 1
            if self.objects[oid]["age"] > self.max_age:
                del self.objects[oid]

        assigned = set()
        for oid, info in self.objects.items():
            ocx, ocy = info["center"]
            best_idx, best_dist = None, 1e9
            for i, c in enumerate(centers):
                if i in assigned:
                    continue
                dist = (c[0]-ocx)**2 + (c[1]-ocy)**2
                if dist < best_dist:
                    best_dist, best_idx = dist, i
            if best_idx is not None and best_dist <= self.match_thresh**2:
                self.objects[oid]["center"] = centers[best_idx]
                self.objects[oid]["age"] = 0
                assigned.add(best_idx)

        for i, c in enumerate(centers):
            if i in assigned:
                continue
            oid = self.next_id
            self.next_id += 1
            self.objects[oid] = {"center": c, "age": 0}

        out = []
        for oid, info in self.objects.items():
            label = f"{self.color_name}{oid}"
            out.append((oid, info["center"], label))
        return out

trackers = {name: StableTracker(name, match_thresh, max_age, empty_reset_age)
            for name in color_ranges.keys()}

# Tkinter GUI
root = tk.Tk()
root.title("Renge Dayalı Nesne Takip Uygulaması")

running = False
cap = None

# Log kaydı kontrolü
log_enabled = tk.BooleanVar(value=False)
ttk.Checkbutton(root, text="CSV Log Kaydı", variable=log_enabled).pack(pady=4)

# Çoklu renk seçimi (checkboxlar)
selected_colors = {}
ttk.Label(root, text="Renk seçimi (çoklu)").pack(pady=4)
for cname in color_ranges.keys():
    var = tk.BooleanVar(value=False)
    selected_colors[cname] = var
    ttk.Checkbutton(root, text=cname.capitalize(), variable=var).pack(anchor="w")

# Sliderlar
ttk.Label(root, text="Min Area").pack(pady=2)
min_area_var = tk.IntVar(value=min_area)
tk.Scale(root, from_=500, to=30000, orient="horizontal", variable=min_area_var).pack(fill="x")

ttk.Label(root, text="Min Circularity x100").pack(pady=2)
min_circ_var = tk.IntVar(value=int(min_circ*100))
tk.Scale(root, from_=10, to=100, orient="horizontal", variable=min_circ_var).pack(fill="x")

ttk.Label(root, text="Blur ksize").pack(pady=2)
blur_var = tk.IntVar(value=blur_ksize)
tk.Scale(root, from_=3, to=11, orient="horizontal", variable=blur_var).pack(fill="x")

ttk.Label(root, text="Morph iter").pack(pady=2)
morph_var = tk.IntVar(value=morph_iter)
tk.Scale(root, from_=1, to=5, orient="horizontal", variable=morph_var).pack(fill="x")

# Önceki merkezleri saklamak için dictionary (hız hesabı)
prev_centers = {}

# Takip Fonksiyonu
def start_tracking():
    global running, cap, min_area, min_circ, blur_ksize, morph_iter, prev_centers
    if running: 
        return
    running = True
    cap = cv2.VideoCapture(0)

    prev_time = time.time()

    if log_enabled.get():
        log_file = open("log.csv", "w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(["time", "color", "id", "area", "speed"])
    else:
        log_file = None
        writer = None

    out = None
    video_recording = False

    while running:
        ret, frame = cap.read()
        if not ret: 
            break

        frame = cv2.flip(frame, 1)

        # Parametreleri GUI’den oku
        min_area = int(min_area_var.get())
        min_circ = float(min_circ_var.get())/100.0
        blur_ksize = int(blur_var.get())
        morph_iter = int(morph_var.get())  

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        active_colors = [c for c, var in selected_colors.items() if var.get()]
        if not active_colors:
            active_colors = ["red"]

        for cname in active_colors:
            ranges = color_ranges[cname]
            mask = make_mask(hsv, ranges)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            centers = []
            for c in contours:
                area = cv2.contourArea(c)
                if area < min_area: 
                    continue
                if circularity(c) < min_circ: 
                    continue
                M = cv2.moments(c)
                if M["m00"] == 0: 
                    continue
                cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
                centers.append((cx, cy))

            tracker = trackers[cname]
            id_list = tracker.update(centers)
            draw_color = draw_colors[cname]

            for c in contours:
                if cv2.contourArea(c) >= min_area and circularity(c) >= min_circ:
                    cv2.drawContours(frame, [c], -1, draw_color, 2)

            for oid, center, label in id_list:
                cv2.circle(frame, center, 8, draw_color, -1)
                nearest = None; best = 1e9
                for c in contours:
                    M = cv2.moments(c)
                    if M["m00"] == 0: 
                        continue
                    cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
                    d = (cx-center[0])**2 + (cy-center[1])**2
                    if d < best: 
                        best, nearest = d, c
                if nearest is not None:
                    a = cv2.contourArea(nearest)

                    # Hız Hesabı
                    if oid in prev_centers:
                        px, py, pt = prev_centers[oid]
                        dx = center[0] - px
                        dy = center[1] - py
                        dt = time.time() - pt
                        if dt > 0:
                            speed = math.sqrt(dx*dx + dy*dy) / dt
                        else:
                            speed = 0
                    else:
                        speed = 0
                    prev_centers[oid] = (center[0], center[1], time.time())

                    cv2.putText(frame, f"{label} A={int(a)} v={speed:.1f}px/s",
                                (center[0]+10, center[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)

                    if writer is not None:
                        writer.writerow([time.time(), cname, label, int(a), speed])

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.putText(frame, "ESC: cikis | S: snapshot | V: video toggle",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        cv2.imshow("Takip Ekrani", frame)

        # Eğer video kaydı açıksa kareyi dosyaya yaz
        if video_recording and out is not None:
            out.write(frame)

        k = cv2.waitKey(1) & 0xFF

        if k == 27:  # ESC
            close_app()
        elif k == ord('s'):  # Ekran Fotoğrafı
            filename = f"snapshot_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            print(f"Snapshot kaydedildi: {filename}")
        elif k == ord('v'):  # Video kaydı
            if not video_recording:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                h, w = frame.shape[:2]
                out = cv2.VideoWriter("output.avi", fourcc, 20.0, (w, h))
                video_recording = True
                print("Video kaydi basladi.")
            else:
                video_recording = False
                if out is not None:
                    out.release()
                    out = None
                print("Video kaydi durduruldu.")

    if log_file is not None:
        log_file.close()
    if out is not None:
        out.release()

# Grafik Göster Fonksiyonu (alan ve hız)
def show_graphs():
    try:
        df = pd.read_csv("log.csv")   
    except FileNotFoundError:
        print("Log dosyası bulunamadı!")  
        return

    # Zamanı normalize etme
    df["time"] = df["time"] - df["time"].min()

    # Alan grafiği
    fig1 = plt.figure(figsize=(10,6))
    for color in df["color"].unique():   
        subset = df[df["color"] == color]
        plt.plot(subset["time"], subset["area"], label=f"{color} area")
    plt.xlabel("Zaman (s)")
    plt.ylabel("Alan (px^2)")
    plt.title("Nesne Alanı Zamanla")
    plt.legend()
    fig1.savefig("alan_grafik.png")      

    # Hız grafiği
    fig2 = plt.figure(figsize=(10,6))
    for color in df["color"].unique():
        subset = df[df["color"] == color]
        plt.plot(subset["time"], subset["speed"], label=f"{color} speed")
    plt.xlabel("Zaman (s)")
    plt.ylabel("Hız (px/s)")
    plt.title("Nesne Hızı Zamanla")
    plt.legend()
    fig2.savefig("hiz_grafik.png")

    plt.show()   
    print("Grafikler alan_grafik.png ve hiz_grafik.png olarak kaydedildi.")

# Close Fonksiyonu
def close_app():
    cv2.destroyAllWindows()  
    if cap is not None:
        cap.release()         
    root.destroy()           

# Tkinter Butonları
ttk.Button(root, text="Başlat", command=start_tracking).pack(pady=6)   
ttk.Button(root, text="Kapat", command=close_app).pack(pady=6)         
ttk.Button(root, text="Grafik Göster", command=show_graphs).pack(pady=6) 

# Pencereyi kapatmak için protokol
root.protocol("WM_DELETE_WINDOW", close_app)

# Tkinter ana döngüsü, GUI sürekli çalışır.
root.mainloop()