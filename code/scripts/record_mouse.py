import tkinter as tk
from collections import deque
import time


class MouseTracker:
    def __init__(self, root, update_frequency=100, max_size=100):
        self.update_position = False
        self.root = root
        self.update_frequency = update_frequency
        self.frame = tk.Frame(root, bg='yellow', width=500, height=500)
        self.frame.bind("<Motion>", self.track_mouse)
        self.frame.bind("<Button-1>", self.toggle_update)
        self.frame.pack()
        self.last_event = None
        self.positions = deque(maxlen=max_size)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def track_mouse(self, event):
        self.last_event = event

    def showxy(self):
        if self.update_position and self.last_event:
            xm, ym = self.last_event.x, self.last_event.y
            timestamp = time.time()
            self.positions.append((xm, ym, timestamp))
            str1 = f"mouse at x={xm}  y={ym}"
            self.root.title(str1)
            self.frame.config(bg='white')
        self.root.after(self.update_frequency, self.showxy)

    def toggle_update(self, event):
        self.update_position = not self.update_position
        if self.update_position:
            self.showxy()

    def on_closing(self):
        # Print the positions and timestamps
        for position in self.positions:
            print(position)
        self.root.destroy()


root = tk.Tk()
# Update frequency in milliseconds, max size of the deque
app = MouseTracker(root, update_frequency=100, max_size=100)
root.mainloop()
