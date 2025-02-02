import tkinter as tk
import time
from collections import deque


class MouseRecorder:
    def __init__(self, root, update_frequency=50, max_length=100):
        self.root = root
        self.update_position = False
        self.last_event = None
        self.update_frequency = update_frequency  # Update frequency in milliseconds
        self.positions = deque(maxlen=max_length)
        self.canvas = tk.Canvas(root, width=500, height=500)
        self.canvas.pack()
        self.canvas.bind("<Motion>", self.motion)
        self.canvas.bind("<Button-1>", self.toggle_update)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def motion(self, event):
        self.last_event = event

    def showxy(self):
        if self.update_position and self.last_event:
            xm, ym = self.last_event.x, self.last_event.y
            timestamp = time.time()
            if len(self.positions) == self.positions.maxlen:
                old_x, old_y, _ = self.positions[0]
                self.canvas.create_rectangle(
                    old_x-2, old_y-2, old_x+2, old_y+2, fill='white')
            self.positions.append((xm, ym, timestamp))
            self.canvas.create_rectangle(xm-2, ym-2, xm+2, ym+2, fill='black')
            str1 = f"mouse at x={xm}  y={ym}"
            self.root.title(str1)
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
app = MouseRecorder(
    root,
    update_frequency=30,
    max_length=1000,
)
root.mainloop()
