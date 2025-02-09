import tkinter as tk
import time
from collections import deque
import os
import json
import numpy as np


SAVE = True
save_name = "eight"
save_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    os.pardir,
    os.pardir,
    "data",
)


class MouseRecorder:
    def __init__(
        self,
        root,
        save_path: str,
        save_name: str,
        update_frequency: int = 50,
        max_length: int = 100,
        save: bool = False,
    ):
        self.root = root
        self.update_position = False
        self.last_event = None
        self.update_frequency = update_frequency  # Update frequency in milliseconds

        self.recordings = []
        self.max_length = max_length

        self.save = save
        self.save_path = save_path
        self.save_name = save_name

        self.canvas = tk.Canvas(root, width=800, height=800)
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
            if len(self.recordings[-1]) == self.recordings[-1].maxlen:
                old_x, old_y, _ = self.recordings[-1][0]
                self.canvas.create_rectangle(
                    old_x - 2, old_y - 2, old_x + 2, old_y + 2, fill="white"
                )

            # append the new position and timestamp
            self.recordings[-1].append((xm, ym, timestamp))
            # draw the new position
            self.canvas.create_rectangle(
                xm - 2, ym - 2, xm + 2, ym + 2, fill="black")
            # if we are past half of the max length, turn window backgropund to yellow
            l = len(self.recordings[-1])
            if (l > self.max_length / 2) and (l < self.max_length):
                self.canvas.config(bg="yellow")
            # if we are past the maximum length, turn window background to red
            elif len(self.recordings[-1]) == self.max_length:
                self.canvas.config(bg="red")
            else:
                self.canvas.config(bg="white")

            str1 = f"mouse at x={xm}  y={ym}"
            self.root.title(str1)

        self.root.after(self.update_frequency, self.showxy)

    def toggle_update(self, event):
        self.update_position = not self.update_position
        if self.update_position:
            self.recordings.append(deque(maxlen=self.max_length))
            self.showxy()
        else:
            # clean the canvas
            self.canvas.delete("all")

    def on_closing(self):
        # Save the positions and timestamps
        if self.save:
            for i, sequence in enumerate(self.recordings):
                # converting to numpy and back to a list is not very efficient
                # but slicing with indexes is very convenient on numpy arrays
                # and also normalizing is easier
                sequence = np.array(sequence)
                p = sequence[:, :2]

                # revert y coordinates, those in the windows are top to bottom
                # but would like them to be the opposite
                p[:, 1] = self.canvas.winfo_height() - p[:, 1]

                # normalize in the range [0, 1] but keeping the aspect ratio
                p = p / max(self.canvas.winfo_width(),
                            self.canvas.winfo_height())

                seq_dict = {
                    "p": p.tolist(),
                    "t": sequence[:, 2].tolist(),
                }
                with open(
                    os.path.join(self.save_path,
                                 f"{self.save_name}_{i}.json"), "w"
                ) as f:
                    json.dump(seq_dict, f)
        # close the window
        self.root.destroy()


root = tk.Tk()
app = MouseRecorder(
    root,
    save_path=save_path,
    save_name=save_name,
    update_frequency=100,
    max_length=500,
    save=SAVE,
)
root.mainloop()
