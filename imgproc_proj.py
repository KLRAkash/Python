import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk

IMAGE_SIZE = (300, 300)

def detect_cracks(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Could not open or read image '{image_path}'.")
        return None, None, None

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel, iterations=2)

    edges = cv2.Canny(morph, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 100
    filtered_edges = np.zeros_like(edges)

    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if aspect_ratio < 10:  # Filter based on aspect ratio
                cv2.drawContours(filtered_edges, [contour], -1, 255, -1)

    if np.sum(filtered_edges) > 0:
        result = "Crack detected."
    else:
        result = "No crack detected."

    return image, thresholded, filtered_edges, result

def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image, thresholded, edges, result = detect_cracks(file_path)
        if image is not None:
            display_images(image, thresholded, edges)
            result_label.config(text=result)

def display_images(image, thresholded, edges):
    images = [image, thresholded, edges]
    for i, img in enumerate(images):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, IMAGE_SIZE)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)

        if i == 0:
            original_label.config(image=img)
            original_label.image = img
        elif i == 1:
            thresholded_label.config(image=img)
            thresholded_label.image = img
        elif i == 2:
            edges_label.config(image=img)
            edges_label.image = img

root = tk.Tk()
root.title("Crack Detection")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

frame.columnconfigure(0, weight=1)
frame.columnconfigure(1, weight=1)
frame.columnconfigure(2, weight=1)

title_label = ttk.Label(frame, text="Crack Detection System", font=("Helvetica", 20, "bold"))
title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10), sticky=tk.N)

select_button = ttk.Button(frame, text="Select Image", command=select_image)
select_button.grid(row=1, column=0, columnspan=3, pady=10)

result_label = ttk.Label(frame, text="", font=("Helvetica", 12, "italic"))
result_label.grid(row=2, column=0, columnspan=3, pady=10)

original_title = ttk.Label(frame, text="Original Image", font=("Helvetica", 12, "bold"))
original_title.grid(row=3, column=0, pady=10)

thresholded_title = ttk.Label(frame, text="Thresholded Image", font=("Helvetica", 12, "bold"))
thresholded_title.grid(row=3, column=1, pady=10)

edges_title = ttk.Label(frame, text="Edges Image", font=("Helvetica", 12, "bold"))
edges_title.grid(row=3, column=2, pady=10)

original_label = ttk.Label(frame)
original_label.grid(row=4, column=0, padx=10, pady=10)

thresholded_label = ttk.Label(frame)
thresholded_label.grid(row=4, column=1, padx=10, pady=10)

edges_label = ttk.Label(frame)
edges_label.grid(row=4, column=2, padx=10, pady=10)

root.mainloop()
