import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def keep_selected_colors(image, color_ranges):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for color_range in color_ranges:
        lower_color = np.array(color_range[0])
        upper_color = np.array(color_range[1])
        color_mask = cv2.inRange(hsv, lower_color, upper_color)
        mask = cv2.bitwise_or(mask, color_mask)

    image_with_selected_colors = cv2.bitwise_and(image, image, mask=mask)
    return image_with_selected_colors

# Cargar la imagen
image = cv2.imread('img/Captura de pantalla (164).png')

# Definir los rangos de color para los colores que deseas conservar
selected_color_ranges = [
    #[(10, 100, 100), (10, 255, 255)],  # Rojo
    [(170, 130, 200), (250, 255, 255)]
    #[(25, 100, 100), (35, 255, 255)],  # Amarillo
    #[(140, 100, 100), (160, 255, 255)],  # Morado
    # Agrega más rangos de colores aquí si deseas conservar otros colores
]

# Conservar solo los colores seleccionados en la imagen
image_with_selected_colors = keep_selected_colors(image, selected_color_ranges)

# Convertir la imagen a escala de grises
gray = cv2.cvtColor(image_with_selected_colors, cv2.COLOR_BGR2GRAY)

# Aplicar detección de contornos
edges = cv2.Canny(gray, 50, 150)

# Encontrar los contornos en la imagen
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Función para clasificar las figuras
def classify_shape(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    
    num_vertices = len(approx)
    
    if num_vertices == 3:
        return "Triangulo"
    elif num_vertices == 4:
        # Verificar si es cuadrado o rectángulo
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.95 <= aspect_ratio <= 1.05:
            return "Cuadrado"
        else:
            return "Rectangulo"
    elif num_vertices == 5:
        return "Pentagono"
    else:
        return "Otro"

# Función para asignar colores a las figuras
def assign_color(shape_type):
    if shape_type == "Triangulo":
        return (0, 255, 0)  # Verde
    elif shape_type == "Cuadrado":
        return (0, 0, 255)  # Rojo
    elif shape_type == "Rectangulo":
        return (255, 0, 0)  # Azul
    elif shape_type == "Pentagono":
        return (255, 255, 0)  # Amarillo
    else:
        return (0, 0, 0)  # Negro

# Dibujar las figuras identificadas en la imagen original
for contour in contours:
    shape_type = classify_shape(contour)
    color = assign_color(shape_type)
    cv2.drawContours(image_with_selected_colors, [contour], -1, color, 2)

window_width = 1980
window_height = 720

cv2.namedWindow('Resultado', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Resultado', window_width, window_height)

cv2.imshow('Resultado', image_with_selected_colors)
cv2.waitKey(0)
cv2.destroyAllWindows()
