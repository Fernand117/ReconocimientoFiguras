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
gray_image = cv2.cvtColor(image_with_selected_colors, cv2.COLOR_BGR2GRAY)

# Aplicar un umbral para convertir la imagen en binaria
_, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

# Buscar contornos en la imagen binaria
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Crear una copia de la imagen original para dibujar los contornos
contour_image = image.copy()

# Dibujar los contornos encontrados en la copia de la imagen original
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)

# Lista para almacenar los rectángulos mínimos de cada contorno
bounding_rectangles = []

# Recorrer cada contorno encontrado
for contour in contours:
    # Obtener el rectángulo mínimo que encierra el contorno
    rect = cv2.minAreaRect(contour)
    bounding_rectangles.append(rect)

# Crear una copia de la imagen original para dibujar los rectángulos mínimos
rectangle_image = image.copy()

# Dibujar los rectángulos mínimos en la copia de la imagen original
for rect in bounding_rectangles:
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(rectangle_image, [box], 0, (255, 0, 0), 1)

# Función para calcular el ángulo de desviación a partir de un rectángulo mínimo
def calculate_angle(rect):
    # Obtener el ángulo de rotación del rectángulo (en grados)
    angle = rect[-1]

    # Si el ángulo es menor que -45 grados, ajustarlo para obtener el ángulo positivo equivalente
    if angle < -45:
        angle += 90

    # Calcular el ángulo de desviación respecto al eje horizontal
    deviation_angle = 90 - abs(angle)

    return deviation_angle

# Calcular y mostrar el ángulo de desviación para cada rectángulo mínimo
for rect in bounding_rectangles:
    deviation_angle = calculate_angle(rect)
    print("Ángulo de desviación: {:.2f} grados".format(deviation_angle))

window_width = 1980
window_height = 720

cv2.namedWindow('Resultado', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Resultado', window_width, window_height)

cv2.imshow('Resultado', image_with_selected_colors)
cv2.waitKey(0)
cv2.destroyAllWindows()