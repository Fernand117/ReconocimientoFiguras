import cv2
import numpy as np

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
image = cv2.imread('img/Captura de pantalla (161).png')

# Definir los rangos de color para los colores que deseas conservar
selected_color_ranges = [
    [(170, 130, 200), (250, 255, 255)]
]

# Conservar solo los colores seleccionados en la imagen
image_with_selected_colors = keep_selected_colors(image, selected_color_ranges)

# Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(image_with_selected_colors, cv2.COLOR_BGR2GRAY)

# Aplicar un umbral a la imagen para convertirla en blanco y negro
_, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

# Encontrar los contornos de la imagen binaria
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar los contornos en la imagen original
#cv2.drawContours(image_with_selected_colors, contours, -1, (0, 255, 0), 3)

# Utilizar la transformada de Hough probabilística para detectar líneas horizontales
lines = cv2.HoughLinesP(binary_image, 1, np.pi/180, threshold=70, minLineLength=1, maxLineGap=10)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Filtrar líneas horizontales (pequeña diferencia en el eje Y, diferencia significativa en el eje X)
        if np.abs(y2 - y1) < 10 and np.abs(x2 - x1) > 10:
            cv2.line(image_with_selected_colors, (x1, y1), (x2, y2), (0, 255, 0), 2)

window_width = 1980
window_height = 720

cv2.namedWindow('Resultado', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Resultado', window_width, window_height)

cv2.imshow('Resultado', image_with_selected_colors)
cv2.waitKey(0)
cv2.destroyAllWindows()
