import cv2
import numpy as np

# Cargar la imagen
image = cv2.imread('img/Captura de pantalla (227).png')

# Convertir la imagen a espacio de color HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Definir los rangos de color para el rojo, amarillo y morado
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

lower_yellow = np.array([25, 100, 100])
upper_yellow = np.array([35, 255, 255])

lower_purple = np.array([140, 100, 100])
upper_purple = np.array([160, 255, 255])

# NUEVA MÁSCARA
lower_red_2 = np.array([160, 100, 100])
upper_red_2 = np.array([180, 255, 255])

# Definir el rango de color para los tonos naranjas
lower_orange = np.array([11, 100, 100])
upper_orange = np.array([25, 255, 255])

# Aplicar los umbrales de color para obtener las máscaras binarias
mask_red = cv2.inRange(hsv, lower_red, upper_red)
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)

mask_red_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

# Combinar las máscaras binarias
combined_mask = cv2.bitwise_or(mask_red, mask_purple, mask_red_2, mask_orange)

# Aplicar la máscara a la imagen original
masked_image = cv2.bitwise_and(image, image, mask=combined_mask)

# Aplicar el filtro de contraste a la imagen
gray_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
equalized_image = cv2.equalizeHist(gray_image)

# Convertir la imagen de vuelta a color para trazar las líneas en verde
lines_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)

# Encontrar los contornos en la imagen resultante
contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours is not None:
    for contour in contours:
        # Aproximar el contorno poligonal
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Dibujar las líneas rectas ajustadas al contorno
        cv2.drawContours(lines_image, [approx], 0, (0, 255, 0), 2)

window_width = 1980
window_height = 720

cv2.namedWindow('Resultado', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Resultado', window_width, window_height)

# Mostrar la imagen resultante
cv2.imshow('Resultado', lines_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
