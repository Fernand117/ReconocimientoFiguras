import cv2
import numpy as np

def remove_color(image, color_lower, color_upper):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, color_lower, color_upper)
    masked_image = cv2.bitwise_and(image, image, mask=~mask)
    return masked_image

# Cargar la imagen
image = cv2.imread('img/Captura de pantalla (161).png')

# Convertir el color a espacio HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Definir los rangos de color para los tonos morados, violetas y magenta
lower_purple1 = np.array([125, 100, 100])
upper_purple1 = np.array([145, 255, 255])

lower_purple2 = np.array([145, 100, 100])
upper_purple2 = np.array([155, 255, 255])

lower_violet = np.array([155, 100, 100])
upper_violet = np.array([170, 255, 255])

lower_magenta = np.array([170, 100, 100])
upper_magenta = np.array([180, 255, 255])

# Crear las máscaras para los píxeles dentro de los rangos de color deseados
mask_purple1 = cv2.inRange(hsv, lower_purple1, upper_purple1)
mask_purple2 = cv2.inRange(hsv, lower_purple2, upper_purple2)
mask_violet = cv2.inRange(hsv, lower_violet, upper_violet)
mask_magenta = cv2.inRange(hsv, lower_magenta, upper_magenta)

# Combinar las máscaras binarias para abarcar todos los tonos de morado
combined_mask = cv2.bitwise_or(mask_purple1, mask_purple2, mask_violet)


# Aplicar la máscara a la imagen original para eliminar el color
image_without_color = cv2.bitwise_and(image, image, mask=~combined_mask)

window_width = 1980
window_height = 720

cv2.namedWindow('Imagen Original', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Imagen Original', window_width, window_height)
cv2.imshow('Imagen Original', image)

cv2.namedWindow('Imagen sin Color', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Imagen sin Color', window_width, window_height)
cv2.imshow('Imagen sin Color', image_without_color)

cv2.waitKey(0)
cv2.destroyAllWindows()
