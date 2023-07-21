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
    #[(10, 100, 100), (10, 255, 255)],  # Rojo
    [(170, 130, 200), (250, 255, 255)]
    #[(25, 100, 100), (35, 255, 255)],  # Amarillo
    #[(140, 100, 100), (160, 255, 255)],  # Morado
    # Agrega más rangos de colores aquí si deseas conservar otros colores
]

# Conservar solo los colores seleccionados en la imagen
image_with_selected_colors = keep_selected_colors(image, selected_color_ranges)

# Encontrar los píxeles no cero en la imagen
non_zero_pixels = cv2.findNonZero(cv2.cvtColor(image_with_selected_colors, cv2.COLOR_BGR2GRAY))

for coord in non_zero_pixels:
    x, y = coord[0]  # Obtener las coordenadas x, y del píxel no cero
    cv2.circle(image_with_selected_colors, (x, y), 1, (0, 255, 0), -1)  # Marcar la coordenada con un círculo verde

window_width = 1980
window_height = 720

cv2.namedWindow('Resultado', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Resultado', window_width, window_height)

cv2.imshow('Resultado', image_with_selected_colors)
cv2.waitKey(0)
cv2.destroyAllWindows()