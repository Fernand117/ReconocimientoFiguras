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
image = cv2.imread('img/Captura de pantalla (164).png')

# Definir los rangos de color para los colores que deseas conservar
selected_color_ranges = [
    [(170, 130, 200), (250, 255, 255)]
]

# Conservar solo los colores seleccionados en la imagen
image_with_selected_colors = keep_selected_colors(image, selected_color_ranges)

# Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(image_with_selected_colors, cv2.COLOR_BGR2GRAY)

# Aplicar el operador de Canny para detectar los bordes
edges = cv2.Canny(gray_image, threshold1=30, threshold2=100)

# Encontrar los contornos en la imagen binaria
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Trazar líneas verticales y horizontales
for contour in contours:
    vertical_lines_l = cv2.HoughLinesP(edges, rho=1, theta=np.pi/2, threshold=1, minLineLength=40, maxLineGap=1)
    if vertical_lines_l is not None:
        for line in vertical_lines_l:
            x1, y1, x2, y2 = line[0]
            # Puntos de inicio y fin en el borde superior e inferior de la imagen
            cv2.line(image_with_selected_colors, (x1, 0), (x2, image_with_selected_colors.shape[0]), (0, 0, 255), 2)

    vertical_lines_r = cv2.HoughLinesP(edges, rho=1, theta=np.pi/2, threshold=1, minLineLength=30, maxLineGap=1)
    if vertical_lines_r is not None:
        for line in vertical_lines_r:
            x1, y1, x2, y2 = line[0]
            # Puntos de inicio y fin en el borde superior e inferior de la imagen
            cv2.line(image_with_selected_colors, (x1, 0), (x2, image_with_selected_colors.shape[1]), (0, 0, 255), 2)
    
    horizontal_lines_l = cv2.HoughLinesP(edges, rho=1, theta=np.pi/2, threshold=1, minLineLength=22, maxLineGap=1)
    if horizontal_lines_l is not None:
        for line in horizontal_lines_l:
            x1, y1, x2, y2 = line[0]
            # Puntos de inicio y fin en el borde superior e inferior de la imagen
            cv2.line(image_with_selected_colors, (0, y1), (image_with_selected_colors.shape[0], y2), (0, 255, 0), 2)

window_width = 1980
window_height = 720
cv2.namedWindow('Resultado', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Resultado', window_width, window_height)
cv2.imshow('Resultado', image_with_selected_colors)
cv2.waitKey(0)
cv2.destroyAllWindows()