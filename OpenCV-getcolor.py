import cv2
import numpy as np

def get_color_values(event, x, y, flags, param):
    global point1, point2, drawing, image_copy
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        point1 = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        point2 = (x, y)
        cv2.rectangle(image_copy, point1, point2, (0, 255, 0), 2)
        cv2.imshow('Imagen', image_copy)

def remove_color_range(image, point1, point2, color_lower, color_upper):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[point1[1]:point2[1], point1[0]:point2[0]] = 255
    mask = cv2.bitwise_not(mask)

    color_mask = cv2.inRange(hsv, color_lower, color_upper)
    mask = cv2.bitwise_and(mask, mask, mask=color_mask)

    image_without_color = cv2.bitwise_and(image, image, mask=mask)
    return image_without_color

# Cargar la imagen
image = cv2.imread('img/Captura de pantalla (167).png')
image_copy = image.copy()

# Variables para el área seleccionada
point1, point2 = (-1, -1), (-1, -1)
drawing = False

# Crear una ventana para la imagen
window_width = 1980
window_height = 720

cv2.namedWindow('Imagen', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Imagen', window_width, window_height)
cv2.namedWindow('Imagen')
cv2.imshow('Imagen', image)

# Establecer la función de detección de eventos del mouse
cv2.setMouseCallback('Imagen', get_color_values)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Presionar la tecla 'Esc' para salir del bucle
        break

    # Si se ha seleccionado un área válida, continuar con las operaciones
    if point1[0] != -1 and point2[0] != -1:
        # Verificar si el área seleccionada tiene un tamaño mayor que cero
        if point1[0] < point2[0] and point1[1] < point2[1]:
            # Definir el rango de color para el área seleccionada
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            color_lower = np.min(hsv[point1[1]:point2[1], point1[0]:point2[0]], axis=(0, 1))
            color_upper = np.max(hsv[point1[1]:point2[1], point1[0]:point2[0]], axis=(0, 1))
            print("Color Lower Bound:", color_lower)
            print("Color Upper Bound:", color_upper)

            # Eliminar el rango de colores de la imagen
            image_without_color = remove_color_range(image, point1, point2, color_lower, color_upper)
            window_width = 1980
            window_height = 720

            cv2.namedWindow('Imagen sin color', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Imagen sin color', window_width, window_height)
            cv2.namedWindow('Imagen sin color')
            cv2.imshow('Imagen sin color', image_without_color)
        else:
            print("El área seleccionada es inválida. Asegúrate de seleccionar un área válida antes de continuar.")

cv2.destroyAllWindows()
