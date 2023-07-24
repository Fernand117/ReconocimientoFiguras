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

# Encontrar los píxeles no cero en la imagen
non_zero_pixels = cv2.findNonZero(cv2.cvtColor(image_with_selected_colors, cv2.COLOR_BGR2GRAY))

# Verificar si se encontraron píxeles no cero
if non_zero_pixels is not None:
    # Obtener solo las coordenadas x, y de los píxeles no cero
    coordinates = [coord[0] for coord in non_zero_pixels]

    # Agrupar las coordenadas cercanas usando DBSCAN
    eps = 1  # Valor para determinar la cercanía de las coordenadas
    db = DBSCAN(eps=eps, min_samples=1).fit(coordinates)
    labels = db.labels_
    unique_labels = np.unique(labels)

    # Función para obtener las coordenadas de las esquinas de una agrupación
    def get_corners(coords):
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]
        return (min(x_coords), min(y_coords)), (max(x_coords), max(y_coords))

    # Trazar líneas para las esquinas de cada agrupación
    for label in unique_labels:
        if label == -1:
            continue  # Ignorar puntos ruido (sin agrupación)
        coords_group = [coordinates[i] for i, db_label in enumerate(labels) if db_label == label]
        corner1, corner2 = get_corners(coords_group)
        cv2.rectangle(image_with_selected_colors, corner1, corner2, (0, 255, 0), 1)
        #cv2.line(image_with_selected_colors, corner1, corner2, (0, 255, 0), 1)
else:
    print("No se encontraron píxeles no cero en la imagen.")

window_width = 1980
window_height = 720

cv2.namedWindow('Resultado', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Resultado', window_width, window_height)

cv2.imshow('Resultado', image_with_selected_colors)
cv2.waitKey(0)
cv2.destroyAllWindows()