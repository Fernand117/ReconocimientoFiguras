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

def detect_circles(image):
    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar el operador de Canny para detectar los bordes
    edges = cv2.Canny(gray_image, threshold1=50, threshold2=150)

    # Detección de círculos
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=62, minRadius=100, maxRadius=200)

    if circles is not None:
        circles = np.uint16(np.around(circles))

        # Obtener el índice del círculo con el radio más grande
        largest_circle_index = np.argmax(circles[0, :, 2])
        center_x, center_y, radius = circles[0, largest_circle_index]

        # Crear una copia de la imagen original para dibujar el círculo sin modificar la imagen original
        result_image = image.copy()

        # Dibujar el círculo más grande encontrado en la copia de la imagen original
        cv2.circle(result_image, (center_x, center_y), radius, (0, 255, 0), 2)

        # Crear una imagen binaria con los bordes cercanos al círculo
        binary_image = np.zeros_like(edges)
        max_distance = int(radius * 1)
        for y in range(center_y - max_distance, center_y + max_distance):
            for x in range(center_x - radius, center_x + radius):
                if edges[y, x] > 0:
                    binary_image[y, x] = 255

        # Encontrar los puntos más cercanos al círculo en el lado derecho e izquierdo
        closest_points_left = []
        closest_points_right = []
        for y in range(center_y - max_distance, center_y + max_distance):
            for x in range(center_x - radius, center_x):
                if binary_image[y, x] > 0:
                    # Verificar si el punto está fuera del círculo
                    distance_to_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                    if distance_to_center > radius:
                        closest_points_left.append((x, y))
            for x in range(center_x, center_x + radius):
                if binary_image[y, x] > 0:
                    # Verificar si el punto está fuera del círculo
                    distance_to_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                    if distance_to_center > radius:
                        closest_points_right.append((x, y))

        # Dibujar los puntos azules si están lo suficientemente cerca del círculo
        for closest_point in closest_points_left:
            cv2.circle(result_image, closest_point, 3, (255, 0, 0), -1)
        for closest_point in closest_points_right:
            cv2.circle(result_image, closest_point, 3, (0, 0, 255), -1)

        return result_image

    else:
        return image

# Cargar la imagen
image = cv2.imread('img/Captura de pantalla (161).png')

# Definir los rangos de color para los colores que deseas conservar
selected_color_ranges = [
    [(170, 130, 200), (250, 255, 255)],
    #[(160, 100, 100), (180, 255, 255)],
    [(0, 100, 100), (10, 255, 255)],
    [(170, 130, 200), (250, 255, 255)]
]

# Conservar solo los colores seleccionados en la imagen
image_with_selected_colors = keep_selected_colors(image, selected_color_ranges)

# Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(image_with_selected_colors, cv2.COLOR_BGR2GRAY)

# Aplicar el operador de Canny para detectar los bordes
edges = cv2.Canny(gray_image, threshold1=50, threshold2=150)

# Encontrar los contornos en la imagen binaria
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Trazar rectángulos alrededor de los contornos detectados
all_rectangles = []

# Trazar líneas verticales y horizontales
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    # Calcular el centro del rectángulo
    center_x = x + w // 2
    center_y = y + h // 2

    # Calcular el área del rectángulo
    area = w * h

    # Agregar el rectángulo a la lista de todos los rectángulos
    all_rectangles.append((x, y, w, h, center_x, center_y, area))

    vertical_lines_r = cv2.HoughLinesP(edges, rho=1, theta=np.pi/2, threshold=1, minLineLength=1, maxLineGap=1)

# Ordenar los rectángulos por área (de mayor a menor)
all_rectangles.sort(key=lambda rect: rect[6], reverse=True)

# Obtener el rectángulo más grande (el primero en la lista)
largest_rectangle = all_rectangles[0]

x, y, w, h = largest_rectangle[:4]

# Crear una máscara binaria del rectángulo más grande
mask = np.zeros(gray_image.shape, dtype=np.uint8)
cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

# Aplicar la máscara para eliminar los colores fuera del rectángulo
image_with_selected_colors = cv2.bitwise_and(image, image, mask=mask)
image_with_selected_colors = keep_selected_colors(image_with_selected_colors, selected_color_ranges)

cv2.rectangle(image_with_selected_colors, (x, y), (x + w, y + h), (0, 255, 0), 2)
# Definir un umbral de distancia (ajústalo según tus necesidades)
umbral_distancia = 40

# Coordenadas de las esquinas del rectángulo más grande
esquina_sup_izq = (x, y)
esquina_sup_der = (x + w, y)
esquina_inf_izq = (x, y + h)
esquina_inf_der = (x + w, y + h)

# Puntos más cercanos a las esquinas del rectángulo más grande (en la contraparte)
punto_sup_izq = None
punto_sup_der = None
punto_inf_izq = None
punto_inf_der = None

dist_minima_sup_izq = float('inf')
dist_minima_sup_der = float('inf')
dist_minima_inf_izq = float('inf')
dist_minima_inf_der = float('inf')

# Encontrar los puntos más cercanos a las esquinas del rectángulo (en la contraparte)
for line in vertical_lines_r:
    x1, y1, x2, y2 = line[0]

    # Calcular la distancia entre el punto y cada una de las esquinas del rectángulo
    dist_inf_izq = np.sqrt((x1 - esquina_inf_izq[0]) ** 2 + (y2 - esquina_inf_izq[1]) ** 2)
    dist_inf_der = np.sqrt((x1 - esquina_inf_der[0]) ** 2 + (y2 - esquina_inf_der[1]) ** 2)

    if dist_inf_izq < umbral_distancia:
        if punto_inf_izq is None or dist_inf_izq < dist_minima_inf_izq:
            punto_inf_izq = (x1 + 5, y2 + 6)
            dist_minima_inf_izq = dist_inf_izq

    if dist_inf_der < umbral_distancia:
        if punto_inf_der is None or dist_inf_der < dist_minima_inf_der:
            punto_inf_der = (x1, y2)
            dist_minima_inf_der = dist_inf_der

# Dibujar la línea horizontal desde el punto más cercano a la esquina inferior izquierda (punto_inf_izq) hacia la contraparte del rectángulo
if punto_inf_izq is not None and punto_inf_der is not None:
    cv2.line(image_with_selected_colors, punto_inf_izq, punto_inf_der, (0, 0, 255), 2)
    #cv2.putText(image_with_selected_colors, f"{punto_inf_izq}", punto_inf_izq, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.circle(image_with_selected_colors, punto_inf_izq, 1, (0, 255, 0), 2)
    #cv2.putText(image_with_selected_colors, f"{punto_inf_der}", punto_inf_der, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.circle(image_with_selected_colors, punto_inf_der, 1, (0, 255, 0), 2)

image_with_selected_colors = detect_circles(image_with_selected_colors)

window_width = 1080
window_height = 720
cv2.namedWindow('Resultado', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Resultado', window_width, window_height)
cv2.imshow('Resultado', image_with_selected_colors)
cv2.waitKey(0)
cv2.destroyAllWindows()