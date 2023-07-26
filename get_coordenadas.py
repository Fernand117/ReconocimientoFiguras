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
    [(170, 130, 200), (250, 255, 255)],
    [(160, 100, 100), (180, 255, 255)],
    [(0, 100, 100), (10, 255, 255)]
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
    #cv2.rectangle(image_with_selected_colors, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
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
punto_inf_izq = None
dist_minima_sup_izq = float('inf')
dist_minima_inf_izq = float('inf')

# Encontrar los puntos más cercanos a las esquinas del rectángulo (en la contraparte)
for line in vertical_lines_r:
    x1, y1, x2, y2 = line[0]
    # Calcular la distancia entre el punto y cada una de las esquinas del rectángulo
    dist_sup_izq = np.sqrt((x1 - esquina_sup_izq[0]) ** 2 + (y2 - esquina_sup_izq[1]) ** 2)
    dist_sup_der = np.sqrt((x1 - esquina_sup_der[0]) ** 2 + (y2 - esquina_sup_der[1]) ** 2)
    dist_inf_izq = np.sqrt((x1 - esquina_inf_izq[0]) ** 2 + (y2 - esquina_inf_izq[1]) ** 2)
    dist_inf_der = np.sqrt((x1 - esquina_inf_der[0]) ** 2 + (y2 - esquina_inf_der[1]) ** 2)

    # Si alguna de las distancias es menor que el umbral, pintar el punto
    if dist_sup_izq < umbral_distancia or dist_sup_der < umbral_distancia or dist_inf_izq < umbral_distancia or dist_inf_der < umbral_distancia:
        cv2.putText(image_with_selected_colors, f"{x1}", (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 255, 255), 2)

    if dist_inf_izq < 100:
        punto_inf_izq = (x1, y2)
        dist_minima_inf_izq = dist_inf_izq

# Dibujar la línea horizontal desde el punto más cercano a la esquina inferior izquierda (punto_inf_izq) hacia la contraparte del rectángulo
if punto_inf_izq is not None:
    cv2.line(image_with_selected_colors, (punto_inf_izq[0], esquina_inf_izq[1]), (x + w, esquina_inf_izq[1]), (0, 0, 255), 2)

window_width = 1080
window_height = 720
cv2.namedWindow('Resultado', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Resultado', window_width, window_height)
cv2.imshow('Resultado', image_with_selected_colors)
cv2.waitKey(0)
cv2.destroyAllWindows()