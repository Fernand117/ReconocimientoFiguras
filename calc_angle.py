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

'''
start_point = (680, 30)
end_point = (970, 550)
color = (0, 0, 255)
thickness = 3
image_with_rectangle = cv2.rectangle(
    img = image_with_selected_colors,
    pt1 = start_point,
    pt2 = end_point, 
    color = color, 
    thickness = thickness
)
'''
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

    '''vertical_lines_l = cv2.HoughLinesP(edges, rho=1, theta=np.pi/2, threshold=1, minLineLength=40, maxLineGap=1)
    if vertical_lines_l is not None:
        for line in vertical_lines_l:
            x1, y1, x2, y2 = line[0]
            # Puntos de inicio y fin en el borde superior e inferior de la imagen
            cv2.line(image_with_selected_colors, (x1, 0), (x2, image_with_selected_colors.shape[0]), (0, 0, 255), 2)
            cv2.putText(image_with_selected_colors, f"x1_lvl: {x1}", (x1, x1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image_with_selected_colors, f"x2_lvl: {x2}", (x2, x2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image_with_selected_colors, f"y1_lvl: {y1}", (y1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image_with_selected_colors, f"y2_lvl: {y2}", (y2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    vertical_lines_r = cv2.HoughLinesP(edges, rho=1, theta=np.pi/2, threshold=1, minLineLength=1, maxLineGap=1)
    if vertical_lines_r is not None:
        for line in vertical_lines_r:
            x1, y1, x2, y2 = line[0]
            cv2.putText(image_with_selected_colors, f"{x1}", (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 255, 255), 2)

            #cv2.putText(image_with_selected_colors, f"{x1}", (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 255, 255), 2)
            # Puntos de inicio y fin en el borde superior e inferior de la imagen
            #cv2.line(image_with_selected_colors, (x1, y2), (x2, y2), (0, 0, 255), 2)
            #vertical_line = f"Linea vertical"
            #cv2.putText(image_with_selected_colors, f"{x1}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 0), 2)
            #cv2.line(image_with_selected_colors, (x1, y2), (x1, y1), (0, 0, 255), 1)
            #cv2.putText(image_with_selected_colors, f"y1_lvr: {y1}", (y1, x1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            #cv2.putText(image_with_selected_colors, f"y2_lvr: {y2}", (y2, x2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    horizontal_lines_l = cv2.HoughLinesP(edges, rho=1, theta=np.pi/2, threshold=1, minLineLength=22, maxLineGap=1)
    if horizontal_lines_l is not None:
        for line in horizontal_lines_l:
            x1, y1, x2, y2 = line[0]
            # Puntos de inicio y fin en el borde superior e inferior de la imagen
            cv2.line(image_with_selected_colors, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_with_selected_colors, f"x1_lhl: {x1}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image_with_selected_colors, f"x2_lhl: {x2}", (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image_with_selected_colors, f"y1_lhl: {y1}", (y1, x1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image_with_selected_colors, f"y2_lhl: {y2}", (y2, x2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    '''

# Ordenar los rectángulos por área (de mayor a menor)
all_rectangles.sort(key=lambda rect: rect[6], reverse=True)

# Obtener el rectángulo más grande (el primero en la lista)
largest_rectangle = all_rectangles[0]

x, y, w, h = largest_rectangle[:4]
cv2.rectangle(image_with_selected_colors, (x, y), (x + w, y + h), (0, 255, 0), 2)

window_width = 1080
window_height = 720
cv2.namedWindow('Resultado', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Resultado', window_width, window_height)
cv2.imshow('Resultado', image_with_selected_colors)
cv2.waitKey(0)
cv2.destroyAllWindows()