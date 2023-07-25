# Función para calcular el ángulo de desviación a partir de un rectángulo mínimo
def calculate_angle(rect):
    # Obtener el ángulo de rotación del rectángulo (en grados)
    angle = rect[-1]

    # Si el ángulo es menor que -45 grados, ajustarlo para obtener el ángulo positivo equivalente
    if angle < -45:
        angle += 90

    # Calcular el ángulo de desviación respecto al eje horizontal
    deviation_angle = 90 - abs(angle)

    return deviation_angle

# Calcular y mostrar el ángulo de desviación para cada rectángulo mínimo
for rect in bounding_rectangles:
    deviation_angle = calculate_angle(rect)
    print("Ángulo de desviación: {:.2f} grados".format(deviation_angle))

result = np.hstack((contour_image, rectangle_image))
