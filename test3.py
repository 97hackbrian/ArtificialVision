import cv2
import depthai

# Configuración de la cámara DepthAI
pipeline = depthai.Pipeline()

# Configuración de salida de disparidad
stereo = pipeline.createStereoDepth()
stereo.setConfidenceThreshold(200)

# Configuración de la cámara monocromática (rectificada)
mono_left = pipeline.createMonoCamera()
mono_left.setBoardSocket(depthai.CameraBoardSocket.LEFT)

mono_right = pipeline.createMonoCamera()
mono_right.setBoardSocket(depthai.CameraBoardSocket.RIGHT)

# Conectar cámaras mono a la entrada de disparidad
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

# Configuración de las salidas
xout_left = pipeline.createXLinkOut()
xout_left.setStreamName("rectified_left")
stereo.rectifiedLeft.link(xout_left.input)

xout_right = pipeline.createXLinkOut()
xout_right.setStreamName("rectified_right")
stereo.rectifiedRight.link(xout_right.input)

xout_disp = pipeline.createXLinkOut()
xout_disp.setStreamName("disparity")
stereo.disparity.link(xout_disp.input)

# Inicializar la cámara
with depthai.Device(pipeline) as device:
    while True:
        # Obtener los cuadros
        left_frame = device.getOutputQueue("rectified_left", 8, blocking=False).get().getCvFrame()
        right_frame = device.getOutputQueue("rectified_right", 8, blocking=False).get().getCvFrame()
        disp_frame = device.getOutputQueue("disparity", 8, blocking=False).get().getCvFrame()

        # Calcular el desplazamiento en píxeles
        displacement_pixels = cv2.mean(disp_frame)[0]

        # Mostrar las imágenes y el desplazamiento
        cv2.imshow("Left Frame", left_frame)
        cv2.imshow("Right Frame", right_frame)
        cv2.imshow("Disparity", disp_frame)

        print(f"Desplazamiento en píxeles: {displacement_pixels:.2f}")

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar recursos y cerrar ventanas
cv2.destroyAllWindows()
