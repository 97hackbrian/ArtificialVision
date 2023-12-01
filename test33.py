import cv2
import depthai
import open3d as o3d
import numpy as np

# Configuración de la cámara DepthAI
pipeline = depthai.Pipeline()

# Configuración de la salida de disparidad y profundidad
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

# Configuración de la salida de puntos en 3D
xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# Inicializar la cámara
with depthai.Device(pipeline) as device:
    # Configurar la nube de puntos 3D
    pcd = o3d.geometry.PointCloud()

    while True:
        # Obtener el cuadro actual y la nube de puntos en 3D
        frame = device.getOutputQueue("rectified_left", 8, blocking=False).get().getCvFrame()
        depth_frame = device.getOutputQueue("depth", 8, blocking=False).get().getCvFrame()

        # Convertir la imagen de profundidad a una matriz numpy
        depth_data = depth_frame / 255.0  # Normalizar a valores entre 0 y 1
        depth_data = np.uint16(depth_data * 1000)  # Convertir a milímetros
        depth_data = depth_data.astype(np.uint16)

        # Configurar la nube de puntos 3D con coordenadas XYZ y color de la imagen original
        h, w, _ = frame.shape
        pcd.points = o3d.utility.Vector3dVector(np.zeros((h * w, 3)))
        pcd.colors = o3d.utility.Vector3dVector(frame.reshape(-1, 3) / 255.0)

        # Calcular las coordenadas 3D y actualizar la nube de puntos
        for y in range(h):
            for x in range(w):
                z = depth_data[y, x]
                if z > 0:
                    pcd.points[y * w + x] = ((x - w / 2) * z / stereo.getFocalLength(), (y - h / 2) * z / stereo.getFocalLength(), z)

        # Visualizar la nube de puntos en 3D
        o3d.visualization.draw_geometries([pcd])

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
