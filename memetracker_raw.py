import os
import cv2
import math
import numpy as np
import mediapipe as mp

# Manejo de importaciones compatibles con distintas versiones de MediaPipe
try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    try:
        from mediapipe.python.solutions import hands as mp_hands
        from mediapipe.python.solutions.drawing_utils as mp_drawing
    except ModuleNotFoundError:
        import mediapipe.python.solutions.hands as mp_hands
        import mediapipe.python.solutions.drawing_utils as mp_drawing

# =============================================================================
# 1. CONFIGURACIÓN Y PARÁMETROS EDITABLES
# =============================================================================

# Ruta relativa al directorio de imágenes.
# Personaliza esta carpeta según la estructura de tu proyecto local o repositorio.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "assets", "images")

# Diccionario de mapeo entre identificadores de poses y archivos de imagen.
# Reemplaza los nombres de archivo por tus propias imágenes de superposición (.jpg, .jpeg, .png).
POSE_IMAGES = {
    "raised_fist_high": os.path.join(IMAGE_DIR, "pose_raised_fist.jpg"), 
    "v_sign": os.path.join(IMAGE_DIR, "pose_v_sign.jpg"),
    "heart_sign": os.path.join(IMAGE_DIR, "pose_heart.jpg"),
    "think_sign": os.path.join(IMAGE_DIR, "pose_think.jpg"),
    "point_down": os.path.join(IMAGE_DIR, "pose_point_down.jpg"),
    "palm_sign": os.path.join(IMAGE_DIR, "pose_palm.jpg"),
    "shush_sign": os.path.join(IMAGE_DIR, "pose_shush.jpeg"),
    "fist_standard": os.path.join(IMAGE_DIR, "pose_fist.jpg"),
}

# Parámetros de ajuste para el modelo de MediaPipe Hands
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS = 2

# Escala y márgenes para la superposición de imágenes emergentes (Overlay)
OVERLAY_SCALE = 0.30       # Porcentaje de ancho del frame que ocupará la imagen
OVERLAY_MARGIN_X = 10     # Margen en píxeles desde el borde izquierdo
OVERLAY_MARGIN_Y = 10     # Margen en píxeles desde el borde superior

# Configuración de resolución de entrada de la cámara web (HD por defecto)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


# =============================================================================
# 2. DETECTOR Y CLASIFICADOR VECTORIAL ADAPTATIVO
# =============================================================================

class HandPoseDetector:
    """
    Clase encargada de inicializar el rastreador de manos de MediaPipe,
    extraer puntos de referencia (landmarks) y clasificar gestos según geometría vectorial.
    """
    def __init__(self):
        # Inicialización de la solución de MediaPipe Hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        self.mp_draw = mp_drawing
        
        # Índices de los puntos clave de los dedos en MediaPipe Hands
        self.TIP_IDS = [4, 8, 12, 16, 20]  # Puntas: Pulgar, Índice, Medio, Anular, Meñique
        self.PIP_IDS = [3, 6, 10, 14, 18]  # Articulaciones intergalángicas proximales
        self.MCP_IDS = [2, 5, 9, 13, 17]   # Articulaciones metacarpofalángicas

    def find_hands(self, img, draw=True):
        """Procesa una imagen en BGR, la convierte a RGB y dibuja los puntos clave detectados."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks and draw:
            for hand_lms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
        return img

    def get_landmarks_list(self, img, hand_index=0):
        """Convierte las coordenadas normalizadas de los landmarks a coordenadas en píxeles de la imagen."""
        lm_list = []
        if self.results.multi_hand_landmarks and hand_index < len(self.results.multi_hand_landmarks):
            my_hand = self.results.multi_hand_landmarks[hand_index]
            h, w, _ = img.shape
            for lm in my_hand.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))
        return lm_list

    def get_hand_label(self, hand_index=0):
        """Retorna la lateralidad detectada para la mano dada ('Left' o 'Right')."""
        if self.results.multi_handedness and hand_index < len(self.results.multi_handedness):
            return self.results.multi_handedness[hand_index].classification[0].label
        return None

    def _is_finger_extended_dist(self, lm_list, finger_idx):
        """Evalúa si un dedo está extendido comparando la distancia euclidiana de la punta a la muñeca."""
        wrist = lm_list[0]
        tip = lm_list[self.TIP_IDS[finger_idx]]
        pip = lm_list[self.PIP_IDS[finger_idx]]
        
        dist_tip = math.hypot(tip[0] - wrist[0], tip[1] - wrist[1])
        dist_pip = math.hypot(pip[0] - wrist[0], pip[1] - wrist[1])
        return dist_tip > dist_pip

    # --- REGLAS GEOMÉTRICAS DE CLASIFICACIÓN DE POSES ---

    def is_pose_raised_fist(self, lm_list_list, labels, img_height):
        """Detecta un puño cerrado en la parte alta del encuadre (mitad superior)."""
        for lm_list in lm_list_list:
            if not lm_list:
                continue
            
            extended_count = sum(1 for f in range(1, 5) if self._is_finger_extended_dist(lm_list, f))
            wrist_y = lm_list[0][1]
            
            if extended_count == 0 and wrist_y < (img_height * 0.55):
                return True
        return False

    def is_pose_heart(self, lm_list_list, labels):
        """Detecta la seña de corazón formada uniendo ambas manos."""
        if len(lm_list_list) < 2:
            return False

        lm_1, lm_2 = lm_list_list[0], lm_list_list[1]
        dist_thumbs = math.hypot(lm_1[4][0] - lm_2[4][0], lm_1[4][1] - lm_2[4][1])
        dist_index_mcp = math.hypot(lm_1[5][0] - lm_2[5][0], lm_1[5][1] - lm_2[5][1])
        hand_scale = math.hypot(lm_1[0][0] - lm_1[9][0], lm_1[0][1] - lm_1[9][1])
        
        if dist_thumbs < (hand_scale * 1.5) and dist_index_mcp < (hand_scale * 2.2):
            return True
        return False

    def is_pose_down_index(self, lm_list_list, labels):
        """Detecta el dedo índice apuntando hacia abajo."""
        for lm_list in lm_list_list:
            if not lm_list:
                continue

            wrist = lm_list[0]
            index_mcp = lm_list[5]
            index_tip = lm_list[8]

            pointing_down = (index_tip[1] > index_mcp[1]) or (index_tip[1] > wrist[1] + 20)
            dist_index = math.hypot(index_tip[0] - wrist[0], index_tip[1] - wrist[1])

            others_closed = True
            for f in range(2, 5):
                tip_other = lm_list[self.TIP_IDS[f]]
                dist_other = math.hypot(tip_other[0] - wrist[0], tip_other[1] - wrist[1])
                if dist_other > (dist_index * 0.75):
                    others_closed = False
                    break

            if pointing_down and others_closed:
                return True
        return False

    def is_pose_shush(self, lm_list_list, labels):
        """Detecta el dedo índice apuntando en dirección vertical (seña de silencio)."""
        for lm_list in lm_list_list:
            if not lm_list:
                continue

            index_mcp = lm_list[5]
            index_tip = lm_list[8]
            index_extended = self._is_finger_extended_dist(lm_list, 1)
            pointing_up = (index_mcp[1] - index_tip[1]) > 20

            dx = abs(index_tip[0] - index_mcp[0])
            dy = abs(index_tip[1] - index_mcp[1])
            is_vertical = dx < (dy * 0.5)
            others_closed = not any(self._is_finger_extended_dist(lm_list, f) for f in [2, 3, 4])

            if index_extended and pointing_up and is_vertical and others_closed:
                return True
        return False

    def is_pose_think(self, lm_list_list, labels):
        """Detecta el dedo índice inclinado hacia un costado."""
        for lm_list in lm_list_list:
            if not lm_list:
                continue

            index_mcp = lm_list[5]
            index_tip = lm_list[8]
            index_extended = self._is_finger_extended_dist(lm_list, 1)

            dx = abs(index_tip[0] - index_mcp[0])
            dy = abs(index_tip[1] - index_mcp[1])
            is_inclined = dx >= (dy * 0.5)
            others_closed = not any(self._is_finger_extended_dist(lm_list, f) for f in [2, 3, 4])

            if index_extended and is_inclined and others_closed:
                return True
        return False

    def is_pose_v_sign(self, lm_list_list, labels):
        """Detecta la señal de victoria o 'V' (índice y medio extendidos)."""
        for lm_list in lm_list_list:
            if not lm_list:
                continue
            idx_ext = self._is_finger_extended_dist(lm_list, 1)
            mid_ext = self._is_finger_extended_dist(lm_list, 2)
            ring_ext = self._is_finger_extended_dist(lm_list, 3)
            pinky_ext = self._is_finger_extended_dist(lm_list, 4)

            if idx_ext and mid_ext and not ring_ext and not pinky_ext:
                return True
        return False

    def is_pose_fist(self, lm_list_list, labels):
        """Detecta un puño cerrado estándar."""
        for lm_list in lm_list_list:
            if not lm_list:
                continue
            extended_count = sum(1 for f in range(1, 5) if self._is_finger_extended_dist(lm_list, f))
            if extended_count == 0:
                return True
        return False

    def is_pose_palm(self, lm_list_list, labels):
        """Detecta una palma abierta (tres o más dedos extendidos)."""
        for lm_list in lm_list_list:
            if not lm_list:
                continue
            extended_count = sum(1 for f in range(1, 5) if self._is_finger_extended_dist(lm_list, f))
            if extended_count >= 3:
                return True
        return False

    def close(self):
        """Libera los recursos asignados a MediaPipe."""
        self.hands.close()


# =============================================================================
# 3. FUNCIONES AUXILIARES DE IMAGEN Y SUPERPOSICIÓN
# =============================================================================

def load_pose_images_unicode(paths_dict):
    """
    Carga imágenes desde el disco garantizando compatibilidad con caracteres UTF-8
    y extensiones alternativas (.jpg, .png, .jpeg).
    """
    loaded_images = {}
    print("Cargando imágenes de referencia...")
    for pose_id, path in paths_dict.items():
        final_path = path
        if not os.path.exists(final_path):
            base, _ = os.path.splitext(path)
            for ext in [".jpeg", ".png", ".JPG", ".PNG"]:
                if os.path.exists(base + ext):
                    final_path = base + ext
                    break

        try:
            img_array = np.fromfile(final_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is not None:
                loaded_images[pose_id] = img
                print(f"  [OK] Cargada pose '{pose_id}' desde: {final_path}")
            else:
                print(f"  [ERROR] No se pudo decodificar el archivo: {final_path}")
        except Exception as e:
            print(f"  [ERROR] Excepción al leer {final_path}: {e}")
            
    return loaded_images

def overlay_image(bg_img, overlay_img, margin_x=OVERLAY_MARGIN_X, margin_y=OVERLAY_MARGIN_Y, scale=OVERLAY_SCALE):
    """Superpone una imagen de referencia sobre el frame de la cámara redimensionándola proporcionalmente."""
    h_bg, w_bg, _ = bg_img.shape
    h_ov, w_ov, _ = overlay_img.shape
    
    target_w = int(w_bg * scale)
    ratio = target_w / w_ov
    target_h = int(h_ov * ratio)
    
    resized_ov = cv2.resize(overlay_img, (target_w, target_h))
    
    end_x = min(margin_x + target_w, w_bg)
    end_y = min(margin_y + target_h, h_bg)
    actual_w = end_x - margin_x
    actual_h = end_y - margin_y
    
    bg_img[margin_y:end_y, margin_x:end_x] = resized_ov[:actual_h, :actual_w]
    return bg_img


# =============================================================================
# 4. BUCLE PRINCIPAL DE CAPTURA Y PROCESAMIENTO
# =============================================================================

def main():
    images = load_pose_images_unicode(POSE_IMAGES)
    if not images:
        print("Error crítico: No se encontraron imágenes válidas. Verifica la carpeta de recursos.")
        return

    detector = HandPoseDetector()
    cap = cv2.VideoCapture(0)

    # Configuración de resolución de la videocámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara web.")
        return

    cv2.namedWindow("Reconocimiento de Poses de Mano", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Reconocimiento de Poses de Mano", FRAME_WIDTH, FRAME_HEIGHT)

    print("\n[INFO] Sistema iniciado. Muestra gestos a la cámara. Presiona 'q' para salir.\n")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            # Efecto espejo para mayor comodidad en la interacción
            frame = cv2.flip(frame, 1)
            h_frame, w_frame, _ = frame.shape
            frame = detector.find_hands(frame, draw=True)
            
            lm_list_list = []
            labels = []
            if detector.results.multi_hand_landmarks:
                for i in range(len(detector.results.multi_hand_landmarks)):
                    lm_list_list.append(detector.get_landmarks_list(frame, i))
                    labels.append(detector.get_hand_label(i))

            detected_pose = None
            
            # ORDEN DE EVALUACIÓN: Prioridad para poses complejas/compuestas antes de gestos genéricos
            if detector.is_pose_heart(lm_list_list, labels):
                detected_pose = "heart_sign"
            elif detector.is_pose_raised_fist(lm_list_list, labels, h_frame):
                detected_pose = "raised_fist_high"
            elif detector.is_pose_down_index(lm_list_list, labels):
                detected_pose = "point_down"
            elif detector.is_pose_shush(lm_list_list, labels):
                detected_pose = "shush_sign"
            elif detector.is_pose_think(lm_list_list, labels):
                detected_pose = "think_sign"
            elif detector.is_pose_v_sign(lm_list_list, labels):
                detected_pose = "v_sign"
            elif detector.is_pose_fist(lm_list_list, labels):
                detected_pose = "fist_standard"
            elif detector.is_pose_palm(lm_list_list, labels):
                detected_pose = "palm_sign"

            # Renderizado de imagen emergente y texto descriptivo
            if detected_pose and detected_pose in images:
                frame = overlay_image(frame, images[detected_pose])
                cv2.putText(frame, f"Pose activa: {detected_pose}", (20, frame.shape[0] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Reconocimiento de Poses de Mano", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

if __name__ == "__main__":
    main()