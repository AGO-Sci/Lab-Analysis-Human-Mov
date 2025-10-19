"""Herramienta interactiva para extraer trazos de colores en dibujos infantiles.

El script está pensado para ejecutarse directamente en Spyder. Solo debes
actualizar las variables del apartado "Parámetros de usuario" o dejar que se
abra un cuadro de diálogo para elegir la imagen cuando lances el archivo."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class ColorSegment:
    """Pequeño contenedor para almacenar la información de cada color."""

    label: int
    mask: np.ndarray
    image: np.ndarray


class PointCollector:
    """Registra 4 clics del usuario sobre una ventana de OpenCV."""

    def __init__(self, window_name: str, image: np.ndarray) -> None:
        self.window_name = window_name
        self.image = image.copy()
        self.points: List[Tuple[int, int]] = []
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)

    def _mouse_callback(self, event: int, x: int, y: int, *_args) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            cv2.circle(self.image, (x, y), 6, (0, 255, 0), -1)
            cv2.imshow(self.window_name, self.image)

    def collect(self) -> List[Tuple[int, int]]:
        """Bloquea la ejecución hasta recibir 4 puntos."""
        print("Haz clic en las 4 esquinas de la cartulina (sentido horario o antihorario).")
        print("Presiona 'q' para cancelar.")
        cv2.imshow(self.window_name, self.image)
        while len(self.points) < 4:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise RuntimeError("Selección cancelada por el usuario")
        cv2.destroyWindow(self.window_name)
        return self.points


def order_points(pts: List[Tuple[int, int]]) -> np.ndarray:
    """Ordena los puntos en el orden esperado por la transformación de perspectiva."""

    pts_np = np.array(pts, dtype="float32")
    s = pts_np.sum(axis=1)
    diff = np.diff(pts_np, axis=1)

    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts_np[np.argmin(s)]  # superior izquierda
    ordered[2] = pts_np[np.argmax(s)]  # inferior derecha
    ordered[1] = pts_np[np.argmin(diff)]  # superior derecha
    ordered[3] = pts_np[np.argmax(diff)]  # inferior izquierda
    return ordered


def four_point_transform(image: np.ndarray, pts: List[Tuple[int, int]]) -> np.ndarray:
    """Recorta la región seleccionada usando una transformación de perspectiva."""

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def preprocess_for_segmentation(image: np.ndarray) -> np.ndarray:
    """Aplica un preprocesado ligero para mejorar la segmentación."""

    # Reducción de ruido y homogeneización de color.
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # Convertimos a espacio de color LAB (útil para separar colores perceptualmente).
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    return lab


def segment_colors(image: np.ndarray, n_colors: int = 4, min_area: int = 500) -> List[ColorSegment]:
    """Segmenta los colores usando K-Means y devuelve cada color como máscara separada."""

    lab = preprocess_for_segmentation(image)
    pixel_values = lab.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    compactness, labels, centers = cv2.kmeans(
        pixel_values,
        n_colors,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.reshape(image.shape[:2])

    segments: List[ColorSegment] = []
    for label in range(n_colors):
        mask = np.uint8(labels == label) * 255
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # Eliminamos pequeños ruidos.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                cv2.drawContours(clean_mask, [contour], -1, 255, -1)

        colored = cv2.bitwise_and(image, image, mask=clean_mask)
        if np.count_nonzero(clean_mask) == 0:
            continue
        segments.append(ColorSegment(label=label, mask=clean_mask, image=colored))
    print(f"Segmentación completada. Compactness: {compactness:.2f}. Colores detectados: {len(segments)}")
    return segments


def save_segments(segments: List[ColorSegment], output_dir: str) -> None:
    """Guarda las imágenes segmentadas en el disco."""

    os.makedirs(output_dir, exist_ok=True)
    for segment in segments:
        output_path = os.path.join(output_dir, f"color_{segment.label}.png")
        cv2.imwrite(output_path, segment.image)
        print(f"Color {segment.label} guardado en {output_path}")


def interactive_pipeline(image_path: str, n_colors: int, min_area: int, output_dir: str) -> None:
    """Orquesta el flujo completo desde la imagen original hasta los trazos individuales."""

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

    collector = PointCollector("Selecciona 4 puntos", image)
    pts = collector.collect()
    cropped = four_point_transform(image, pts)
    segments = segment_colors(cropped, n_colors=n_colors, min_area=min_area)
    save_segments(segments, output_dir)

    # Vista previa de los resultados.
    cv2.namedWindow("Recorte", cv2.WINDOW_NORMAL)
    cv2.imshow("Recorte", cropped)
    for segment in segments:
        cv2.namedWindow(f"Color {segment.label}", cv2.WINDOW_NORMAL)
        cv2.imshow(f"Color {segment.label}", segment.image)
    print("Presiona cualquier tecla sobre una ventana para cerrar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _choose_image_with_dialog() -> str | None:
    """Devuelve la ruta a una imagen elegida mediante un cuadro de diálogo (Spyder-friendly)."""

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:  # pragma: no cover - tkinter no siempre está disponible
        return None

    root = tk.Tk()
    root.withdraw()
    root.update()
    filename = filedialog.askopenfilename(
        title="Selecciona la imagen del dibujo",
        filetypes=[
            ("Imágenes", "*.jpg *.jpeg *.png *.bmp"),
            ("Todos los archivos", "*.*"),
        ],
    )
    root.destroy()
    return filename or None


def main() -> None:
    """Entrada pensada para ejecutar el script desde Spyder."""

    # ---------------------------------------------------------------------
    # Parámetros de usuario
    # ---------------------------------------------------------------------
    image_path = IMAGE_PATH
    n_colors = N_COLORS
    min_area = MIN_AREA
    output_dir = OUTPUT_DIR
    # ---------------------------------------------------------------------

    if not image_path:
        print("No se definió IMAGE_PATH. Abriendo diálogo para seleccionar la imagen...")
        image_path = _choose_image_with_dialog()
        if not image_path:
            raise RuntimeError(
                "No se seleccionó ninguna imagen. Ajusta IMAGE_PATH o elige un archivo en el diálogo."
            )

    print(f"Procesando: {image_path}")
    interactive_pipeline(image_path, n_colors=n_colors, min_area=min_area, output_dir=output_dir)


# -------------------------------------------------------------------------
# Valores predeterminados para facilitar su edición en Spyder
# -------------------------------------------------------------------------
IMAGE_PATH = ""  # Escribe aquí la ruta a tu fotografía si prefieres evitar el diálogo.
N_COLORS = 4
MIN_AREA = 500
OUTPUT_DIR = "segmentos"


if __name__ == "__main__":
    main()
