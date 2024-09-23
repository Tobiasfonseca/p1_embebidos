import cv2
import numpy as np
import openvino as ov
from pathlib import Path

## DETECCIÓN DE OBJETOS CON OPENVINO
##### INICIALIZACIÓN DEL MODELO

base_model_dir = Path("model")
model_name = "ssdlite_mobilenet_v2"
precision = "FP16"
converted_model_path = Path("model") / f"{model_name}_{precision.lower()}.xml"

core = ov.Core()
model = core.read_model(model=converted_model_path)
compiled_model = core.compile_model(model=model)
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
height, width = list(input_layer.shape)[1:3]

# Clases de objetos
classes = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign",
    "parking meter", "bench", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear"
]

# Colores para las clases
colors = cv2.applyColorMap(
    src=np.arange(0, 255, 255 / len(classes), dtype=np.float32).astype(np.uint8),
    colormap=cv2.COLORMAP_RAINBOW,
).squeeze()

## FUNCIONES

def process_results(frame, results, thresh=0.4):
    h, w = frame.shape[:2]
    results = results.squeeze()
    boxes = []
    labels = []
    scores = []
    
    for _, label, score, xmin, ymin, xmax, ymax in results:
        if score > thresh:  # Solo contar si la confianza es alta
            boxes.append(
                tuple(map(int, (xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h)))
            )
            labels.append(int(label))
            scores.append(float(score))

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes, scores=scores, score_threshold=thresh, nms_threshold=0.4
    )

    if len(indices) == 0:
        return []

    return [(labels[idx], scores[idx], boxes[idx]) for idx in indices.flatten()]

def draw_boxes(frame, boxes):
    for label, score, box in boxes:
        color = tuple(map(int, colors[label]))
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        cv2.rectangle(img=frame, pt1=box[:2], pt2=(x2, y2), color=color, thickness=3)
        cv2.putText(
            img=frame,
            text=f"{classes[label]}",
            org=(box[0] + 10, box[1] + 30),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=frame.shape[1] / 1000,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return frame

def run_object_detection(frame, flip=False):
    try:
        if flip:
            frame = cv2.flip(frame, 1)

        scale = 1280 / max(frame.shape)
        if scale < 1:
            frame = cv2.resize(
                src=frame,
                dsize=None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )

        input_img = cv2.resize(
            src=frame, dsize=(width, height), interpolation=cv2.INTER_AREA
        )
        input_img = input_img[np.newaxis, ...]

        results = compiled_model([input_img])[output_layer]
        boxes = process_results(frame=frame, results=results)
        frame = draw_boxes(frame=frame, boxes=boxes)

        return frame

    except Exception as e:
        print("Error en el bloque de procesamiento 1:", e)
        return None

## MAIN
####### ENTRADA
USE_WEBCAM = False
video_file = input("Ingrese el video en formato mp4: ")
cam_id = 0
source = cam_id if USE_WEBCAM else video_file

cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print("Error al abrir la fuente de video.")
    exit()

######## SALIDA
output_path = "output.mp4"
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

## PIPELINE
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()

    if not ret:
        print("El video terminó.")
        break

    original_frame = frame.copy()
    
    processed_frame = run_object_detection(frame)

    if processed_frame is not None:
        cv2.imshow('Frame', processed_frame)
        out.write(processed_frame)
    else:
        cv2.imshow('Frame', original_frame)
        out.write(original_frame)

    if cv2.waitKey(int(1000 / 30)) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
