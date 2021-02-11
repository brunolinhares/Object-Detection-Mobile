# biblioteca
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# argumentos
'''ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())'''

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# carrega o modelo
print("[INFO] CARREGANDO O MODELO...")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# inicializa o video
print("[INFO] CARREGANDO O VIDEO...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# coloca me loop o video
while True:
	# deixa a camera em loop e redimensiona o frame
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# pegue as dimensões do quadro e converta-o em uma "bolha"
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# passar o blob pela rede e obter as detecções e faz a predição
	net.setInput(blob)
	detections = net.forward()

	# deixa em loop as detecções
	for i in np.arange(0, detections.shape[2]):
		# extrai a confiança nas predições
		confidence = detections[0, 0, i, 2]

		# filtra as detecções fracas
		if confidence > 0.2:
			# extrai o indice da classe
			# pega as coordenadas
			# faz o bouding box
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# coloca o nome da classe
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# mostra o frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# encerra o loop
	if key == ord("q"):
		break

	# autaliza o contador de fps
	fps.update()

# encerrar o contador de fps
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# encerra o programa
cv2.destroyAllWindows()
vs.stop()