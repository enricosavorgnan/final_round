from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List
from datetime import datetime, date
from openai import OpenAI
import decimal
import json

app = FastAPI()

client = OpenAI(api_key='sk-proj-TVRJgZLveipYXfH6ufQZT3BlbkFJslnLIAYVY6vh6LEeO22e')

app.mount("/static", StaticFiles(directory="static"), name="static")

def detect_and_count_people(image):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    results = model(image)
    detections = results.pandas().xyxy[0]

    # Filtra le persone
    people_detections = detections[detections['name'] == 'person']

    # Disegna i bounding box e conta le persone
    for _, detection in people_detections.iterrows():
        xmin, ymin, xmax, ymax = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(image, 'Person', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return image, len(people_detections)

def number_people(image_path):
    # Carica l'immagine
    image = cv2.imread(image_path)

    # Rileva e conta le persone nell'immagine
    image, people_count = detect_and_count_people(image)

    # Mostra l'immagine con le rilevazioni
    cv2.putText(image, f'People Count: {people_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('YOLOv5 People Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # (Opzionale) Salva l'immagine con le rilevazioni
    cv2.imwrite('img/room2.jpg', image)

    return people_count


def get_transcription(my_audio):
    with open("speech.wav", "wb") as f:
        f.write(my_audio)
        speech = open("speech.wav", "rb")
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=speech,
            response_format="text"
        )
        return transcription

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        if data == "startRecording":
            await websocket.send_text("Recording started")
        elif data == "endRecording":
            audio_data = await websocket.receive_bytes()
            transcription = get_transcription(data)
        elif data.startswith("showDetectedImage"):
            image_path = data.split(":", 1)[1]
            people_count = number_people(image_path)
            await websocket.send_text('img/room2.jpg')
        elif

        await websocket.send_text(transcription)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse("templates/index.html")