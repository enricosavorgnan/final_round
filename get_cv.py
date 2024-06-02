import cv2
import torch

# Load the model Yolov5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Function to detect and count people
def detect_and_count_people(image):
    results = model(image)
    detections = results.pandas().xyxy[0]

    # Filtering people 
    people_detections = detections[detections['name'] == 'person']

    # Drawing bounding boxes and counting people
    for _, detection in people_detections.iterrows():
        xmin, ymin, xmax, ymax = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(image, 'Person', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return image, len(people_detections)


def main():
    # Load the image
    image_path = 'img/room.jpg'
    image = cv2.imread(image_path)

    # Detects and counts people in the image
    image, people_count = detect_and_count_people(image)

    # Shows the image with detection
    cv2.putText(image, f'People Count: {people_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('YOLOv5 People Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    Saves image with detections 
    cv2.imwrite('img/room2.jpg', image)

    return people_count
