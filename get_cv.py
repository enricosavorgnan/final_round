import cv2
import torch

# Upload the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


def detect_and_count_people(image):
    # ------------------------------------------------------------------- #
    # This function is used to detect and count the number of people in   #
    # the image.                                                          #
    # input: image - the image to be processed                            #
    # output: image - the image with the detections                       #
    #         len(people_detections) - the number of people detected       #
    # ------------------------------------------------------------------- #

    # Detect objects in the image
    results = model(image)
    detections = results.pandas().xyxy[0]

    # >Filters people
    people_detections = detections[detections['name'] == 'person']

    # Draw bounding boxes and count people
    for _, detection in people_detections.iterrows():
        xmin, ymin, xmax, ymax = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(image, 'Person', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return image, len(people_detections)


def main():
    # ------------------------------------------------------------------- #
    # This function is used to get the number of people in the image.     #
    # output: people_count - the number of people in the image            #
    # ------------------------------------------------------------------- #

    # Load the image
    image_path = 'media/room.jpg'
    image = cv2.imread(image_path)

    # Detect and count people in the image
    image, people_count = detect_and_count_people(image)

    # Show the image with the detections
    cv2.putText(image, f'People Count: {people_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('YOLOv5 People Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('media/room_detected.jpg', image)

    return people_count
