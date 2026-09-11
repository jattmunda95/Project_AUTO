import cv2

for camera_id in range(5):
    camera = cv2.VideoCapture(camera_id)

    if not camera.isOpened():
        camera.release()
        continue

    print(f"Showing camera ID {camera_id}. Press any key for the next camera.")

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        cv2.imshow(f"Camera ID {camera_id}", frame)
        if cv2.waitKey(30) != -1:
            break

    camera.release()
    cv2.destroyAllWindows()