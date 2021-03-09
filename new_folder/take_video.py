import cv2


def main(_argv):
    cap = cv2.VideoCapture(0)

    video_writer = cv2.VideoWriter('./recording.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (640, 480))
    while cap.isOpened():
        is_success, frame = cap.read()
        if is_success:
            cv2.imshow('face Capture', frame)
        key = cv2.waitKey(1) & 0xFF
        video_writer.write(frame)

        if key == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
