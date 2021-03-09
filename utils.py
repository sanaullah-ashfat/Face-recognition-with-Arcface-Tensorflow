import cv2


def draw_box_name(bbox, keypoints, name, frame):
    frame = cv2.line(frame, (bbox[0], bbox[1]), (bbox[0], bbox[1] + int(0.3 * bbox[3])), (0, 255, 255), 3)
    frame = cv2.line(frame, (bbox[0], bbox[1]), (bbox[0] + int(0.3 * bbox[2]), bbox[1]), (0, 255, 255), 3)

    frame = cv2.line(frame, (bbox[0] + bbox[2], bbox[1]), (bbox[0] + bbox[2], bbox[1] + int(0.3 * bbox[3])),
                     (0, 255, 255), 3)
    frame = cv2.line(frame, (bbox[0] + bbox[2], bbox[1]), (bbox[0] + bbox[2] - int(0.3 * bbox[2]), bbox[1]),
                     (0, 255, 255), 3)

    frame = cv2.line(frame, (bbox[0], bbox[1] + bbox[3]), (bbox[0], bbox[1] + bbox[3] - int(0.3 * bbox[3])),
                     (0, 255, 255), 3)
    frame = cv2.line(frame, (bbox[0], bbox[1] + bbox[3]), (bbox[0] + int(0.3 * bbox[2]), bbox[1] + bbox[3]),
                     (0, 255, 255), 3)

    frame = cv2.line(frame, (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                     (bbox[0] + bbox[2], bbox[1] + bbox[3] - int(0.3 * bbox[3])),
                     (0, 255, 255), 3)
    frame = cv2.line(frame, (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                     (bbox[0] + bbox[2] - int(0.3 * bbox[2]), bbox[1] + bbox[3]),
                     (0, 255, 255), 3)

    # for point in keypoints:
    #     cv2.circle(frame, tuple(point), 2, (0, 255, 255), 3)

    if bbox[1] > 100:

        frame = cv2.putText(frame,
                            name ,
                            (bbox[0], bbox[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA)
    else:
        frame = cv2.putText(frame,
                            name,
                            (bbox[0], bbox[1] + bbox[3] + 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 255, 255),
                            3,
                            cv2.LINE_AA)
    return frame
