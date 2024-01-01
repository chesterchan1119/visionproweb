import cv2 as cv
import numpy as np
import argparse
import math
import app
def genOpenpose():
    parser = argparse.ArgumentParser()
    parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
    parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
    parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

    args = parser.parse_args()

    BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

    POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
    POSE_CONNECTIONS=[ ["LShoulder","LElbow","LWrist"]]

    inWidth = args.width
    inHeight = args.height

    net = cv.dnn.readNetFromTensorflow("graph_opt.pb")
    def calculate_angle(a, b, c):
        """Calculates the angle between three points."""
        ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
        return ang + 360 if ang < 0 else ang
    

    cap = cv.VideoCapture(0)
    print("Camera opneded")
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]

        assert(len(BODY_PARTS) == out.shape[1])

        points = []
        for i in range(len(BODY_PARTS)):
            heatMap = out[0, i, :, :]
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            points.append((int(x), int(y)) if conf > args.thr else None)

        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in BODY_PARTS)
            assert(partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]
          
            if points[idFrom] and points[idTo]:
                cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (255, 0, 0), cv.FILLED)
                cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (255, 0, 0), cv.FILLED)
            if points[5] and points[6] and points[7]:
                angle=app.calculate_angle_newnew(points[5],points[6],points[7])
                cv.putText(frame, str(round(angle, 1)), (points[6][0],points[6][1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv.destroyAllWindows()

def genOpenposeVideo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
    parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
    parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

    args = parser.parse_args()
    BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

    POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
    POSE_CONNECTIONS=[ ["LShoulder","LElbow","LWrist"]]
    inWidth = args.width
    inHeight = args.height

    net = cv.dnn.readNetFromTensorflow("graph_opt.pb")
    def calculate_angle(a, b, c):
        """Calculates the angle between three points."""
        ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
        print(ang)
        return ang + 360 if ang < 0 else ang

    cap = cv.VideoCapture('upload_video/video.mp4')  # Read input video from file
    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    output_path = 'output_video/outputOpenpose.mp4'
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    output = cv.VideoWriter(output_path, fourcc, 30, (frameWidth, frameHeight))

    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        # Process the frame...
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]

        assert(len(BODY_PARTS) == out.shape[1])

        points = []
        for i in range(len(BODY_PARTS)):
            heatMap = out[0, i, :, :]
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            points.append((int(x), int(y)) if conf > args.thr else None)

        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in BODY_PARTS)
            assert(partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]
          
            if points[idFrom] and points[idTo]:
                cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (255, 0, 0), cv.FILLED)
                cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (255, 0, 0), cv.FILLED)
            if points[5] and points[6] and points[7]:
                angle=app.calculate_angle_newnew(points[5],points[6],points[7])
                cv.putText(frame, str(round(angle, 1)), (points[6][0],points[6][1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        output.write(frame)  # Write the processed frame to the output video file

        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    output.release()
    cv.destroyAllWindows()    