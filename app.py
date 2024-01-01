import time
import cv2
from flask import Flask, render_template, Response,request
import mediapipe as mp
import math
import demo_singlepose
import numpy as np
import v2_openpose

app = Flask(__name__, static_folder='statics', static_url_path='/statics')
app.add_url_rule('/statics/<path:filename>',
                 endpoint='statics', view_func=app.send_static_file)
app.secret_key = "the secret key"

@app.route('/', methods=['GET'])
def index():

    return render_template('index.html')

@app.route('/movenet', methods=['GET'])
def movenet():

    return render_template('movenet.html')

@app.route('/posenet', methods=['GET'])
def posenet():
    return render_template('posenet.html')

@app.route('/upload', methods=['GET'])
def uploadPage():
    return render_template("upload.html")


@app.route('/upload', methods=['POST'])
def uploadVideo():
    if 'file' not in request.files:
        return render_template("upload.html", label="No file selected")
    print(request.files)
    file = request.files['file']
    if file.filename == '':
        return render_template("upload.html", label="No file selected")

    video_path = "upload_video/video.mp4"
    file.save(video_path)

    api = request.form.get('api') 

    if api == 'mediapipe':
        gen = gen_video()
        for _ in gen:
            pass
    
        img_path = "path/to/processed/image.jpg" 
    elif api == 'movenet':
        gen=generate_video_movenet()
        for _ in gen:
            pass
  
        img_path = "path/to/processed/image.jpg"  
    elif api == 'openpose':
 
        gen=v2_openpose.genOpenposeVideo()
        for _ in gen:
            pass
  
        img_path = "path/to/processed/image.jpg" 
    else:
      
        return render_template("upload.html", label="Invalid API selected")

    return render_template("upload.html", img_path=img_path)
    

@app.route('/pose_metrics')
def calculate_pose_metrics(pose_landmarks):

    pcp_value = 0.85
    pck_value = 0.75
    pdj_value = 0.90
    oks_map_value = 0.80

    return pcp_value, pck_value, pdj_value, oks_map_value


def calculate_angles(landmarks, connections):
    angles = []
    connections = [(11, 13), (13, 15)] 
    for connection in connections:
        joint1 = landmarks[connection[0]]
        joint2 = landmarks[connection[1]]
        if joint1.visibility < 0.5 or joint2.visibility < 0.5:
            angles.append(None)
            continue

        dx = joint2.x - joint1.x
        dy = joint2.y - joint1.y
        angle = math.degrees(math.atan2(dy, dx))
        angles.append(angle)
     
    return angles

def calculate_joint_angle_mediapipe(a, b, c):
    a_coords = np.array([a.x, a.y, a.z])  # First
    b_coords = np.array([b.x, b.y, b.z])  # Mid
    c_coords = np.array([c.x, c.y, c.z])  # End

    radians = np.arctan2(c_coords[1] - b_coords[1], c_coords[0] - b_coords[0]) - np.arctan2(a_coords[1] - b_coords[1], a_coords[0] - b_coords[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    print(angle)
    return angle

def calculate_angle_newnew(a, b, c):
    a_coords = np.array([a[0], a[1]])  # First
    b_coords = np.array([b[0], b[1]])  # Mid
    c_coords = np.array([c[0], c[1]])  # End

    radians = np.arctan2(c_coords[1] - b_coords[1], c_coords[0] - b_coords[0]) - np.arctan2(a_coords[1] - b_coords[1], a_coords[0] - b_coords[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    print(angle)
    return angle

def gen():
    previous_time = 0
    mpDraw = mp.solutions.drawing_utils
    my_pose = mp.solutions.pose
    pose = my_pose.Pose()
    connections = list(my_pose.POSE_CONNECTIONS)

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(imgRGB)

        if result.pose_landmarks:
            mpDraw.draw_landmarks(img, result.pose_landmarks, connections)
            pcp, pck, pdj, oks_map = calculate_pose_metrics(result.pose_landmarks)
    
            print(result.pose_landmarks.landmark[11])
            print(result.pose_landmarks.landmark[13])
            print(result.pose_landmarks.landmark[15])

            angles = calculate_joint_angle_mediapipe(result.pose_landmarks.landmark[11],result.pose_landmarks.landmark[13],result.pose_landmarks.landmark[15])
           
            angle_text = str(round(angles, 1))
            x = int(result.pose_landmarks.landmark[13].x * img.shape[1])
            y = int(result.pose_landmarks.landmark[13].y * img.shape[0])
            cv2.putText(img, angle_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # for angle in angles:
        
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20)
        if key == 27:
            break

def gen_video():
    mpDraw = mp.solutions.drawing_utils
    my_pose = mp.solutions.pose
    pose = my_pose.Pose()
    connections = list(my_pose.POSE_CONNECTIONS)
    previous_time = 0
    cap = cv2.VideoCapture('upload_video/video.mp4')

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
   
 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video/output.mp4', fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
       
        success, img = cap.read()
        if not success:
            break

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(imgRGB)

        if result.pose_landmarks:
            mpDraw.draw_landmarks(img, result.pose_landmarks, connections)
            pcp, pck, pdj, oks_map = calculate_pose_metrics(result.pose_landmarks)
            print(result.pose_landmarks.landmark[11])
            print(result.pose_landmarks.landmark[13])
            print(result.pose_landmarks.landmark[15])

            angles = calculate_joint_angle_mediapipe(result.pose_landmarks.landmark[11], result.pose_landmarks.landmark[13], result.pose_landmarks.landmark[15])

            angle_text = str(round(angles, 1))
            x = int(result.pose_landmarks.landmark[13].x * img.shape[1])
            y = int(result.pose_landmarks.landmark[13].y * img.shape[0])
            cv2.putText(img, angle_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        out.write(img)  # Write the processed frame to the output video file

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        key = cv2.waitKey(20)
        if key == 27:
            break

    cap.release()
    out.release()  # Release the output video writer
    cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



# =======================MoveNet================================

def generate_frames():
    elapsed_time = 0
    args = demo_singlepose.get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    connections = [(0, 1), (1, 2), (2, 3), (3, 7), (7, 6), (6, 5), (5, 4), (3, 8), (8, 9), (9, 10), (10, 11), (3, 12), (12, 13), (13, 14), (14, 15), (0, 16), (16, 17), (17, 18), (18, 19), (0, 20), (20, 21), (21, 22), (22, 23)]

    if args.file is not None:
        cap_device = args.file

    mirror = args.mirror
    model_select = args.model_select
    keypoint_score_th = args.keypoint_score

    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    if model_select == 0:
        model_path = "./saved_model.pb"
        input_size = 192
    elif model_select == 1:
        model_path = "./saved_model.pb"
        input_size = 256
    else:
        demo_singlepose.exit("*** model_select {} is an invalid value. Please use 0-1. ***".format(model_select))

    module = demo_singlepose.tf.saved_model.load("movenet")
    model = module.signatures['serving_default']

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

  
        keypoints, scores, visibilities = demo_singlepose.run_inference(model, input_size, frame)
       

        debug_frame = demo_singlepose.draw_debug(frame, elapsed_time, keypoint_score_th, keypoints, scores)

 
        ret, buffer = cv2.imencode('.jpg', debug_frame)
        frame = buffer.tobytes()


        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def generate_video_movenet():
    elapsed_time = 0
    args = demo_singlepose.get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    connections = [(0, 1), (1, 2), (2, 3), (3, 7), (7, 6), (6, 5), (5, 4), (3, 8), (8, 9), (9, 10), (10, 11), (3, 12), (12, 13), (13, 14), (14, 15), (0, 16), (16, 17), (17, 18), (18, 19), (0, 20), (20, 21), (21, 22), (22, 23)]

    if args.file is not None:
        cap_device = args.file

    mirror = args.mirror
    model_select = args.model_select
    keypoint_score_th = args.keypoint_score

    cap = cv2.VideoCapture('upload_video/video.mp4')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    if model_select == 0:
        model_path = "./saved_model.pb"
        input_size = 192
    elif model_select == 1:
        model_path = "./saved_model.pb"
        input_size = 256
    else:
        demo_singlepose.exit("*** model_select {} is an invalid value. Please use 0-1. ***".format(model_select))

    module = demo_singlepose.tf.saved_model.load("movenet")
    model = module.signatures['serving_default']

    output_file = 'output_video/outputMovenet.mp4'
    output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fps = cap.get(cv2.CAP_PROP_FPS)
    output_codec = cv2.VideoWriter_fourcc(*'mp4v')
    output_writer = cv2.VideoWriter(output_file, output_codec, output_fps, (output_width, output_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

    
        keypoints, scores, visibilities = demo_singlepose.run_inference(model, input_size, frame)
       

        debug_frame = demo_singlepose.draw_debug(frame, elapsed_time, keypoint_score_th, keypoints, scores)

       
        output_writer.write(debug_frame)

        ret, buffer = cv2.imencode('.jpg', debug_frame)
        frame = buffer.tobytes()


        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    output_writer.release()

@app.route('/m_video_feed')
def m_video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')






@app.route('/posenet_video_feed')
def posenet_video_feed():
    return Response(v2_openpose.genOpenpose(), mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == "__main__":
    app.run(debug=True)