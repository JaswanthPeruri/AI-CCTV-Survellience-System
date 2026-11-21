import cv2
import numpy as np
import time
from flask import Flask, render_template_string, Response, redirect, url_for
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

# Global alert variables
last_alert_time = 0
alert_interval = 10
current_alert = None

# Initialize HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_abnormal_activity(frame, prev_frame):
    if prev_frame is None:
        return False, False, False, ""

    people, _ = hog.detectMultiScale(frame, winStride=(8, 8))
    if len(people) == 0:
        return False, False, False, ""

    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    movement = np.sum(thresh) / 255

    theft = movement > 1000
    violence = 500 < movement <= 1000
    harassment = 300 < movement <= 500

    message = ""
    if theft:
        message = "🚨 Theft Detected!"
    elif violence:
        message = "⚠️ Violence Detected!"
    elif harassment:
        message = "⚠️ Harassment Detected!"

    return theft, violence, harassment, message

def generate_frames():
    global last_alert_time, current_alert
    cap = cv2.VideoCapture(0)
    prev_frame = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        theft, violence, harassment, message = detect_abnormal_activity(frame, prev_frame)
        current_time = time.time()

        if (theft or violence or harassment):
            if (current_time - last_alert_time > alert_interval) or (message != current_alert):
                socketio.emit('new_alert', {'message': message})
                last_alert_time = current_time
                current_alert = message

        if theft:
            cv2.putText(frame, "THEFT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        if violence:
            cv2.putText(frame, "VIOLENCE", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
        if harassment:
            cv2.putText(frame, "HARASSMENT", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        prev_frame = frame.copy()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Surveillance System</title>
        <style>
            body {
                font-family: 'Segoe UI', sans-serif;
                margin: 0;
                padding: 0;
                background: #f5f5f5;
            }
            .container {
                max-width: 800px;
                margin: auto;
                padding: 30px;
                background: white;
                border-radius: 10px;
                box-shadow: 0 0 15px rgba(0,0,0,0.2);
                text-align: center;
            }
            h1 {
                color: #333;
                margin-bottom: 20px;
            }
            ul {
                text-align: left;
                font-size: 18px;
                color: #444;
                margin-bottom: 30px;
            }
            li {
                margin: 10px 0;
            }
            .btn {
                background-color: #007BFF;
                color: white;
                padding: 12px 24px;
                font-size: 18px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                text-decoration: none;
                display: inline-block;
            }
            .btn:hover {
                background-color: #0056b3;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AI Surveillance System</h1>
            <ul>
                <li>🔍 Detects theft, violence, and harassment</li>
                <li>📸 Real-time camera feed processing</li>
                <li>💡 Sends alerts with detected abnormal behavior</li>
                <li>🧠 Uses AI-based motion and human detection</li>
                <li>⏱ Alerts every 10 seconds during continuous detection</li>
            </ul>
            <a class="btn" href="{{ url_for('surveillance') }}">🚀 Launch Camera Feed</a>
        </div>
    </body>
    </html>
    """)

@app.route('/surveillance')
def surveillance():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Live Camera Feed</title>
        <style>
            body {
                font-family: 'Segoe UI', sans-serif;
                background: #f4f4f4;
                margin: 0;
                padding: 0;
            }
            .container {
                padding: 30px;
                max-width: 900px;
                margin: auto;
                background: white;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.2);
            }
            h1 {
                text-align: center;
                color: #222;
            }
            #videoFeed {
                display: block;
                margin: auto;
                border-radius: 8px;
                border: 4px solid #444;
            }
            #alerts {
                margin-top: 20px;
                padding: 10px;
                max-height: 200px;
                overflow-y: scroll;
                background: #fff3f3;
                border-left: 5px solid red;
            }
            .alert {
                padding: 10px;
                margin-bottom: 8px;
                border-bottom: 1px solid #ddd;
                font-weight: bold;
                color: #b00020;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Live Camera Feed</h1>
            <img id="videoFeed" src="{{ url_for('video_feed') }}" width="800" height="480">
            <h2>Live Alerts</h2>
            <div id="alerts"></div>
        </div>

        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
        <script>
            const socket = io();
            const alertsDiv = document.getElementById('alerts');
            socket.on('new_alert', function(data) {
                const alertBox = document.createElement('div');
                alertBox.className = 'alert';
                alertBox.innerText = `${new Date().toLocaleTimeString()}: ${data.message}`;
                alertsDiv.prepend(alertBox);

                document.title = "🚨 ALERT - " + data.message;
                setTimeout(() => document.title = "Live Camera Feed", 5000);
            });
        </script>
    </body>
    </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
