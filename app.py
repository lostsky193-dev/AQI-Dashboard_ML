from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Secret key needed for sessions, plus allow cross-origin for WebSockets
socketio = SocketIO(app, cors_allowed_origins="*")

# 1. When you visit the site, show the dashboard
@app.route('/')
def index():
    return render_template('index.html')

# 2. When the ESP32 sends data, broadcast it to the dashboard
@app.route('/api/upload', methods=['POST'])
def upload_data():
    data = request.json
    print("Received from ESP32:", data)
    
    # Instantly push 'live_data' to any open web browsers
    socketio.emit('live_data', data) 
    
    return jsonify({"status": "success", "message": "Data broadcasted"})

if __name__ == '__main__':
    # Run the server on port 5000
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)