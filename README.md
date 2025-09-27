# Robot Simulator with AI Maze Processing

A comprehensive robot simulation system featuring WebSocket-based communication, Flask REST API, AI-powered maze processing with color filtering, and real-time visual simulation. Perfect for robotics research, pathfinding algorithms, and AI navigation testing.

## 🌟 Features

### Core Simulation
- **Real-time robot movement** with visual feedback and collision detection
- **WebSocket-based communication** for instant command processing
- **RESTful API** for robot control and system integration
- **Goal-reaching detection** with visual notifications
- **Canvas capture functionality** with base64 image export
- **Server-side state tracking** for robust operation

### AI Maze Processing
- **Advanced color filtering** using HSV color space
- **Robot and goal exclusion** from obstacle detection
- **Multi-threshold analysis** for optimal obstacle detection
- **Morphological processing** for noise reduction
- **Gray obstacle detection** with color-aware filtering
- **Automated maze analysis** from captured images

### Visual Interface
- **Interactive HTML5 canvas** with responsive design
- **Real-time position tracking** and status updates
- **Clean, modern UI** with gradient styling
- **Connection status monitoring** and automatic reconnection
- **Collision counter** and goal status display

## 🏗️ System Architecture
┌─────────────────┐ WebSocket ┌──────────────────┐
│ Robot Simulator│ ←──────────────→ │ Python Server │
└─────────────────┘ └──────────────────┘
↑
HTTP REST API
│
┌─────────────────┐ │
│ Maze Processor │ ←───────────────────────┘
└─────────────────┘

## 📋 Prerequisites

- **Python 3.7+**
- **Modern web browser** (Chrome, Firefox, Safari, Edge)
- **Required Python packages:**

## 🚀 Quick Start

### 1. Start the Python Server
- python server.py
- - WebSocket Server: `ws://localhost:8080`
- Flask API Server: `http://localhost:5001`

### 2. Open the Robot Simulator
- Open `simulator.html` in your web browser
- The simulator will automatically connect to the WebSocket server
- Watch the connection status in the info panel

### 3. Test the System
Basic movement test
curl -X POST http://localhost:5001/move
-H "Content-Type: application/json"
-d '{"x": 400, "y": 250}'

Capture current canvas state
curl http://localhost:5001/capture

Set random obstacles
curl -X POST http://localhost:5001/obstacles/random
-H "Content-Type: application/json"
-d '{"count": 10}'

## 📁 File Structure

robot-simulator/
├── server.py # Python backend server
├── simulator.html # Visual robot simulator interface
├── enhanced_maze_processor.py # AI maze processing with color filtering
└── README.md # This documentation

## 🎮 API Reference

### Movement Commands

| Endpoint | Method | Description | Example Payload |
|----------|--------|-------------|-----------------|
| `/move` | POST | Move robot to absolute position | `{"x": 325, "y": 300}` |
| `/move_rel` | POST | Move robot relative to current position | `{"angle": 45, "distance": 80}` |
| `/stop` | POST | Stop robot movement | `{}` |

### Goal Management

| Endpoint | Method | Description | Example Payload |
|----------|--------|-------------|-----------------|
| `/goal` | POST | Set goal position | `{"x": 550, "y": 80}` or `{"corner": "NE"}` |
| `/goal/status` | GET | Check if goal is reached | N/A |

### Obstacle Management

| Endpoint | Method | Description | Example Payload |
|----------|--------|-------------|-----------------|
| `/obstacles/random` | POST | Generate random obstacles | `{"count": 8}` |
| `/obstacles/positions` | POST | Set custom obstacle positions | `{"obstacles": [...]}` |

### System Operations

| Endpoint | Method | Description | Returns |
|----------|--------|-------------|---------|
| `/capture` | GET | Capture canvas as base64 image | Canvas image data with metadata |
| `/status` | GET | Get system status | Connection count, collisions, goal status |
| `/reset` | POST | Reset entire system | Confirmation message |
| `/collisions` | GET | Get collision count | Current collision count |

## 🤖 AI Maze Processing Usage

### Basic Maze Processing
from enhanced_maze_processor import ColorFilteredMazeProcessor

Initialize processor
processor = ColorFilteredMazeProcessor()

Load and process maze image
if processor.load_maze_image("maze.png"):
obstacles = processor.get_maze_obstacles()
maze_info = processor.get_maze_info()
print(f"Detected {len(obstacles)} obstacles")

# Save processed visualization
processor.save_processed_maze("output_maze.png")

### Color Filtering Configuration
The processor automatically excludes:
- Red robots (HSV ranges: 0-10, 160-179)
- Green goals (HSV range: 40-80)
- Focuses on gray obstacles (low saturation objects)
Customizable detection parameters:
Config.MIN_OBSTACLE_SIZE = 15 # Minimum obstacle size
Config.MAX_OBSTACLE_SIZE = 60 # Maximum obstacle size
Config.MIN_CONTOUR_AREA = 200 # Minimum contour area
Config.MAZE_THRESHOLDS = # Multi-threshold analysis

## 🎨 Canvas Specifications

- **Dimensions**: 650×600 pixels
- **Robot Size**: 18px radius (red circle with direction indicator)
- **Goal Size**: 15px radius (green flag)
- **Obstacle Size**: 25px default (black squares with gradient)
- **Detection Range**: ~33px (robot + goal radius)

## 📡 WebSocket Messages

### Outgoing Commands (Client → Server)
{"command": "move", "target": {"x": 325, "y": 300}}
{"command": "move_relative", "angle": 45, "distance": 80}
{"command": "stop"}
{"command": "set_goal", "position": {"x": 550, "y": 80}}
{"command": "set_obstacles", "obstacles": [...]}
{"command": "capture_canvas"}
{"command": "reset"}


### Incoming Events (Server → Client)
{"type": "collision", "collision": true, "robot_position": {...}}
{"type": "goal_reached", "robot_position": {...}, "goal_position": {...}}
{"type": "canvas_captured", "image_data": "base64...", "timestamp": "..."}
{"type": "connection", "message": "2D Robot simulator connected"}

## 🔧 Configuration

### Server Configuration
server.py configuration
CANVAS_WIDTH = 650
CANVAS_HEIGHT = 600
WEBSOCKET_PORT = 8080
FLASK_PORT = 5001

### Maze Processor Configuration
enhanced_maze_processor.py configuration
Config.BASE_URL = "http://localhost:5001"
Config.CANVAS_WIDTH = 650
Config.CANVAS_HEIGHT = 600
Config.ROBOT_RADIUS = 18
Config.GOAL_RADIUS = 15
Config.OBSTACLE_SIZE = 25

## 🎯 Corner Positions

| Corner | Coordinates | Usage |
|--------|-------------|-------|
| NW (Top-Left) | (20, 20) | `{"corner": "NW"}` |
| NE (Top-Right) | (630, 20) | `{"corner": "NE"}` |
| SW (Bottom-Left) | (20, 580) | `{"corner": "SW"}` |
| SE (Bottom-Right) | (630, 580) | `{"corner": "SE"}` |

## 📊 Usage Examples

### Canvas Capture with Full Metadata
curl http://localhost:5001/capture

**Response:**
{
"status": "success",
"image_data": "data:image/png;base64,iVBORw0KGgo...",
"timestamp": "2024-01-01T12:00:00.000Z",
"canvas_size": {"width": 650, "height": 600},
"robot_position": {"x": 320, "y": 300, "angle": 45},
"goal_position": {"x": 550, "y": 80},
"obstacles_count": 8,
"message": "Canvas image captured successfully"
}

### Maze Processing Integration
import requests
from enhanced_maze_processor import ColorFilteredMazeProcessor

Capture current canvas
response = requests.get("http://localhost:5001/capture")
if response.status_code == 200:
image_data = response.json()["image_data"]
# Process with AI maze processor
processor = ColorFilteredMazeProcessor()
# Convert base64 to image and process...
# Process with AI maze processor
processor = ColorFilteredMazeProcessor()
# Convert base64 to image and process...

## 🛠️ Troubleshooting

### Connection Issues
- **WebSocket**: Ensure server is running on port 8080
- **Flask API**: Verify server is running on port 5001
- **Browser Console**: Check for CORS or connection errors
- **Firewall**: Allow connections on ports 8080 and 5001

### Canvas Capture Issues
- Ensure WebSocket connection is active
- Check browser console for capture_canvas messages
- Verify simulator handles canvas capture commands properly
- Try manual reconnection using the UI button

### Maze Processing Issues
- **Image Format**: Ensure images are in PNG/JPG format
- **Color Detection**: Verify robot (red) and goal (green) are visible
- **Obstacle Detection**: Check if obstacles have sufficient contrast
- **File Paths**: Use absolute paths for image loading

## 🚀 Advanced Features

### Multi-Threshold Analysis
The maze processor uses multiple threshold values to optimize obstacle detection across different lighting conditions and image qualities.

### Color-Aware Processing
Advanced HSV color space filtering automatically excludes robot and goal colors from obstacle detection, ensuring accurate maze analysis.

### Real-Time Integration
Seamless integration between canvas capture, maze processing, and robot control enables live AI navigation testing.

## 📈 Performance Optimization

- **Asynchronous Processing**: WebSocket operations don't block API calls
- **Efficient Drawing**: Canvas operations optimized for 60fps rendering
- **Memory Management**: Automatic cleanup of WebSocket connections
- **Image Processing**: Optimized OpenCV operations for real-time analysis

## 🔮 Future Enhancements

- Path planning algorithm integration
- Multiple robot support
- Advanced maze generation
- Machine learning model integration
- Real-time path optimization
- 3D visualization support

## 📄 License

This project is open source. Feel free to use, modify, and distribute according to your needs.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest new features.

---

**Ready to start your robotics AI journey?** 🤖✨

Run `python server.py`, open `simulator.html`, and begin exploring the exciting world of AI-powered robot navigation!

