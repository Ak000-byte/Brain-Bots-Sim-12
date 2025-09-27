
import requests
import json
import time
import math
import random
import os
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image

# Configuration with advanced color filtering
class Config:
    BASE_URL = "http://localhost:5001"
    REQUEST_TIMEOUT = 3.0
    MAX_RETRIES = 2

    # Canvas Configuration
    CANVAS_WIDTH = 650
    CANVAS_HEIGHT = 600
    ROBOT_RADIUS = 18
    GOAL_RADIUS = 15
    OBSTACLE_SIZE = 25

    # Navigation Configuration
    GOAL_TOLERANCE = 15
    MOVEMENT_DELAY = 0.1
    SAFETY_MARGIN = 40

    # Enhanced Color-Filtered Maze Processing
    MAZE_RESIZE_WIDTH = 650
    MAZE_RESIZE_HEIGHT = 600
    MAZE_THRESHOLDS = [80, 100, 127, 150, 180]
    MIN_OBSTACLE_SIZE = 15
    MAX_OBSTACLE_SIZE = 60
    MIN_CONTOUR_AREA = 200
    MAX_CONTOUR_AREA = 4000

    # HSV Color Filtering Ranges
    # Red robot (circular, bright red)
    RED_LOWER1 = np.array([0, 100, 100])
    RED_UPPER1 = np.array([10, 255, 255])
    RED_LOWER2 = np.array([160, 100, 100])
    RED_UPPER2 = np.array([179, 255, 255])

    # Green goal (square, bright green)
    GREEN_LOWER = np.array([40, 100, 100])
    GREEN_UPPER = np.array([80, 255, 255])

    # Gray obstacles (what we want to detect)
    GRAY_LOWER = np.array([0, 0, 30])
    GRAY_UPPER = np.array([179, 30, 120])

    CORNER_COORDINATES = {
        'NE': (600, 50), 'NW': (50, 50), 
        'SE': (600, 550), 'SW': (50, 550)
    }

@dataclass
class Point:
    x: float
    y: float

    def distance_to(self, other: 'Point') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class ColorFilteredMazeProcessor:
    """Advanced maze processor with color filtering"""

    def __init__(self):
        self.current_maze = None
        self.processed_obstacles = []
        self.color_masks = {}
        self.maze_info = {
            'loaded': False,
            'filename': None,
            'dimensions': (0, 0),
            'obstacle_count': 0,
            'excluded_colors': []
        }

    def create_exclusion_mask(self, image: np.ndarray) -> np.ndarray:
        """Create mask excluding robot and goal colors"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red robot masks
        red_mask1 = cv2.inRange(hsv, Config.RED_LOWER1, Config.RED_UPPER1)
        red_mask2 = cv2.inRange(hsv, Config.RED_LOWER2, Config.RED_UPPER2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Green goal mask
        green_mask = cv2.inRange(hsv, Config.GREEN_LOWER, Config.GREEN_UPPER)

        # Expand masks for safety margin
        kernel = np.ones((15, 15), np.uint8)
        red_mask = cv2.dilate(red_mask, kernel, iterations=2)
        green_mask = cv2.dilate(green_mask, kernel, iterations=2)

        # Combine exclusion masks
        exclusion_mask = cv2.bitwise_or(red_mask, green_mask)
        inclusion_mask = cv2.bitwise_not(exclusion_mask)

        # Store for debugging
        self.color_masks = {
            'red_mask': red_mask,
            'green_mask': green_mask,
            'exclusion_mask': exclusion_mask,
            'inclusion_mask': inclusion_mask
        }

        red_pixels = np.sum(red_mask > 0)
        green_pixels = np.sum(green_mask > 0)

        print(f"🎨 Color Filtering:")
        print(f"├── Red robot excluded: {red_pixels} pixels")
        print(f"├── Green goal excluded: {green_pixels} pixels")
        print(f"└── Safe detection area: {np.sum(inclusion_mask > 0)} pixels")

        return inclusion_mask

    def detect_obstacles_filtered(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect obstacles with color filtering"""
        print("🔍 Color-Filtered Obstacle Detection")

        # Create exclusion mask
        inclusion_mask = self.create_exclusion_mask(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        best_obstacles = []
        best_count = 0
        best_threshold = 127

        for threshold_val in Config.MAZE_THRESHOLDS:
            # Threshold for dark objects
            _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)

            # Apply color filtering
            filtered_binary = cv2.bitwise_and(binary, inclusion_mask)

            # Clean up with morphological operations
            kernel = np.ones((3, 3), np.uint8)
            filtered_binary = cv2.morphologyEx(filtered_binary, cv2.MORPH_OPEN, kernel)
            filtered_binary = cv2.morphologyEx(filtered_binary, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(filtered_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            current_obstacles = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if Config.MIN_CONTOUR_AREA <= area <= Config.MAX_CONTOUR_AREA:
                    x, y, w, h = cv2.boundingRect(contour)

                    if (Config.MIN_OBSTACLE_SIZE <= w <= Config.MAX_OBSTACLE_SIZE and 
                        Config.MIN_OBSTACLE_SIZE <= h <= Config.MAX_OBSTACLE_SIZE):

                        center_x = x + w // 2
                        center_y = y + h // 2

                        # Final check: ensure not in exclusion zone
                        if inclusion_mask[center_y, center_x] > 0:
                            size = max(w, h)
                            current_obstacles.append({
                                'x': float(center_x),
                                'y': float(center_y),
                                'size': float(min(size, Config.MAX_OBSTACLE_SIZE)),
                                'area': area
                            })

            print(f"  ├── Threshold {threshold_val}: {len(current_obstacles)} obstacles")

            if len(current_obstacles) > best_count:
                best_count = len(current_obstacles)
                best_obstacles = current_obstacles
                best_threshold = threshold_val

        print(f"  └── Best result: {best_count} obstacles at threshold {best_threshold}")
        return best_obstacles

    def load_maze_image(self, image_path: str) -> bool:
        """Load maze with advanced color filtering"""
        try:
            if not os.path.exists(image_path):
                print(f"❌ File not found: {image_path}")
                return False

            print(f"📸 Loading maze with color filtering: {image_path}")

            # Load image
            pil_image = Image.open(image_path)
            original_size = pil_image.size
            print(f"📏 Original: {original_size[0]}x{original_size[1]}")

            pil_image = pil_image.resize((Config.MAZE_RESIZE_WIDTH, Config.MAZE_RESIZE_HEIGHT), Image.Resampling.LANCZOS)
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # Analyze colors
            hsv = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2HSV)
            red_pixels = np.sum(cv2.inRange(hsv, Config.RED_LOWER1, Config.RED_UPPER1) > 0) +                         np.sum(cv2.inRange(hsv, Config.RED_LOWER2, Config.RED_UPPER2) > 0)
            green_pixels = np.sum(cv2.inRange(hsv, Config.GREEN_LOWER, Config.GREEN_UPPER) > 0)

            excluded_colors = []
            if red_pixels > 1000:
                excluded_colors.append('red_robot')
            if green_pixels > 500:
                excluded_colors.append('green_goal')

            print(f"🎨 Detected colors:")
            print(f"├── Red robot: {'✅ YES' if 'red_robot' in excluded_colors else '❌ NO'} ({red_pixels} pixels)")
            print(f"└── Green goal: {'✅ YES' if 'green_goal' in excluded_colors else '❌ NO'} ({green_pixels} pixels)")

            # Detect obstacles with filtering
            obstacles = self.detect_obstacles_filtered(opencv_image)

            # Convert to server format
            server_obstacles = []
            for obs in obstacles:
                server_obstacles.append({
                    'x': obs['x'],
                    'y': obs['y'],
                    'size': obs['size']
                })

            # Store results
            self.current_maze = opencv_image
            self.processed_obstacles = server_obstacles
            self.maze_info = {
                'loaded': True,
                'filename': os.path.basename(image_path),
                'dimensions': (Config.MAZE_RESIZE_WIDTH, Config.MAZE_RESIZE_HEIGHT),
                'obstacle_count': len(server_obstacles),
                'excluded_colors': excluded_colors
            }

            print(f"\n✅ Color-filtered processing complete!")
            print(f"├── Obstacles detected: {len(server_obstacles)}")
            print(f"├── Excluded: {', '.join(excluded_colors) if excluded_colors else 'None'}")
            print(f"└── Ready for navigation!")

            if len(server_obstacles) > 0:
                print(f"\n📍 Gray obstacle positions:")
                for i, obs in enumerate(server_obstacles, 1):
                    print(f"├── #{i}: ({obs['x']:.0f}, {obs['y']:.0f}) size {obs['size']:.0f}px")
                print(f"└── Total: {len(server_obstacles)} gray obstacles detected")

            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    def get_maze_obstacles(self) -> List[Dict[str, Any]]:
        return self.processed_obstacles.copy()

    def get_maze_info(self) -> Dict[str, Any]:
        return self.maze_info.copy()

    def clear_maze(self):
        self.current_maze = None
        self.processed_obstacles = []
        self.color_masks = {}
        self.maze_info = {
            'loaded': False,
            'filename': None,
            'dimensions': (0, 0),
            'obstacle_count': 0,
            'excluded_colors': []
        }
        print("🗑️ Color-filtered maze cleared")

    def save_processed_maze(self, output_path: str = "color_filtered_debug.png") -> bool:
        """Save with color filtering visualization"""
        if self.current_maze is None:
            return False

        try:
            vis_image = self.current_maze.copy()

            # Draw detected obstacles in bright green
            for i, obs in enumerate(self.processed_obstacles):
                center = (int(obs['x']), int(obs['y']))
                radius = int(obs['size'] // 2)

                cv2.circle(vis_image, center, radius, (0, 255, 0), 3)  # Green
                cv2.circle(vis_image, center, 5, (0, 255, 255), -1)   # Yellow center
                cv2.putText(vis_image, str(i+1), (center[0]-10, center[1]-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show excluded areas
            if 'exclusion_mask' in self.color_masks:
                exclusion_colored = cv2.applyColorMap(self.color_masks['exclusion_mask'], cv2.COLORMAP_HOT)
                vis_image = cv2.addWeighted(vis_image, 0.85, exclusion_colored, 0.15, 0)

            # Legend
            cv2.putText(vis_image, "GREEN: Gray Obstacles Only", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis_image, f"EXCLUDED: Robot + Goal", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imwrite(output_path, vis_image)
            print(f"💾 Color-filtered debug saved: {output_path}")
            return True

        except Exception as e:
            print(f"❌ Save error: {e}")
            return False

class RobotController:
    """Robot controller with color-filtered maze integration"""

    def __init__(self, server_url: str = Config.BASE_URL):
        self.server_url = server_url
        self.maze_processor = ColorFilteredMazeProcessor()

        try:
            self.session = requests.Session()
        except:
            self.session = None

        self.performance_stats = {
            'total_navigations': 0,
            'successful_navigations': 0,
            'maze_tests': 0,
            'successful_maze_tests': 0,
            'collision_count': 0
        }

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[requests.Response]:
        url = f"{self.server_url}{endpoint}"

        for attempt in range(Config.MAX_RETRIES):
            try:
                response = (self.session or requests).request(
                    method, url, timeout=Config.REQUEST_TIMEOUT, **kwargs
                )
                if response.status_code == 200:
                    return response
                else:
                    print(f"⚠️ Server status {response.status_code}")
            except requests.exceptions.RequestException as e:
                if attempt < Config.MAX_RETRIES - 1:
                    time.sleep(0.2 * (attempt + 1))
                else:
                    print(f"❌ Request failed: {e}")
        return None

    # Core robot methods
    def get_robot_status(self) -> Dict[str, Any]:
        response = self._make_request("GET", "/status")
        return response.json() if response else {}

    def move_robot(self, x: float, y: float) -> bool:
        response = self._make_request("POST", "/move", json={"x": x, "y": y}, 
                                    headers={"Content-Type": "application/json"})
        return response is not None

    def set_goal(self, x: float, y: float) -> bool:
        response = self._make_request("POST", "/goal", json={"x": x, "y": y}, 
                                    headers={"Content-Type": "application/json"})
        if response:
            print(f"🎯 Goal: ({x:.1f}, {y:.1f})")
        return response is not None

    def is_goal_reached(self) -> bool:
        response = self._make_request("GET", "/goal/status")
        return response.json().get('goal_reached', False) if response else False

    def reset_simulation(self) -> bool:
        response = self._make_request("POST", "/reset")
        if response:
            print("✅ Reset complete")
        return response is not None

    def set_custom_obstacles(self, obstacles: List[Dict]) -> bool:
        response = self._make_request("POST", "/obstacles/positions", 
                                    json={"obstacles": obstacles},
                                    headers={"Content-Type": "application/json"})
        if response:
            print(f"🚧 Set {len(obstacles)} obstacles")
        return response is not None

    # Color-filtered maze methods
    def load_maze_from_image(self, image_path: str) -> bool:
        """Load maze excluding robot and goal colors"""
        success = self.maze_processor.load_maze_image(image_path)
        if success:
            obstacles = self.maze_processor.get_maze_obstacles()
            if obstacles:
                return self.set_custom_obstacles(obstacles)
            else:
                print("ℹ️ No gray obstacles found after color filtering")
                return True
        return success

    def test_maze_navigation(self, start_x: float = 50, start_y: float = 50,
                           goal_x: float = 600, goal_y: float = 550) -> bool:
        """Test navigation on color-filtered maze"""
        if not self.maze_processor.maze_info['loaded']:
            print("❌ No maze loaded")
            return False

        info = self.maze_processor.maze_info
        print(f"🗺️ Testing maze: {info['filename']}")
        print(f"📊 Gray obstacles: {info['obstacle_count']}")
        print(f"🎨 Excluded: {', '.join(info['excluded_colors'])}")

        self.performance_stats['maze_tests'] += 1

        if not self.set_goal(goal_x, goal_y) or not self.move_robot(goal_x, goal_y):
            return False

        # Wait for completion
        start_time = time.time()
        while time.time() - start_time < 15:
            if self.is_goal_reached():
                self.performance_stats['successful_maze_tests'] += 1
                duration = time.time() - start_time
                print(f"✅ Navigation complete in {duration:.2f}s!")
                return True
            time.sleep(0.2)

        print("❌ Navigation timeout")
        return False

    def get_maze_status(self) -> Dict[str, Any]:
        return self.maze_processor.get_maze_info()

    def clear_maze(self):
        self.maze_processor.clear_maze()
        self.reset_simulation()

    def print_performance_stats(self):
        nav_rate = (self.performance_stats['successful_navigations'] / 
                   max(1, self.performance_stats['total_navigations']) * 100)
        maze_rate = (self.performance_stats['successful_maze_tests'] / 
                    max(1, self.performance_stats['maze_tests']) * 100)

        print(f"""
📊 Color-Filtered Navigation Stats:
├── Standard Navigation: {self.performance_stats['successful_navigations']}/{self.performance_stats['total_navigations']} ({nav_rate:.1f}%)
├── Maze Navigation: {self.performance_stats['successful_maze_tests']}/{self.performance_stats['maze_tests']} ({maze_rate:.1f}%)
├── Current Maze: {self.maze_processor.maze_info.get('filename', 'None')}
└── Excluded Colors: {', '.join(self.maze_processor.maze_info.get('excluded_colors', []))}
""")

def get_coordinates(prompt: str, default_x: float = 320, default_y: float = 300) -> Tuple[float, float]:
    print(f"🎯 {prompt}")
    try:
        x_input = input(f"X (default {default_x}): ").strip()
        x = float(x_input) if x_input else default_x
        x = max(30, min(620, x))

        y_input = input(f"Y (default {default_y}): ").strip()
        y = float(y_input) if y_input else default_y
        y = max(30, min(570, y))

        return (x, y)
    except (ValueError, KeyboardInterrupt):
        return (default_x, default_y)

def get_image_path() -> Optional[str]:
    try:
        path = input("Maze image path: ").strip().strip('"').strip("'")
        if not path:
            return None
        if os.path.exists(path):
            return path
        else:
            print(f"❌ File not found: {path}")
            return None
    except KeyboardInterrupt:
        return None

def main():
    """Main with color-filtered maze processing"""
    print("=" * 70)
    print("🎨 COLOR-FILTERED ROBOT NAVIGATION SYSTEM")
    print("🤖 Excludes Robot & Goal | 🗺️ Detects Gray Obstacles Only")
    print("=" * 70)

    controller = RobotController()

    # Test connection
    print("🔍 Testing server...")
    status = controller.get_robot_status()
    if not status:
        print("❌ Server not found! Run: py -3.9 server.py")
        return

    print("✅ Server connected!")

    # Check libraries
    try:
        import cv2, numpy as np
        from PIL import Image
        print("✅ Color processing libraries ready")
    except ImportError as e:
        print(f"⚠️ Missing library: {e}")
        print("Install: pip install opencv-python pillow numpy")

    while True:
        print(f"""
🎮 Color-Filtered Navigation System:

🎨 COLOR-FILTERED MAZE PROCESSING:
1. 🖼️ Load Maze (Exclude Robot/Goal Colors)
2. 🗺️ Test Maze Navigation (Gray Obstacles Only)
3. 📊 Maze Status & Color Analysis
4. 🗑️ Clear Maze
5. 💾 Save Debug Visualization

🤖 ROBOT NAVIGATION:
6. ⚡ Quick Test
7. 🎯 Custom Navigation
8. 🔄 Reset System
9. 📈 Performance Stats
10. ❌ Exit

""")

        choice = input("Choose (1-10): ").strip()

        try:
            if choice == '1':
                print("🖼️ Color-Filtered Maze Loading")
                print("🎨 Will exclude: Red robot, Green goal")
                print("🔍 Will detect: Gray square obstacles only")

                image_path = get_image_path()
                if image_path:
                    if controller.load_maze_from_image(image_path):
                        info = controller.get_maze_status()
                        print(f"\n🎉 Success!")
                        print(f"├── File: {info['filename']}")
                        print(f"├── Gray obstacles: {info['obstacle_count']}")
                        print(f"└── Excluded: {', '.join(info['excluded_colors'])}")
                    else:
                        print("❌ Loading failed")

            elif choice == '2':
                print("🗺️ Color-Filtered Maze Navigation")
                info = controller.get_maze_status()

                if not info['loaded']:
                    print("❌ No maze loaded (use option 1)")
                else:
                    print(f"📋 Maze: {info['filename']}")
                    print(f"🎨 Excluded colors: {', '.join(info['excluded_colors'])}")
                    print(f"🚧 Gray obstacles: {info['obstacle_count']}")

                    start_x, start_y = get_coordinates("START position", 50, 50)
                    goal_x, goal_y = get_coordinates("GOAL position", 600, 550)

                    print(f"\n🚀 Testing: ({start_x:.0f},{start_y:.0f}) → ({goal_x:.0f},{goal_y:.0f})")
                    success = controller.test_maze_navigation(start_x, start_y, goal_x, goal_y)
                    print(f"Result: {'🎉 SUCCESS' if success else '❌ FAILED'}")

            elif choice == '3':
                print("📊 Color-Filtered Maze Status")
                info = controller.get_maze_status()

                if info['loaded']:
                    print("✅ Maze Information:")
                    print(f"├── File: {info['filename']}")
                    print(f"├── Dimensions: {info['dimensions'][0]}x{info['dimensions'][1]}")
                    print(f"├── Gray obstacles: {info['obstacle_count']}")
                    print(f"├── Excluded colors: {', '.join(info['excluded_colors']) if info['excluded_colors'] else 'None'}")
                    print(f"└── Status: {'Ready' if info['obstacle_count'] > 0 else 'No obstacles'}")

                    obstacles = controller.maze_processor.get_maze_obstacles()
                    if obstacles:
                        print(f"\n📍 Gray Obstacle Positions:")
                        for i, obs in enumerate(obstacles, 1):
                            print(f"├── #{i}: ({obs['x']:.0f}, {obs['y']:.0f}) size {obs['size']:.0f}")
                else:
                    print("❌ No maze loaded")

            elif choice == '4':
                print("🗑️ Clear Color-Filtered Maze")
                controller.clear_maze()
                print("✅ Maze cleared, system reset")

            elif choice == '5':
                print("💾 Save Color-Filtered Debug")
                info = controller.get_maze_status()

                if info['loaded']:
                    filename = input("Filename (default: color_filtered_debug.png): ").strip()
                    filename = filename or "color_filtered_debug.png"

                    if controller.maze_processor.save_processed_maze(filename):
                        print(f"✅ Saved: {filename}")
                        print("├── Green circles: Gray obstacles detected")
                        print("└── Red overlay: Excluded robot/goal areas")
                else:
                    print("❌ No maze to save")

            elif choice == '6':
                print("⚡ Quick Test")
                controller.reset_simulation()
                time.sleep(0.3)

                start_time = time.time()
                success = (controller.move_robot(550, 100) and 
                          controller.set_goal(550, 100))
                if success:
                    # Wait for completion
                    for _ in range(50):  # 10 seconds max
                        if controller.is_goal_reached():
                            break
                        time.sleep(0.2)
                    success = controller.is_goal_reached()

                duration = time.time() - start_time
                print(f"Result: {'✅ SUCCESS' if success else '❌ FAILED'} ({duration:.2f}s)")

            elif choice == '7':
                print("🎯 Custom Navigation")
                start_x, start_y = get_coordinates("START", 100, 100)
                goal_x, goal_y = get_coordinates("GOAL", 550, 500)

                print(f"\n🚀 Navigating: ({start_x:.0f},{start_y:.0f}) → ({goal_x:.0f},{goal_y:.0f})")

                start_time = time.time()
                success = (controller.move_robot(goal_x, goal_y) and 
                          controller.set_goal(goal_x, goal_y))
                if success:
                    for _ in range(75):  # 15 seconds max
                        if controller.is_goal_reached():
                            break
                        time.sleep(0.2)
                    success = controller.is_goal_reached()

                duration = time.time() - start_time
                print(f"Result: {'✅ SUCCESS' if success else '❌ FAILED'} ({duration:.2f}s)")

            elif choice == '8':
                if controller.reset_simulation():
                    print("✅ System reset complete")

            elif choice == '9':
                controller.print_performance_stats()

            elif choice == '10':
                print("👋 Thank you for using Color-Filtered Navigation!")
                break

            else:
                print("❌ Invalid choice (1-10)")

        except KeyboardInterrupt:
            print("\n⚠️ Cancelled")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
