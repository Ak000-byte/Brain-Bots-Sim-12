
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

# Enhanced Configuration with color filtering
class Config:
    """Configuration with color filtering for robot and goal"""
    BASE_URL = "http://localhost:5001"
    REQUEST_TIMEOUT = 3.0
    MAX_RETRIES = 2

    # Canvas Configuration
    CANVAS_WIDTH = 650
    CANVAS_HEIGHT = 600
    ROBOT_RADIUS = 18
    GOAL_RADIUS = 15
    OBSTACLE_SIZE = 25
    DETECTION_RANGE = 33

    # Navigation Configuration
    GOAL_TOLERANCE = 15
    MOVEMENT_DELAY = 0.1
    SAFETY_MARGIN = 40

    # Enhanced Maze Processing with Color Filtering
    MAZE_RESIZE_WIDTH = 650
    MAZE_RESIZE_HEIGHT = 600

    # Obstacle detection parameters
    MAZE_THRESHOLDS = [80, 100, 127, 150, 180]
    MIN_OBSTACLE_SIZE = 15
    MAX_OBSTACLE_SIZE = 60
    MIN_CONTOUR_AREA = 200
    MAX_CONTOUR_AREA = 4000

    # Color filtering parameters (HSV color space)
    # Red robot color ranges (in HSV)
    RED_LOWER1 = np.array([0, 100, 100])    # Lower red range
    RED_UPPER1 = np.array([10, 255, 255])
    RED_LOWER2 = np.array([160, 100, 100])  # Upper red range  
    RED_UPPER2 = np.array([179, 255, 255])

    # Green goal color range (in HSV)
    GREEN_LOWER = np.array([40, 100, 100])
    GREEN_UPPER = np.array([80, 255, 255])

    # Gray obstacle color range (what we WANT to detect)
    GRAY_LOWER = np.array([0, 0, 30])     # Dark gray
    GRAY_UPPER = np.array([179, 30, 120]) # Light gray (low saturation)

    # Corner coordinates
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

@dataclass 
class Obstacle:
    x: float
    y: float
    size: float = 25

class ColorFilteredMazeProcessor:
    """Enhanced maze processor with color filtering to exclude robot and goal"""

    def __init__(self):
        self.current_maze = None
        self.processed_obstacles = []
        self.color_masks = {}  # Store color masks for debugging
        self.maze_info = {
            'loaded': False,
            'filename': None,
            'dimensions': (0, 0),
            'obstacle_count': 0,
            'processing_method': None,
            'excluded_colors': []
        }

    def create_color_exclusion_mask(self, image: np.ndarray) -> np.ndarray:
        """Create a mask that excludes red robot and green goal"""
        # Convert to HSV color space for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create masks for red robot (two ranges for red in HSV)
        red_mask1 = cv2.inRange(hsv, Config.RED_LOWER1, Config.RED_UPPER1)
        red_mask2 = cv2.inRange(hsv, Config.RED_LOWER2, Config.RED_UPPER2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Create mask for green goal
        green_mask = cv2.inRange(hsv, Config.GREEN_LOWER, Config.GREEN_UPPER)

        # Expand the masks to exclude surrounding areas
        kernel = np.ones((15, 15), np.uint8)  # Larger kernel for safety
        red_mask = cv2.dilate(red_mask, kernel, iterations=2)
        green_mask = cv2.dilate(green_mask, kernel, iterations=2)

        # Combine exclusion masks
        exclusion_mask = cv2.bitwise_or(red_mask, green_mask)

        # Create final mask (areas to INCLUDE for obstacle detection)
        inclusion_mask = cv2.bitwise_not(exclusion_mask)

        # Store for debugging
        self.color_masks = {
            'red_mask': red_mask,
            'green_mask': green_mask,
            'exclusion_mask': exclusion_mask,
            'inclusion_mask': inclusion_mask
        }

        # Count excluded pixels for reporting
        red_pixels = np.sum(red_mask > 0)
        green_pixels = np.sum(green_mask > 0)

        print(f"🎨 Color Filtering Applied:")
        print(f"├── Red robot pixels excluded: {red_pixels}")
        print(f"├── Green goal pixels excluded: {green_pixels}")
        print(f"└── Total excluded area: {red_pixels + green_pixels} pixels")

        return inclusion_mask

    def detect_obstacles_with_color_filtering(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect obstacles while excluding robot and goal colors"""

        print("🔍 Enhanced Detection with Color Filtering")

        # Create color exclusion mask
        inclusion_mask = self.create_color_exclusion_mask(image)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        best_obstacles = []
        best_count = 0
        best_threshold = 127

        # Try multiple thresholds
        for threshold_val in Config.MAZE_THRESHOLDS:
            # Apply threshold (THRESH_BINARY_INV for dark objects)
            _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)

            # Apply color filtering mask
            filtered_binary = cv2.bitwise_and(binary, inclusion_mask)

            # Morphological operations
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

                        # Additional check: ensure center is not in exclusion zone
                        if inclusion_mask[center_y, center_x] > 0:
                            size = max(w, h)

                            current_obstacles.append({
                                'x': float(center_x),
                                'y': float(center_y),
                                'size': float(min(size, Config.MAX_OBSTACLE_SIZE)),
                                'area': area,
                                'method': f'color_filtered_threshold_{threshold_val}'
                            })

            print(f"  ├── Threshold {threshold_val}: {len(current_obstacles)} obstacles (color filtered)")

            if len(current_obstacles) > best_count:
                best_count = len(current_obstacles)
                best_obstacles = current_obstacles
                best_threshold = threshold_val

        print(f"  └── Best: threshold {best_threshold} with {best_count} color-filtered obstacles")
        return best_obstacles

    def detect_obstacles_gray_color_range(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect gray obstacles specifically, excluding colored objects"""

        print("🔍 Gray Color Range Detection")

        # Convert to HSV for better color filtering
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create mask for gray/dark objects (low saturation)
        gray_mask = cv2.inRange(hsv, Config.GRAY_LOWER, Config.GRAY_UPPER)

        # Create exclusion mask for colored objects
        inclusion_mask = self.create_color_exclusion_mask(image)

        # Combine masks
        final_mask = cv2.bitwise_and(gray_mask, inclusion_mask)

        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        obstacles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if Config.MIN_CONTOUR_AREA <= area <= Config.MAX_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(contour)

                if (Config.MIN_OBSTACLE_SIZE <= w <= Config.MAX_OBSTACLE_SIZE and 
                    Config.MIN_OBSTACLE_SIZE <= h <= Config.MAX_OBSTACLE_SIZE):

                    center_x = x + w // 2
                    center_y = y + h // 2
                    size = max(w, h)

                    obstacles.append({
                        'x': float(center_x),
                        'y': float(center_y),
                        'size': float(min(size, Config.MAX_OBSTACLE_SIZE)),
                        'area': area,
                        'method': 'gray_color_range'
                    })

        print(f"  └── Gray color detection: {len(obstacles)} obstacles")
        return obstacles

    def merge_and_filter_obstacles(self, all_obstacles: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Merge obstacles and remove duplicates"""
        print("🔄 Merging and filtering color-aware obstacles...")

        # Flatten all obstacles
        merged = []
        for method_obstacles in all_obstacles:
            merged.extend(method_obstacles)

        if not merged:
            return []

        # Remove duplicates
        filtered = []
        merge_distance = 30

        for obs in merged:
            is_duplicate = False
            for existing in filtered:
                distance = math.sqrt((obs['x'] - existing['x'])**2 + (obs['y'] - existing['y'])**2)
                if distance < merge_distance:
                    if obs.get('area', 0) > existing.get('area', 0):
                        filtered.remove(existing)
                        filtered.append(obs)
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered.append(obs)

        print(f"├── Total detected: {len(merged)}")
        print(f"├── After deduplication: {len(filtered)}")

        # Sort by confidence (area)
        filtered.sort(key=lambda x: x.get('area', 0), reverse=True)

        return filtered

    def analyze_image_colors(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image colors to detect robot and goal presence"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Check for red robot
        red_mask1 = cv2.inRange(hsv, Config.RED_LOWER1, Config.RED_UPPER1)
        red_mask2 = cv2.inRange(hsv, Config.RED_LOWER2, Config.RED_UPPER2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_pixels = np.sum(red_mask > 0)

        # Check for green goal
        green_mask = cv2.inRange(hsv, Config.GREEN_LOWER, Config.GREEN_UPPER)
        green_pixels = np.sum(green_mask > 0)

        # Overall image statistics
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)

        analysis = {
            'has_red_robot': red_pixels > 1000,  # Threshold for robot detection
            'has_green_goal': green_pixels > 500, # Threshold for goal detection
            'red_pixel_count': red_pixels,
            'green_pixel_count': green_pixels,
            'background_intensity': mean_intensity,
            'background_type': 'light' if mean_intensity > 127 else 'dark'
        }

        return analysis

    def load_maze_image(self, image_path: str) -> bool:
        """Load maze with advanced color filtering"""
        try:
            if not os.path.exists(image_path):
                print(f"❌ File not found: {image_path}")
                return False

            print(f"📸 Loading maze with color filtering: {image_path}")

            # Load and resize image
            pil_image = Image.open(image_path)
            original_size = pil_image.size
            print(f"📏 Original image size: {original_size[0]}x{original_size[1]}")

            pil_image = pil_image.resize((Config.MAZE_RESIZE_WIDTH, Config.MAZE_RESIZE_HEIGHT), Image.Resampling.LANCZOS)
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # Analyze colors in the image
            color_analysis = self.analyze_image_colors(opencv_image)

            print(f"🎨 Color Analysis Results:")
            print(f"├── Red robot detected: {'✅ YES' if color_analysis['has_red_robot'] else '❌ NO'}")
            print(f"├── Green goal detected: {'✅ YES' if color_analysis['has_green_goal'] else '❌ NO'}")
            print(f"├── Red pixels: {color_analysis['red_pixel_count']}")
            print(f"├── Green pixels: {color_analysis['green_pixel_count']}")
            print(f"└── Background: {color_analysis['background_type']} ({color_analysis['background_intensity']:.1f})")

            # Run obstacle detection with color filtering
            print("\n🔍 Running color-filtered obstacle detection...")

            method1_obstacles = self.detect_obstacles_with_color_filtering(opencv_image)
            method2_obstacles = self.detect_obstacles_gray_color_range(opencv_image)

            # Merge results
            all_methods = [method1_obstacles, method2_obstacles]
            final_obstacles = self.merge_and_filter_obstacles(all_methods)

            # Convert to server format
            server_obstacles = []
            for obs in final_obstacles:
                server_obstacles.append({
                    'x': obs['x'],
                    'y': obs['y'],
                    'size': obs['size']
                })

            # Store processed data
            self.current_maze = opencv_image
            self.processed_obstacles = server_obstacles

            excluded_colors = []
            if color_analysis['has_red_robot']:
                excluded_colors.append('red_robot')
            if color_analysis['has_green_goal']:
                excluded_colors.append('green_goal')

            self.maze_info = {
                'loaded': True,
                'filename': os.path.basename(image_path),
                'dimensions': (Config.MAZE_RESIZE_WIDTH, Config.MAZE_RESIZE_HEIGHT),
                'obstacle_count': len(server_obstacles),
                'processing_method': 'color_filtered_multi_method',
                'excluded_colors': excluded_colors
            }

            print(f"\n✅ Color-filtered maze processing completed!")
            print(f"├── Resized to: {Config.MAZE_RESIZE_WIDTH}x{Config.MAZE_RESIZE_HEIGHT}")
            print(f"├── Obstacles detected: {len(server_obstacles)} (color filtered)")
            print(f"├── Excluded colors: {', '.join(excluded_colors) if excluded_colors else 'None'}")
            print(f"└── Ready for navigation testing")

            if len(server_obstacles) > 0:
                print(f"\n📍 Detected obstacle positions (excluding robot/goal):")
                for i, obs in enumerate(server_obstacles[:8], 1):  # Show first 8
                    print(f"├── #{i}: ({obs['x']:.0f}, {obs['y']:.0f}) size {obs['size']:.0f}px")
                if len(server_obstacles) > 8:
                    print(f"└── ... and {len(server_obstacles) - 8} more obstacles")
            else:
                print("\n⚠️ No obstacles detected after color filtering")

            return True

        except Exception as e:
            print(f"❌ Error processing maze image: {e}")
            import traceback
            traceback.print_exc()
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
            'processing_method': None,
            'excluded_colors': []
        }
        print("🗑️ Color-filtered maze data cleared")

    def save_processed_maze(self, output_path: str = "color_filtered_maze_debug.png") -> bool:
        """Save processed maze with color filtering visualization"""
        if self.current_maze is None:
            print("❌ No maze loaded to save")
            return False

        try:
            # Create detailed visualization
            vis_image = self.current_maze.copy()

            # Draw detected obstacles in green
            for i, obs in enumerate(self.processed_obstacles):
                center = (int(obs['x']), int(obs['y']))
                radius = int(obs['size'] // 2)

                # Draw obstacle detection
                cv2.circle(vis_image, center, radius, (0, 255, 0), 2)  # Green circle
                cv2.circle(vis_image, center, 3, (0, 255, 255), -1)   # Yellow center

                # Add obstacle number
                cv2.putText(vis_image, str(i+1), (center[0]-10, center[1]-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Highlight excluded areas if available
            if 'exclusion_mask' in self.color_masks:
                # Show excluded areas in red overlay
                exclusion_overlay = cv2.bitwise_and(vis_image, vis_image, mask=self.color_masks['exclusion_mask'])
                exclusion_colored = cv2.applyColorMap(self.color_masks['exclusion_mask'], cv2.COLORMAP_HOT)
                vis_image = cv2.addWeighted(vis_image, 0.8, exclusion_colored, 0.2, 0)

            # Add legend
            cv2.putText(vis_image, "GREEN: Detected Obstacles", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_image, "RED: Excluded Robot/Goal", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imwrite(output_path, vis_image)
            print(f"💾 Color-filtered maze saved as: {output_path}")
            print(f"├── Obstacles marked: {len(self.processed_obstacles)}")
            print(f"├── Green circles: Detected obstacles")
            print(f"└── Red overlay: Excluded robot/goal areas")
            return True

        except Exception as e:
            print(f"❌ Error saving processed maze: {e}")
            return False

# Test function
def test_color_filtered_processor():
    processor = ColorFilteredMazeProcessor()

    print("🎨 Color-Filtered Maze Processor")
    print("=" * 50)
    print("🔧 Advanced Features:")
    print("├── HSV color space filtering")
    print("├── Red robot exclusion") 
    print("├── Green goal exclusion")
    print("├── Gray obstacle detection")
    print("├── Multi-threshold analysis")
    print("├── Morphological processing")
    print("└── Color-aware debug visualization")
    print("\n✅ Optimized for your canva_capture image!")

if __name__ == "__main__":
    test_color_filtered_processor()
