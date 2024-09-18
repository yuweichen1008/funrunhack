import os
import cv2
import numpy as np
import mss
import pyautogui
import time
import threading
import queue  # For passing images between threads
from pynput import keyboard
from datetime import datetime

# Path to store screenshots
base_dir = "screenshots"

class ScreenshotAutomation:
    def __init__(self):
        self.center_x = 0
        self.center_y = 0
        self.img = None
        self.playbutton = cv2.imread('play.png', cv2.IMREAD_GRAYSCALE)
        self.contbutton = cv2.imread('continue.png', cv2.IMREAD_GRAYSCALE)
        self.ingame = cv2.imread('up.png', cv2.IMREAD_GRAYSCALE)
        self.league = cv2.imread('league.png', cv2.IMREAD_GRAYSCALE)
        self.w, self.h = self.playbutton.shape[::-1]
        self.low_canny = 100
        self.high_canny = 200
        self.key_actions = {
            'w': 'up',
            's': 'down',
            'q': 'drag_left',
            'a': 'use_tool',
            'd': 'dash',
            'e': 'bet'
        }

        # Queue for sharing images between threads
        self.image_queue = queue.Queue()

        # Ensure base directory exists
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # Start the automation process in a separate thread
        automate_thread = threading.Thread(target=self.automate, daemon=True)
        automate_thread.start()

        # Start listening for keyboard inputs in the main thread
        self.listen_for_keyboard()

    def find_pattern(self):
        """Locate the template pattern on the screen."""
        with mss.mss() as sct:
            screenshot = np.array(sct.grab(sct.monitors[1]))
            threshold = 0.8
            img_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

            result = cv2.matchTemplate(img_gray, self.ingame, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)
            if len(locations[0]) != 0:
                return False

            result = cv2.matchTemplate(img_gray, self.playbutton, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)

            if self.center_x == 0 and self.center_y == 0:
                for point in zip(*locations[::-1]):
                    self.center_x = (point[0] + self.w // 2) // 2
                    self.center_y = (point[1] + self.h // 2) // 2
                    self.move_mouse(self.center_x, self.center_y)
                    self.img = screenshot
                    print(f"Pattern found at ({self.center_x}, {self.center_y})")
                    return True  # Pattern found
            if len(locations[0]) != 0:
                print(f'Find Play')
                return True
            
            # search for continue button
            result = cv2.matchTemplate(img_gray, self.contbutton, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)
            if len(locations[0]) != 0:
                self.move_mouse(self.center_x, self.center_y)
                print(f'Find Continue')
                return True

        return False  # Pattern not found

    def automate(self):
        """Keep searching for the pattern and click it whenever it's found."""
        print("Detecting and clicking on 'play.png'...")
        while True:
            if self.find_pattern():
                # Click on the detected pattern
                self.click_on_play()
            else:
                print("Pattern not found, retrying...")
            time.sleep(2)  # Sleep to avoid excessive CPU usage

    def listen_for_keyboard(self):
        """Listen for keyboard inputs and perform actions."""
        print("Listening for keyboard inputs...")
        with keyboard.Listener(on_press=self.on_press) as listener:
            # Also process the image queue in the main thread
            while listener.running:
                try:
                    img = self.image_queue.get_nowait()  # Get the image if available
                    # self.display_image(img)
                except queue.Empty:
                    pass

                time.sleep(0.01)

    def display_image(self, img):
        """Display the image in a window in the main thread."""
        cv2.imshow("AI Peak", img)
        cv2.waitKey(1000)  # Display for 1 second
        cv2.destroyAllWindows()

    def click_on_play(self):
        """Click on the 'play' button once it is found."""
        # Move the mouse and click at the detected coordinates
        if self.center_x != 0 and self.center_y != 0:
            self.move_mouse(self.center_x, self.center_y)
            print(f"Clicked on 'play' at ({self.center_x}, {self.center_y})")

    def save_screenshot(self, action_folder):
        """Save a screenshot with a limited area and apply grayscale + Canny edge detection."""
        folder_path = os.path.join(base_dir, action_folder)
        os.makedirs(folder_path, exist_ok=True)

        # Create a unique filename based on timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{action_folder}_{timestamp}.png"
        filepath = os.path.join(folder_path, filename)

        # Define screenshot area (limit to region around center)
        monitor = {"top": self.center_y - 264, "left": self.center_x - 560, "width": 650, "height": 285}
        
        # Capture the screenshot in the defined area
        with mss.mss() as sct:
            img = np.array(sct.grab(monitor))
        
        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray_img, self.low_canny, self.high_canny)

        # Add the edges to the image queue for display in the main thread
        self.image_queue.put(edges)

        # Save the processed image
        cv2.imwrite(filepath, edges)
        print(f"Screenshot with Canny({self.low_canny}/{self.high_canny}) edges saved to: {filepath}")

    def move_mouse(self, new_x, new_y):
        """Move the mouse to the specified coordinates."""
        pyautogui.click(new_x, new_y, duration=0.05)
        # print(f"Moved mouse to ({new_x}, {new_y})")

    def on_press(self, key):
        """Handle key press events and take appropriate actions."""
        try:
            key_char = key.char
            if key_char in self.key_actions:
                # Calculate new mouse coordinates based on the key action
                new_x, new_y = self.calculate_new_position(key_char)

                if self.center_x != 0 and self.center_y != 0:
                    # Save screenshot and move the mouse
                    self.save_screenshot(self.key_actions[key_char])
                    self.move_mouse(new_x, new_y)
                    if key_char == 'q':
                        pyautogui.drag(-30, 0, 2, button='left')
                else:
                    print("Pattern not found, saving screenshot of the current screen...")
                    self.capture_screenshot(self.key_actions[key_char])
            # Handle the special keys using str(key) for keys like =, -, [, ]
            elif key_char == '=':
                self.low_canny += 5
            elif key_char == '-':
                self.low_canny -= 5
            elif key_char == ']':
                self.high_canny += 5
            elif key_char == '[':
                self.high_canny -= 5
        except AttributeError:
            # Handle any other unexpected key issues gracefully
            print(f"Special key pressed or an unhandled exception occurred: {key}")
            

    def calculate_new_position(self, key_char):
        """Calculate the new mouse position based on the key pressed."""
        if key_char == 'w':  # up
            return self.center_x + 40, self.center_y - 15
        elif key_char == 's':  # down
            return self.center_x - 40, self.center_y + 15
        elif key_char == 'q':  # drag left
            return self.center_x - 450, self.center_y + 5
        elif key_char == 'a':  # use tool
            return self.center_x - 450, self.center_y + 5
        elif key_char == 'e':  # bet
            return self.center_x - 430, self.center_y - 50
        elif key_char == 'd':  # dash
            return self.center_x - 420, self.center_y + 10
        return self.center_x // 2, self.center_y // 2  # Default to current center

    def capture_screenshot(self, action_folder):
        """Capture and save a screenshot regardless of pattern detection."""
        with mss.mss() as sct:
            self.img = np.array(sct.grab(sct.monitor[0]))
            self.save_screenshot(action_folder)

# Instantiate the class and start listening for key presses
automation = ScreenshotAutomation()

# Set up the keyboard listener
# with keyboard.Listener(on_press=automation.on_press) as listener:
#     listener.join()
