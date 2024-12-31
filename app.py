import sys
import os
import cv2
import csv
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QTimer
from ultralytics import YOLO
import yaml
import time
from datetime import datetime
from classifier import Classifier
from scipy.spatial import distance
import torch  # Import to check CUDA availability

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Player with Area Detection")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO("yolo11x.pt").to(self.device)
        self.classifier = Classifier(model_paths={"employee": "employee-best.pt"}, device=self.device)
        self.mobile_classifier = Classifier(model_paths={"mobile": "mobile-best.pt"}, device=self.device)
        self.video_path = None
        self.cap = None
        self.timer = QTimer()
        self.data_yaml = None
        self.colors = None
        self.start_time = None
        self.id_storage = {}
        self.csv_file = "stats.csv"
        self.next_id = 0
        self.employee_id_storage = {}  
        self.customer_id_storage = {}  
        self.employee_id_counter = 0  
        self.customer_id_counter = 0 
        self.customer_times = {}
        self.employee_times = {}
        self.area_time = {"Waiting area": {}, "Service area": {}}
        self.frame_rate = 30

        self.waiting_area = [(950, 0), (1600, 550)]

        self.initUI()

    def initUI(self):
        # UI Components
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(720, 480)  # Set a fixed size for the video screen

        open_button = QPushButton("Open Video", self)
        open_button.setStyleSheet("background-color: green; color: white;")
        infer_button = QPushButton("Run Inference", self)
        infer_button.setStyleSheet("background-color: blue; color: white;")

        open_button.clicked.connect(self.open_video)
        infer_button.clicked.connect(self.run_inference)

        button_layout = QHBoxLayout()
        button_layout.addWidget(open_button)
        button_layout.addWidget(infer_button)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addLayout(button_layout)

        self.container = QWidget()
        self.container.setLayout(layout)
        self.setCentralWidget(self.container)

    def open_video(self):
        self.video_path = "activity.mp4"  # Video path
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print("Error: Could not open video.")
                return

            # Read video frame rate
            self.frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS))

            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)

    def run_inference(self):
        if not self.cap or not self.cap.isOpened():
            print("Error: Video not opened. Please open a video first.")
            return

        with open("./data.yaml", 'r') as f:
            self.data_yaml = yaml.safe_load(f)

        self.colors = np.random.randint(0, 255, size=(len(self.data_yaml['names']), 3), dtype=int).tolist()

        self.start_time = time.time()
        self.timer.timeout.connect(self.process_next_frame)
        self.timer.start(int(1000 / self.frame_rate))  # Adjust timer interval based on video frame rate

    def process_next_frame(self):
        frame_start_time = time.time()

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            print("Video processing finished.")
            self.write_oncall_csv()  # Write on-call status
            self.write_area_csv()    # Write area time details
            return

        results = self.model(frame, verbose=False, classes=[0])
        final_boxes = []
        employee_centers = {}

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    final_boxes.append([x1, y1, x2, y2])

        if final_boxes:
            updated_id_storage = {}
            for (x1, y1, x2, y2) in final_boxes:
                bboxes = [(x1, y1, x2, y2)]
                try:
                    appearance = self.classifier.classify(bboxes, frame)[0][0]
                except Exception as e:
                    print(f"Error in appearance classification: {e}")
                    appearance = "unknown"

                try:
                    mobile_status = self.mobile_classifier.classify(bboxes, frame)[0][0]
                except Exception as e:
                    print(f"Error in mobile classification: {e}")
                    mobile_status = "unknown"

                person_type = "employee" if appearance == "employee" else "customer"
                x_center, y_center = self.calculate_center(x1, y1, x2, y2)
                obj_id = self.match_or_assign_id(x_center, y_center, person_type)

                if person_type == "employee":
                    employee_centers[obj_id] = (x_center, y_center)

                area = self.determine_area(x_center, y_center)

                # Update start time for employee and customer tracking
                if person_type == "customer":
                    if obj_id not in self.customer_times:
                        self.customer_times[obj_id] = int(time.time())  # Record the start time
                elif person_type == "employee":
                    if obj_id not in self.employee_times:
                        self.employee_times[obj_id] = int(time.time())  # Record the start time for employees

                # Track time spent in each area
                if obj_id not in self.area_time[area]:
                    self.area_time[area][obj_id] = {"start": time.time(), "total": 0}
                else:
                    self.area_time[area][obj_id]["total"] += time.time() - self.area_time[area][obj_id]["start"]
                    self.area_time[area][obj_id]["start"] = time.time()

                # Update on-call and no-call times
                if obj_id not in updated_id_storage:
                    start_time = self.id_storage[obj_id]["time"][0] if obj_id in self.id_storage else int(time.time())
                    updated_id_storage[obj_id] = {
                        'coords': [[x_center, y_center]],
                        "time": [start_time, start_time],
                        "label": person_type,
                        "oncall_time": 0,
                        "nocall_time": 0,
                        "last_oncall_start": None
                    }
                else:
                    updated_id_storage[obj_id]["coords"].append([x_center, y_center])
                    updated_id_storage[obj_id]["time"][1] = int(time.time())

                    if mobile_status == "on_call":
                        if updated_id_storage[obj_id]["last_oncall_start"] is None:
                            updated_id_storage[obj_id]["last_oncall_start"] = int(time.time())
                    else:
                        if updated_id_storage[obj_id]["last_oncall_start"] is not None:
                            start = updated_id_storage[obj_id]["last_oncall_start"]
                            updated_id_storage[obj_id]["oncall_time"] += int(time.time()) - start
                            updated_id_storage[obj_id]["last_oncall_start"] = None

                        # Track no-call time
                        updated_id_storage[obj_id]["nocall_time"] += int(time.time()) - updated_id_storage[obj_id]["time"][1]

                # Draw bounding box and log data
                self.draw_bounding_box(frame, x1, y1, x2, y2, person_type, obj_id, area)
                self.write_csv(person_type, obj_id, x_center, y_center, area)

            self.check_talking_employees(frame, employee_centers)

            # Log employee activity (active/inactive time) with proximity checks
            self.log_employee_activity()

            self.id_storage = updated_id_storage

        self.display_frame(frame)

        elapsed_time = time.time() - frame_start_time
        delay = max(int(1000 / self.frame_rate) - int(elapsed_time * 1000), 1)
        self.timer.start(delay)

    def log_employee_activity(
        self,
        inactivity_threshold=100,
        activity_check_interval=5,
        conversation_threshold=50,
        conversation_time=10
    ):
        """
        Analyze and log active/inactive time of employees.
        
        Args:
            inactivity_threshold: Minimum movement (in pixels) to consider an employee as active.
            activity_check_interval: Interval (in seconds) for checking activity.
            conversation_threshold: Proximity distance (in pixels) to consider employees close.
            conversation_time: Time (in seconds) for which close employees are marked inactive.
        """
        # Initialize or update the employee activity tracker
        if not hasattr(self, 'employee_activity'):
            self.employee_activity = {}

        # Open the CSV file for appending data
        csv_file = "employee_activity.csv"
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            
            if not file_exists:
                # Write the header if the file does not exist
                writer.writerow(["object_id", "status", "total_time"])

            # Track proximity timestamps
            proximity_timestamps = {}

            # Iterate through each employee in the ID storage
            for obj_id, data in self.id_storage.items():
                if data["label"] == "employee":
                    # Initialize employee activity tracking
                    if obj_id not in self.employee_activity:
                        self.employee_activity[obj_id] = {
                            "last_position": data["coords"][-1],
                            "last_activity_time": time.time(),
                            "status": "inactive",  # Initially inactive
                            "inactive_duration": 0,
                            "active_duration": 0,
                        }

                    current_center = data["coords"][-1]
                    last_position = self.employee_activity[obj_id]["last_position"]
                    last_activity_time = self.employee_activity[obj_id]["last_activity_time"]
                    status = self.employee_activity[obj_id]["status"]

                    # Calculate distance from last position
                    dist = distance.euclidean(current_center, last_position)

                    # Calculate elapsed time since last activity check
                    time_elapsed = time.time() - last_activity_time

                    # Track proximity between employees
                    for other_id, other_data in self.id_storage.items():
                        if other_id != obj_id and other_data["label"] == "employee":
                            other_center = other_data["coords"][-1]
                            proximity = distance.euclidean(current_center, other_center)

                            if proximity <= conversation_threshold:
                                if (obj_id, other_id) not in proximity_timestamps:
                                    proximity_timestamps[(obj_id, other_id)] = time.time()
                                elif time.time() - proximity_timestamps[(obj_id, other_id)] > conversation_time:
                                    # Mark both employees as inactive due to conversation
                                    self.employee_activity[obj_id]["status"] = "inactive"
                                    self.employee_activity[other_id]["status"] = "inactive"
                                    self.employee_activity[obj_id]["last_activity_time"] = time.time()
                                    self.employee_activity[other_id]["last_activity_time"] = time.time()
                            else:
                                if (obj_id, other_id) in proximity_timestamps:
                                    del proximity_timestamps[(obj_id, other_id)]

                    # Update active/inactive status
                    if dist > inactivity_threshold:  # Movement detected (active)
                        if status != "active":
                            self.employee_activity[obj_id]["status"] = "active"
                            self.employee_activity[obj_id]["last_activity_time"] = time.time()

                        self.employee_activity[obj_id]["active_duration"] += time_elapsed
                        self.employee_activity[obj_id]["inactive_duration"] = 0  # Reset inactive time

                    else:  # No significant movement (inactive)
                        if status != "inactive":
                            self.employee_activity[obj_id]["status"] = "inactive"
                            self.employee_activity[obj_id]["last_activity_time"] = time.time()

                        self.employee_activity[obj_id]["inactive_duration"] += time_elapsed
                        self.employee_activity[obj_id]["active_duration"] = 0  # Reset active time

                    # Log the active/inactive time at each check interval
                    if time_elapsed >= activity_check_interval:  # Log every 'activity_check_interval' seconds
                        if self.employee_activity[obj_id]["status"] == "active":
                            total_time = self.employee_activity[obj_id]["active_duration"]
                        else:
                            total_time = self.employee_activity[obj_id]["inactive_duration"]

                        minutes = int(total_time // 60)
                        seconds = int(total_time % 60)
                        time_formatted = f"{minutes:02d}:{seconds:02d}"

                        # Log the status and total time
                        writer.writerow([obj_id, self.employee_activity[obj_id]["status"], time_formatted])

                    # Update last position for the next check
                    self.employee_activity[obj_id]["last_position"] = current_center
                    self.employee_activity[obj_id]["last_activity_time"] = time.time()

    def check_talking_employees(self, frame, employee_centers, talking_threshold=100):
        """Check if employees are close enough to be considered talking."""
        timestamp = int(time.time())  # Record the current timestamp

        csv_file = "talking_threshold.csv"
        file_exists = os.path.isfile(csv_file)

        # Open the CSV file for appending data
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                # Write the header if the file does not exist
                writer.writerow(["timestamp", "id1", "id2", "distance"])

            employee_ids = list(employee_centers.keys())
            for i in range(len(employee_ids)):
                for j in range(i, len(employee_ids)):  # Compare each pair only once
                    id1, id2 = employee_ids[i], employee_ids[j]

                    # Get the positions of both employees
                    center1 = employee_centers[id1]
                    center2 = employee_centers[id2]

                    # Calculate the Euclidean distance between the two employee centers
                    dist = distance.euclidean(center1, center2)

                    # Check if the distance is below the talking threshold
                    if dist < talking_threshold:
                        writer.writerow([timestamp, id1, id2, round(dist, 2)])

    def write_oncall_csv(self):
        with open("oncall_status.csv", "w") as file:
            file.write("ObjectID,Label,FormattedOnCallTime\n")
            for obj_id, data in self.id_storage.items():
                if data["label"] == "employee":
                    oncall_time = data["oncall_time"]
                    minutes, seconds = divmod(oncall_time, 60)
                    formatted_oncall_time = f"{minutes:02}:{seconds:02}"
                    file.write(f"{obj_id},{data['label']},{formatted_oncall_time}\n")

    def write_area_csv(self):
        """Write the total time spent in each area to a CSV file."""
        with open("area_details.csv", "w") as file:
            file.write("PersonType,ObjectID,Area,TotalTime\n")
            for area, data in self.area_time.items():
                for obj_id, time_data in data.items():
                    total_time = int(time_data["total"])
                    minutes, seconds = divmod(total_time, 60)
                    formatted_time = f"{minutes:02}:{seconds:02}"
                    file.write(f"{'employee' if obj_id < self.employee_id_counter else 'customer'},{obj_id},{area},{formatted_time}\n")

    def match_or_assign_id(self, x_center, y_center, label, threshold=50):
        min_distance = float('inf')
        matched_id = None

        if label == "employee":
            # Match employees in employee dictionary
            for obj_id, data in self.employee_id_storage.items():
                prev_x, prev_y = data['coords'][-1]
                dist = distance.euclidean((x_center, y_center), (prev_x, prev_y))
                if dist < threshold and dist < min_distance:
                    min_distance = dist
                    matched_id = obj_id

            if matched_id is None:
                matched_id = self.employee_id_counter
                self.employee_id_counter += 1
                self.employee_id_storage[matched_id] = {'coords': [(x_center, y_center)]}

        else:  
            for obj_id, data in self.customer_id_storage.items():
                prev_x, prev_y = data['coords'][-1]
                dist = distance.euclidean((x_center, y_center), (prev_x, prev_y))
                if dist < threshold and dist < min_distance:
                    min_distance = dist
                    matched_id = obj_id

            if matched_id is None:
                matched_id = self.customer_id_counter
                self.customer_id_counter += 1
                self.customer_id_storage[matched_id] = {'coords': [(x_center, y_center)]}

        return matched_id

    def draw_bounding_box(self, frame, x1, y1, x2, y2, label, obj_id, area):
        color = (0, 255, 0) if label == "employee" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        font = cv2.FONT_HERSHEY_COMPLEX
        font_scale = 0.5
        thickness = 1

        # Track elapsed time for employees and customers
        if label == "employee" and obj_id in self.employee_times:
            total_time = int(time.time()) - self.employee_times[obj_id]
        elif label == "customer" and obj_id in self.customer_times:
            total_time = int(time.time()) - self.customer_times[obj_id]
        else:
            total_time = 0

        minutes, seconds = divmod(total_time, 60)
        detection_time = f"{minutes:02}:{seconds:02}"

        # Add area information to the text
        text = f"{label} ID: {obj_id} Time: {detection_time}"

        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = x1
        text_y = max(y1 - 10, text_height + 5)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

    def write_csv(self, label, obj_id, x_center, y_center, area):
        with open(self.csv_file, "a") as file:
            timestamp = int(time.time())
            file.write(f"{label},{obj_id},{timestamp},{x_center},{y_center},{area}\n")

    def calculate_center(self, x1, y1, x2, y2):
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        return x_center, y_center

    def determine_area(self, x_center, y_center):

        wa_tl_x, wa_tl_y = self.waiting_area[0]
        wa_br_x, wa_br_y = self.waiting_area[1]

        # Check if the center is within the waiting area
        if wa_tl_x <= x_center <= wa_br_x and wa_tl_y <= y_center <= wa_br_y:
            return "Waiting area"
        # If not in waiting area, then classify as service area
        else:
            return "Service area"

    def display_frame(self, frame):
        label_width = self.video_label.width()
        label_height = self.video_label.height()
        resized_frame = cv2.resize(frame, (label_width, label_height), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        h, w, c = frame_rgb.shape
        qt_image = QImage(frame_rgb.data, w, h, c * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap(qt_image)
        self.video_label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec())
