from ultralytics import YOLO
import cv2
import os
 
# Load your trained model
model = YOLO("/train/weights/best.pt")
 
root_dir =  "/data"
 
for image in os.listdir(root_dir):
    image_path =  os.path.join(root_dir,image)
    file_name = image.split(".")[0]
    # Run object detection
    results = model(image_path)  
    # Load image to get dimensions
    image = cv2.imread(image_path)
    # Get image height & width
    img_height, img_width, _ = image.shape  
 
    # Prepare text file to save bounding box details
    txt_output_path = f"/data/{file_name}.txt"
 
    with open(txt_output_path, "w") as file:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get absolute bounding box coordinates
            class_id = int(box.cls[0])  # Class ID
            # Convert to YOLO format (normalized)
            x_center = (x1 + x2) / 2 / img_width
            y_center = (y1 + y2) / 2 / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            # Write data in YOLO format
            file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
