import cv2
import easyocr
from ultralytics import YOLO
import sqlite3
import datetime
import os
import imagehash
from PIL import Image, ImageDraw, ImageFont, ImageOps
import logging
import pyttsx3
import time
import numpy as np
import qrcode

# --- Configuration Constants ---
DB_NAME = 'parking_log.db'
PLATE_IMAGES_DIR = 'plate_images'
TICKET_IMAGES_DIR = 'tickets'
TICKET_FONT_PATH = 'arial.ttf' 

# YOLO Models
PLATE_DETECTION_MODEL = 'best.pt'
VEHICLE_CLASSIFICATION_MODEL = 'yolov8n.pt' # General object detection for vehicle type
# --- LIVE FEED CONFIGURATION (NEW) ---
LIVE_FEED_DIR = 'live_feed' # Directory to save live frames
LIVE_FRAME_PATH = os.path.join(LIVE_FEED_DIR, 'latest_frame.jpg') # Full path for the latest frame image
SAVE_FRAME_INTERVAL_SECONDS = 0.1 # How often to save a frame (e.g., 0.1s for 10 frames/sec)

# Confidence Thresholds
CONFIDENCE_THRESHOLD_PLATE = 0.5        # Minimum confidence for plate detection
CONFIDENCE_THRESHOLD_VEHICLE = 0.6      # Minimum confidence for vehicle classification

# Hash Matching
LOOSE_HASH_THRESHOLD = 15               # Max hamming distance for 'loose' hash matching (for exit/cooldown)
STRICT_HASH_THRESHOLD = 8               # Max hamming distance for 'strict' hash matching (for pending entry unique samples)

# Entry/Exit Logic
ENTRY_CONFIRMATION_SECONDS = 9          # How long a plate must be consistently detected for entry confirmation
MAX_ABSENCE_FOR_PENDING_ENTRY_SECONDS = 2 # How long a pending plate can be absent before cancelling
SAMPLES_TO_COLLECT_ON_ENTRY = 3        # Number of unique hashes to collect for entry confirmation
RECENTLY_EXITED_COOLDOWN_SECONDS = 60   # How long an exited vehicle is on cooldown to prevent re-entry

# New constant for idle period after entry
POST_ENTRY_IDLE_SECONDS = 10 

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Global variable for camera
cap = None
plate_detection_model = None
vehicle_classification_model = None
ocr_reader = None
engine = None 
# Global variable for tracking last frame save time (NEW)
last_save_time = time.time()

def initialize_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS parking_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            entry_time TEXT NOT NULL,
            exit_time TEXT,
            duration_minutes REAL,
            cost REAL,
            status TEXT NOT NULL,
            vehicle_type TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vehicle_hashes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id INTEGER,
            hash_value TEXT NOT NULL,
            image_path TEXT,
            FOREIGN KEY (record_id) REFERENCES parking_records(id)
        )
    ''')
    conn.commit()
    conn.close()
    logging.info("Database initialized.")

def load_models():
    global plate_detection_model, vehicle_classification_model, ocr_reader, engine
    try:
        plate_detection_model = YOLO(PLATE_DETECTION_MODEL)
        logging.info(f"Loaded plate detection model: {PLATE_DETECTION_MODEL}")
        vehicle_classification_model = YOLO(VEHICLE_CLASSIFICATION_MODEL)
        logging.info(f"Loaded vehicle classification model: {VEHICLE_CLASSIFICATION_MODEL}")
        ocr_reader = easyocr.Reader(['en']) 
        logging.info("Loaded EasyOCR reader for English.")
        
        engine = pyttsx3.init()
        engine.setProperty('rate', 150) 
        logging.info("Initialized Text-to-Speech engine.")

    except Exception as e:
        logging.error(f"Error loading models or TTS engine: {e}")
        exit()

def get_db_connection():
    return sqlite3.connect(DB_NAME)

def record_entry(plate_number, vehicle_type):
    conn = get_db_connection()
    cursor = conn.cursor()
    entry_time = datetime.datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO parking_records (plate_number, entry_time, status, vehicle_type) VALUES (?, ?, ?, ?)",
        (plate_number, entry_time, 'parked', vehicle_type)
    )
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()
    logging.info(f"Entry recorded: ID:{new_id}, Plate:{plate_number}, Type:{vehicle_type}")
    return new_id

def record_exit(record_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    exit_time = datetime.datetime.now()
    
    # Fetch entry_time and vehicle_type
    cursor.execute("SELECT entry_time, vehicle_type FROM parking_records WHERE id = ?", (record_id,))
    result = cursor.fetchone()
    if not result:
        logging.error(f"Record ID {record_id} not found for exit processing.")
        conn.close()
        return

    entry_time_str, vehicle_type = result
    entry_time = datetime.datetime.fromisoformat(entry_time_str)
    
    duration = (exit_time - entry_time).total_seconds() / 60
    
    # --- NEW COST CALCULATION LOGIC ---
    cost_per_minute = 20 # Default for 'Rest' (bus, truck, unknown)

    if vehicle_type: # Ensure vehicle_type is not None
        vehicle_type_lower = vehicle_type.lower()
        if vehicle_type_lower == "car":
            cost_per_minute = 10
        elif vehicle_type_lower == "motorcycle": # Assuming 'bike' maps to 'motorcycle' from YOLO
            cost_per_minute = 5
        # If it's 'bus', 'truck', or 'unknown', it will use the default 20
    
    cost = duration * cost_per_minute
    # --- END NEW COST CALCULATION LOGIC ---
    
    cursor.execute(
        "UPDATE parking_records SET exit_time = ?, duration_minutes = ?, cost = ?, status = ? WHERE id = ?",
        (exit_time.isoformat(), duration, cost, 'exited', record_id)
    )
    conn.commit()
    conn.close()
    logging.info(f"Exit recorded for ID:{record_id}. Vehicle Type: {vehicle_type}. Duration: {duration:.2f} min, Cost: {cost:.2f}")
    
    generate_ticket(record_id)

def get_parking_records():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, plate_number, entry_time, exit_time, duration_minutes, cost, status, vehicle_type FROM parking_records")
    records = cursor.fetchall()
    conn.close()
    return records

def save_plate_image(image_np):
    os.makedirs(PLATE_IMAGES_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"plate_{timestamp}.jpg"
    filepath = os.path.join(PLATE_IMAGES_DIR, filename)
    try:
        cv2.imwrite(filepath, image_np)
        return filepath
    except Exception as e:
        logging.warning(f"Failed to save plate image to {filepath}: {e}")
        return None

def save_vehicle_hash(record_id, image_path, hash_value_str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO vehicle_hashes (record_id, hash_value, image_path) VALUES (?, ?, ?)",
        (record_id, hash_value_str, image_path)
    )
    conn.commit()
    conn.close()
    logging.debug(f"Saved hash '{hash_value_str}' for record ID {record_id}")

def get_vehicle_hashes(record_id=None, status=None, since_time=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        SELECT vh.record_id, vh.hash_value, vh.image_path, pr.status, pr.entry_time, pr.exit_time, pr.plate_number
        FROM vehicle_hashes vh
        JOIN parking_records pr ON vh.record_id = pr.id
    """
    conditions = []
    params = []

    if record_id is not None:
        conditions.append("vh.record_id = ?")
        params.append(record_id)
    
    if status is not None:
        conditions.append("pr.status = ?")
        params.append(status)

    if since_time is not None:
        conditions.append("pr.exit_time >= ?") 
        params.append(since_time.isoformat())

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()
    
    parsed_results = []
    for row in results:
        record_id, hash_str, image_path, status, entry_time, exit_time, plate_number = row 
        try:
            img_hash = imagehash.hex_to_hash(hash_str)
            parsed_results.append({
                'record_id': record_id,
                'hash_value': img_hash,
                'image_path': image_path,
                'status': status,
                'entry_time': datetime.datetime.fromisoformat(entry_time) if entry_time else None,
                'exit_time': datetime.datetime.fromisoformat(exit_time) if exit_time else None,
                'plate_number': plate_number
            })
        except ValueError as e:
            logging.warning(f"Invalid hash string '{hash_str}' for record_id {record_id}: {e}")
    return parsed_results

def calculate_perceptual_hash(image_np):
    try:
        img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        p_hash = imagehash.phash(pil_img)
        return str(p_hash)
    except Exception as e:
        logging.error(f"Error calculating perceptual hash: {e}")
        return None

# The old get_vehicle_type function is removed as it's replaced by the new detection flow.

def speak_message(text):
    """Speaks a given text message using the TTS engine."""
    if engine:
        logging.info(f"Speaking: '{text}'")
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logging.error(f"Error during TTS engine.say or runAndWait: {e}")
    else:
        logging.warning("Text-to-Speech engine not initialized.")

def generate_ticket(record_id):
    """Generates and displays a parking ticket with a QR code and plate image."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT plate_number, entry_time, exit_time, duration_minutes, cost, vehicle_type FROM parking_records WHERE id = ?", (record_id,))
    record = cursor.fetchone()

    plate_image_path = None
    cursor.execute("SELECT image_path FROM vehicle_hashes WHERE record_id = ? LIMIT 1", (record_id,))
    result = cursor.fetchone()
    if result:
        plate_image_path = result[0]
    conn.close()

    if not record:
        logging.error(f"Record with ID {record_id} not found for ticket generation.")
        return

    plate_number_ocr, entry_time_str, exit_time_str, duration_minutes, cost, vehicle_type = record

    formatted_entry = datetime.datetime.fromisoformat(entry_time_str).strftime("%Y-%m-%d %H:%M:%S")
    formatted_exit = datetime.datetime.fromisoformat(exit_time_str).strftime("%Y-%m-%d %H:%M:%S")
    formatted_duration = f"{duration_minutes:.2f} minutes"
    formatted_cost = f"Rs. {cost:.2f}" 

    width, height = 500, 700
    ticket_image = Image.new('RGB', (width, height), color = (255, 255, 255)) 
    draw = ImageDraw.Draw(ticket_image)

    try:
        font_large = ImageFont.truetype(TICKET_FONT_PATH, 30)
        font_medium = ImageFont.truetype(TICKET_FONT_PATH, 20)
        font_small = ImageFont.truetype(TICKET_FONT_PATH, 16)
    except IOError:
        logging.warning(f"Could not load font '{TICKET_FONT_PATH}', using default PIL font. "
                        "For better aesthetics, place 'arial.ttf' in your project directory "
                        "or specify its full path.")
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()

    y_offset = 30
    text_color = (0, 0, 0) 

    title_text = "Parking Ticket"
    bbox = draw.textbbox((0, 0), title_text, font=font_large) 
    title_width = bbox[2] - bbox[0]
    title_height = bbox[3] - bbox[1]
    
    draw.text(((width - title_width) / 2, y_offset), title_text, fill=text_color, font=font_large)
    y_offset += title_height + 30 

    draw.text((50, y_offset), f"Ticket ID:", fill=text_color, font=font_medium)
    draw.text((200, y_offset), str(record_id), fill=text_color, font=font_medium)
    y_offset += 30
    
    draw.text((50, y_offset), f"Plate Image:", fill=text_color, font=font_medium) 
    y_offset += 5 

    if plate_image_path and os.path.exists(plate_image_path):
        try:
            plate_img_pil = Image.open(plate_image_path)
            
            target_plate_width = 300
            target_plate_height = 100 
            
            plate_img_pil.thumbnail((target_plate_width, target_plate_height), Image.Resampling.LANCZOS)
            
            padded_plate_img = Image.new('RGB', (target_plate_width, target_plate_height), (255, 255, 255)) 
            
            paste_x = (target_plate_width - plate_img_pil.width) // 2
            paste_y = (target_plate_height - plate_img_pil.height) // 2
            padded_plate_img.paste(plate_img_pil, (paste_x, paste_y))

            plate_img_x = (width - padded_plate_img.width) // 2 
            plate_img_y = y_offset 
            
            ticket_image.paste(padded_plate_img, (plate_img_x, plate_img_y))
            y_offset += padded_plate_img.height + 10 
            logging.info(f"Pasted plate image {plate_image_path} onto ticket.")

        except Exception as e:
            logging.warning(f"Could not load or paste plate image from {plate_image_path}: {e}. Falling back to OCR text.")
            draw.text((200, y_offset), plate_number_ocr, fill=text_color, font=font_medium)
            y_offset += 30 
    else:
        logging.warning(f"No plate image found at {plate_image_path} or path invalid. Printing plate number OCR text instead.")
        draw.text((200, y_offset), plate_number_ocr, fill=text_color, font=font_medium)
        y_offset += 30 

    draw.text((50, y_offset), f"Vehicle Type:", fill=text_color, font=font_medium)
    draw.text((200, y_offset), vehicle_type, fill=text_color, font=font_medium)
    y_offset += 50 

    draw.text((50, y_offset), f"Entry Time:", fill=text_color, font=font_small)
    draw.text((200, y_offset), formatted_entry, fill=text_color, font=font_small)
    y_offset += 25
    draw.text((50, y_offset), f"Exit Time:", fill=text_color, font=font_small)
    draw.text((200, y_offset), formatted_exit, fill=text_color, font=font_small)
    y_offset += 25
    draw.text((50, y_offset), f"Duration:", fill=text_color, font=font_small)
    draw.text((200, y_offset), formatted_duration, fill=text_color, font=font_small)
    y_offset += 25
    draw.text((50, y_offset), f"Cost:", fill=text_color, font=font_small)
    draw.text((200, y_offset), formatted_cost, fill=text_color, font=font_small)
    y_offset += 50

    qr_data = (f"Ticket ID: {record_id}\n"
               f"Plate: {plate_number_ocr} (OCR)\n" 
               f"Entry: {formatted_entry}\n"
               f"Exit: {formatted_exit}\n"
               f"Duration: {formatted_duration}\n"
               f"Cost: {formatted_cost}")
    
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(qr_data)
    qr.make(fit=True)

    qr_img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
    
    qr_pos_x = (width - qr_img.size[0]) // 2
    qr_pos_y = y_offset + 20 
    ticket_image.paste(qr_img, (qr_pos_x, qr_pos_y))

    os.makedirs(TICKET_IMAGES_DIR, exist_ok=True)
    ticket_filename = f"ticket_{record_id}_{plate_number_ocr}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    ticket_filepath = os.path.join(TICKET_IMAGES_DIR, ticket_filename)
    try:
        ticket_image.save(ticket_filepath)
        logging.info(f"Ticket generated and saved to: {ticket_filepath}")
        
        ticket_np = np.array(ticket_image)
        ticket_np_bgr = cv2.cvtColor(ticket_np, cv2.COLOR_RGB2BGR)
        
        cv2.imshow(f"Parking Ticket - ID: {record_id}", ticket_np_bgr)
        cv2.waitKey(8000) 
        cv2.destroyWindow(f"Parking Ticket - ID: {record_id}") 
        
    except Exception as e:
        logging.error(f"Failed to save or display ticket image: {e}")

def main_loop():
    # IMPORTANT: Ensure 'last_save_time' is added to your global declaration
    global cap, last_save_time, engine, current_parked_vehicles, last_seen_plates, park_times, exit_detections, last_exit_time # Add other globals if they were already there.

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logging.error("Error: Could not open camera.")
        return

    # --- START OF NEW CODE FOR LIVE FEED DIRECTORY (Step 3) ---
    # Ensure the live_feed directory exists
    os.makedirs(LIVE_FEED_DIR, exist_ok=True)
    # --- END OF NEW CODE FOR LIVE FEED DIRECTORY ---

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logging.debug(f"Camera resolution: {width}x{height}")

    pending_entry_data = {
        'plate_ocr': None,
        'vehicle_type': None, # This will now come directly from vehicle detection
        'first_seen': None,
        'last_seen': None,
        'collected_hashes': []
    }

    entry_idle_tracker = {}

    last_status_message = ""
    start_time = time.time()

    # COCO vehicle classes IDs for filtering (from yolov8n.pt trained on COCO)
    # These are specific to what yolov8n.pt can detect
    relevant_vehicle_classes_ids = [2, 3, 5, 7]  # 2=car, 3=motorbike, 5=bus, 7=truck

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Error: Failed to grab frame. Exiting ...")
                break

            # --- START OF NEW CODE FOR LIVE FEED FRAME SAVING (Step 4) ---
            current_loop_time = time.time()
            # Check if enough time has passed since the last frame was saved
            if current_loop_time - last_save_time >= SAVE_FRAME_INTERVAL_SECONDS:
                # Save the current frame to the specified path
                cv2.imwrite(LIVE_FRAME_PATH, frame)
                # Update the last save time
                last_save_time = current_loop_time
            # --- END OF NEW CODE FOR LIVE FEED FRAME SAVING ---

            current_status_message = ""
            current_time = datetime.datetime.now()

            # --- Cleanup expired idle periods ---
            expired_ids = [
                rec_id for rec_id, expiry_time in entry_idle_tracker.items()
                if current_time >= expiry_time
            ]
            for rec_id in expired_ids:
                del entry_idle_tracker[rec_id]
                logging.debug(f"Idle period expired for record ID: {rec_id}")

            # --- Vehicle and Plate Detection ---
            display_frame = frame.copy()
            stable_detection_info = None # Reset for each frame

            # Step 1: Detect vehicles in the full frame
            # verbose=False reduces console output from YOLO models during inference
            vehicle_results = vehicle_classification_model(frame, verbose=False)

            best_plate_conf = 0.0
            best_found_plate_info = None # To store the best plate info across all vehicles

            # Iterate through all detected vehicles
            for v_result in vehicle_results:
                for v_box in v_result.boxes:
                    v_cls_id = int(v_box.cls[0])
                    v_conf = float(v_box.conf[0])

                    # Only consider relevant vehicle types with sufficient confidence
                    if v_cls_id in relevant_vehicle_classes_ids and v_conf >= CONFIDENCE_THRESHOLD_VEHICLE:
                        v_x1, v_y1, v_x2, v_y2 = map(int, v_box.xyxy[0])

                        # Ensure valid vehicle crop coordinates
                        h, w = frame.shape[:2]
                        v_x1, v_y1 = max(0, v_x1), max(0, v_y1)
                        v_x2, v_y2 = min(w, v_x2), min(h, v_y2)

                        if v_x1 >= v_x2 or v_y1 >= v_y2:
                            continue # Invalid vehicle bounding box

                        vehicle_crop = frame[v_y1:v_y2, v_x1:v_x2]

                        if vehicle_crop.size == 0:
                            logging.warning(f"Vehicle crop for {vehicle_classification_model.names[v_cls_id]} has zero size. Skipping plate detection for this vehicle.")
                            continue

                        # Determine vehicle type string (e.g., 'car', 'motorbike')
                        current_vehicle_type = vehicle_classification_model.names[v_cls_id]
                        if current_vehicle_type == "motorbike": # Normalize 'motorbike' to 'motorcycle' for consistency
                            current_vehicle_type = "motorcycle"

                        # Draw vehicle bounding box and label for visualization
                        cv2.rectangle(display_frame, (v_x1, v_y1), (v_x2, v_y2), (255, 0, 0), 2)
                        cv2.putText(display_frame, f"{current_vehicle_type} ({v_conf:.2f})", (v_x1, v_y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                        # Step 2: Detect license plates inside the vehicle crop
                        plate_results = plate_detection_model(vehicle_crop, verbose=False)

                        if plate_results and len(plate_results[0].boxes) > 0:
                            for p_box in plate_results[0].boxes:
                                p_conf = float(p_box.conf[0])

                                if p_conf >= CONFIDENCE_THRESHOLD_PLATE:
                                    # Get plate coordinates relative to vehicle_crop
                                    px1, py1, px2, py2 = map(int, p_box.xyxy[0])

                                    # Calculate absolute plate coordinates in the original frame
                                    abs_x1, abs_y1 = v_x1 + px1, v_y1 + py1
                                    abs_x2, abs_y2 = v_x1 + px2, v_y1 + py2

                                    # Ensure plate ROI is valid
                                    abs_x1, abs_y1 = max(0, abs_x1), max(0, abs_y1)
                                    abs_x2, abs_y2 = min(w, abs_x2), min(h, abs_y2)

                                    if abs_x1 >= abs_x2 or abs_y1 >= abs_y2:
                                        logging.warning("Plate ROI has zero size after absolute clamping. Skipping OCR.")
                                        continue

                                    plate_roi = frame[abs_y1:abs_y2, abs_x1:abs_x2]

                                    if plate_roi.size == 0:
                                        logging.warning("Plate ROI has zero size. Skipping OCR.")
                                        continue

                                    # Perform OCR and Hash on the plate ROI
                                    ocr_results = ocr_reader.readtext(plate_roi)
                                    ocr_text = "N/A"
                                    if ocr_results:
                                        ocr_text = ocr_results[0][1].replace(" ", "").upper() # Convert to uppercase

                                    current_computed_hash_str = calculate_perceptual_hash(plate_roi)

                                    if current_computed_hash_str:
                                        current_hash = imagehash.hex_to_hash(current_computed_hash_str)

                                        # If this is the most confident plate found so far, store its info
                                        if p_conf > best_plate_conf:
                                            best_plate_conf = p_conf
                                            best_found_plate_info = {
                                                'ocr_text': ocr_text,
                                                'bbox': [abs_x1, abs_y1, abs_x2, abs_y2],
                                                'confidence': p_conf,
                                                'plate_roi_np': plate_roi,
                                                'hash': current_hash,
                                                'vehicle_type': current_vehicle_type # This is the key addition!
                                            }

                                        # Draw plate bounding box and label for visualization
                                        plate_label_vis = f"Plate: {ocr_text} ({p_conf:.2f})"
                                        cv2.rectangle(display_frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 255, 0), 2)
                                        cv2.putText(display_frame, plate_label_vis, (abs_x1, abs_y1 - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # After checking all vehicles and their plates, use the best one found
            stable_detection_info = best_found_plate_info

            # --- Logic for Entry, Exit, Cooldown (Rest of the loop remains largely the same) ---

            parked_vehicles_hashes = get_vehicle_hashes(status='parked')

            cooldown_threshold_time = datetime.datetime.now() - datetime.timedelta(seconds=RECENTLY_EXITED_COOLDOWN_SECONDS)
            recently_exited_vehicles_hashes = get_vehicle_hashes(status='exited', since_time=cooldown_threshold_time)

            matched_existing_vehicle = False
            best_exit_match_id = None
            best_exit_dist = float('inf')

            is_recently_exited_match_on_cooldown = False

            if stable_detection_info:
                current_computed_hash = stable_detection_info['hash']
                current_ocr_text = stable_detection_info['ocr_text']
                current_vehicle_type = stable_detection_info['vehicle_type'] # Get vehicle type from stable_detection_info

                # 2. Check for EXIT (Match with PARKD vehicle)
                for stored_vehicle in parked_vehicles_hashes:
                    dist = stored_vehicle['hash_value'] - current_computed_hash
                    if dist <= LOOSE_HASH_THRESHOLD:
                        if dist < best_exit_dist:
                            best_exit_dist = dist
                            best_exit_match_id = stored_vehicle['record_id']

                        if stored_vehicle['plate_number'] != current_ocr_text and dist > 0:
                            logging.debug(f"    HASH MATCH (ID {stored_vehicle['record_id']}, Dist {dist}): OCR Mismatch detected. Stored '{stored_vehicle['plate_number']}' vs Current '{current_ocr_text}'")
                        break

                if best_exit_match_id is not None:
                    if best_exit_match_id in entry_idle_tracker and \
                        current_time < entry_idle_tracker[best_exit_match_id]:
                        current_status_message = f"INFO: Vehicle ID:{best_exit_match_id} is in post-entry idle period. Ignoring immediate exit."
                        logging.info(current_status_message)
                        matched_existing_vehicle = True
                    else:
                        record_exit(best_exit_match_id)
                        current_status_message = f"EXPLICIT EXIT: ID:{best_exit_match_id}, Plate:{current_ocr_text} (Hash Dist:{best_exit_dist:.2f})"
                        logging.info(current_status_message)
                        speak_message("Thank you.")
                        matched_existing_vehicle = True

                # 3. Check for RECENTLY EXITED vehicle on cooldown (prevent re-entry)
                if not matched_existing_vehicle:
                    best_cooldown_match_id = None
                    best_cooldown_dist = float('inf')
                    for stored_vehicle in recently_exited_vehicles_hashes:
                        dist = stored_vehicle['hash_value'] - current_computed_hash
                        if dist <= LOOSE_HASH_THRESHOLD:
                            if dist < best_cooldown_dist:
                                best_cooldown_dist = dist
                                best_cooldown_match_id = stored_vehicle['record_id']
                                is_recently_exited_match_on_cooldown = True

                    if is_recently_exited_match_on_cooldown:
                        current_status_message = f"RECENTLY EXITED VEHICLE DETECTED (ID:{best_cooldown_match_id}, Dist:{best_cooldown_dist:.2f}). Not creating new entry due to cooldown."
                        logging.info(current_status_message)
                        matched_existing_vehicle = True

            # 4. Handle Pending Entry (if no existing vehicle matched)
            if not matched_existing_vehicle and stable_detection_info:
                # Ensure the current detected plate OCR text is associated with the correct vehicle type
                if pending_entry_data['plate_ocr'] and \
                    (pending_entry_data['hash'] - stable_detection_info['hash'] <= LOOSE_HASH_THRESHOLD):

                    pending_entry_data['last_seen'] = current_time
                    current_duration_pending = (pending_entry_data['last_seen'] - pending_entry_data['first_seen']).total_seconds()

                    img_bytes = cv2.imencode('.jpg', stable_detection_info['plate_roi_np'])[1].tobytes()
                    is_unique_hash = True
                    for stored_hash_str, _ in pending_entry_data['collected_hashes']:
                        stored_hash = imagehash.hex_to_hash(stored_hash_str)
                        if stored_hash - stable_detection_info['hash'] <= STRICT_HASH_THRESHOLD:
                            logging.debug(f"Hash {stable_detection_info['hash']} too similar to existing {stored_hash} (dist: {stored_hash - stable_detection_info['hash']}) - Not collecting.")
                            is_unique_hash = False
                            break

                    if is_unique_hash:
                        pending_entry_data['collected_hashes'].append((str(stable_detection_info['hash']), img_bytes))
                        logging.debug(f"Collected new hash {stable_detection_info['hash']}. Total collected: {len(pending_entry_data['collected_hashes'])}/{SAMPLES_TO_COLLECT_ON_ENTRY}")

                    # Your original logic for ENTRY_CONFIRMATION_SECONDS and SAMPLES_TO_COLLECT_ON_ENTRY
                    if current_duration_pending >= ENTRY_CONFIRMATION_SECONDS and \
                       len(pending_entry_data['collected_hashes']) >= SAMPLES_TO_COLLECT_ON_ENTRY:

                        # Use the vehicle type directly from stable_detection_info
                        vehicle_type_for_entry = stable_detection_info['vehicle_type']
                        new_record_id = record_entry(pending_entry_data['plate_ocr'], vehicle_type_for_entry) # Use the specific type

                        for hash_val_str, img_data_bytes in pending_entry_data['collected_hashes']:
                            np_arr = np.frombuffer(img_data_bytes, np.uint8)
                            reconstructed_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                            filepath = save_plate_image(reconstructed_img)
                            if filepath:
                                save_vehicle_hash(new_record_id, filepath, hash_val_str)
                            else:
                                logging.warning(f"Failed to save image for hash {hash_val_str} during entry confirmation.")

                        current_status_message = f"ENTRY CONFIRMED: ID:{new_record_id}, Plate:{pending_entry_data['plate_ocr']}, Type:{vehicle_type_for_entry}"
                        logging.info(current_status_message)
                        speak_message("You can process.")

                        entry_idle_tracker[new_record_id] = current_time + datetime.timedelta(seconds=POST_ENTRY_IDLE_SECONDS)
                        logging.info(f"Started post-entry idle period for ID:{new_record_id} until {entry_idle_tracker[new_record_id].strftime('%H:%M:%S')}")

                        pending_entry_data = {
                            'plate_ocr': None, 'vehicle_type': None, 'first_seen': None, 'last_seen': None, 'collected_hashes': []
                        }
                    else:
                        current_status_message = f"Potential new vehicle detected. Continuing pending entry for '{pending_entry_data['plate_ocr']}'. Duration: {current_duration_pending:.1f}s, Samples: {len(pending_entry_data['collected_hashes'])}/{SAMPLES_TO_COLLECT_ON_ENTRY}"

                else: # New potential entry, or pending one cleared
                    if pending_entry_data['plate_ocr'] and \
                       (current_time - pending_entry_data['last_seen']).total_seconds() > MAX_ABSENCE_FOR_PENDING_ENTRY_SECONDS:

                        if is_recently_exited_match_on_cooldown:
                            current_status_message = f"Pending entry for '{pending_entry_data['plate_ocr']}' cleared due to cooldown match."
                            logging.info(current_status_message)
                        else:
                            current_status_message = f"Pending entry for '{pending_entry_data['plate_ocr']}' cleared due to absence."
                            logging.info(current_status_message)

                        pending_entry_data = {
                            'plate_ocr': None, 'vehicle_type': None, 'first_seen': None, 'last_seen': None, 'collected_hashes': []
                        }

                    # Start new pending entry if a stable_detection_info is present and no pending entry
                    if not pending_entry_data['plate_ocr'] and stable_detection_info:
                        pending_entry_data['plate_ocr'] = stable_detection_info['ocr_text']
                        pending_entry_data['hash'] = stable_detection_info['hash']
                        pending_entry_data['first_seen'] = current_time
                        pending_entry_data['last_seen'] = current_time
                        # IMPORTANT: Assign vehicle_type from the current stable detection info
                        pending_entry_data['vehicle_type'] = stable_detection_info['vehicle_type']

                        img_bytes = cv2.imencode('.jpg', stable_detection_info['plate_roi_np'])[1].tobytes()
                        pending_entry_data['collected_hashes'].append((str(stable_detection_info['hash']), img_bytes))
                        logging.info(f"Potential new vehicle detected. Starting pending entry process for '{pending_entry_data['plate_ocr']}' ({pending_entry_data['vehicle_type']}).")

            if current_status_message and current_status_message != last_status_message:
                logging.info(current_status_message)
                last_status_message = current_status_message
            elif not current_status_message:
                last_status_message = ""

            cv2.imshow('Parking System - Live Feed', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.05)

    finally:
        logging.info("Parking system stopped.")
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print_current_records()

def print_current_records():
    print("\n--- Current Parking Records in DB ---")
    records = get_parking_records()
    if records:
        print(f"{'ID':<5} | {'Plate':<15} | {'Entry Time':<25} | {'Exit Time':<26} | {'Duration (min)':<15} | {'Cost':<10} | {'Status':<10} | {'Vehicle Type':<15}")
        print("-" * 120)
        for record in records:
            entry_t = datetime.datetime.fromisoformat(record[2]).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
            exit_t = datetime.datetime.fromisoformat(record[3]).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] if record[3] else "N/A"
            duration = f"{record[4]:.2f}" if record[4] is not None else "N/A"
            cost = f"{record[5]:.2f}" if record[5] is not None else "N/A"
            print(f"{record[0]:<5} | {record[1]:<15} | {entry_t:<25} | {exit_t:<26} | {duration:<15} | {cost:<10} | {record[6]:<10} | {record[7]:<15}")
    else:
        print("No records found.")
    print("-" * 120)

if __name__ == "__main__":
    initialize_db()
    load_models()
    try:
        main_loop()
    except KeyboardInterrupt:
        logging.info("System interrupted by user.")
    finally:
        if engine:
            engine.stop()
        logging.info("Parking system stopped.")
        print_current_records()



