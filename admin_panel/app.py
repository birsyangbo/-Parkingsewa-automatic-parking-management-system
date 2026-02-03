from flask import Flask, render_template, send_from_directory, redirect, url_for, flash, request, jsonify
import sqlite3
import os
import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_super_secret_key_here' # IMPORTANT: Change this to a random, strong key!

# --- Configuration ---
# These paths are relative to where app.py is located.
# os.pardir goes up one directory level.
# So, if app.py is in 'Parkingsewa/admin_panel/', then 'os.pardir' points to 'Parkingsewa/'.
DB_NAME = os.path.join(os.pardir, 'parking_log.db')
PLATE_IMAGES_DIR = os.path.join(os.pardir, 'plate_images')
TICKETS_DIR = os.path.join(os.pardir, 'tickets')
LIVE_FEED_DIR = os.path.join(os.pardir, 'live_feed') # New directory for live feed snapshots

# Create the live_feed directory if it doesn't exist
os.makedirs(LIVE_FEED_DIR, exist_ok=True)

# --- Global variable to store the current year for the footer ---
CURRENT_YEAR = datetime.datetime.now().year

# --- Helper Function to get DB connection ---
def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row # This allows accessing columns by name (e.g., record['plate_number'])
    return conn

# --- Routes (URLs for your web app) ---

@app.route('/')
def index():
    """Renders the dashboard page with summary statistics."""
    conn = get_db_connection()
    cursor = conn.cursor()

    total_vehicles = cursor.execute("SELECT COUNT(id) FROM parking_records").fetchone()[0]
    parked_vehicles = cursor.execute("SELECT COUNT(id) FROM parking_records WHERE status = 'parked'").fetchone()[0]

    conn.close()
    return render_template('index.html',
                           total_vehicles=total_vehicles,
                           parked_vehicles=parked_vehicles,
                           current_year=CURRENT_YEAR)

# API endpoint for real-time parked count
@app.route('/api/parked_count')
def get_parked_count():
    conn = get_db_connection()
    parked_vehicles = conn.execute("SELECT COUNT(id) FROM parking_records WHERE status = 'parked'").fetchone()[0]
    conn.close()
    return jsonify({'parked_count': parked_vehicles})

@app.route('/records')
def view_records():
    """Renders the page to view all parking records with search/filter."""
    conn = get_db_connection()
    query = "SELECT * FROM parking_records WHERE 1=1" # Start with a always true condition
    params = []

    # Get filter parameters from the URL query string
    plate_number = request.args.get('plate_number')
    status = request.args.get('status')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    if plate_number:
        query += " AND plate_number LIKE ?"
        params.append(f"%{plate_number}%") # Use % for partial matches

    if status:
        query += " AND status = ?"
        params.append(status)

    if start_date:
        query += " AND entry_time >= ?"
        params.append(f"{start_date} 00:00:00") # Start of the day

    if end_date:
        query += " AND entry_time <= ?"
        params.append(f"{end_date} 23:59:59") # End of the day

    query += " ORDER BY entry_time DESC"

    records = conn.execute(query, params).fetchall()
    conn.close()
    return render_template('records.html', records=records, current_year=CURRENT_YEAR)

@app.route('/records/delete/<int:record_id>', methods=['POST']) # Use POST method for deletion for security
def delete_record(record_id):
    conn = get_db_connection()
    try:
        conn.execute("DELETE FROM parking_records WHERE id = ?", (record_id,))
        conn.commit()
        flash(f'Record {record_id} deleted successfully.', 'success')
    except sqlite3.Error as e:
        flash(f'Error deleting record {record_id}: {e}', 'error')
    finally:
        conn.close()
    return redirect(url_for('view_records'))


@app.route('/images')
def view_images():
    """Renders a gallery of detected plate images."""
    images = []
    if os.path.exists(PLATE_IMAGES_DIR):
        for filename in os.listdir(PLATE_IMAGES_DIR):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                images.append(filename)
        images.sort(reverse=True) # Sort to show most recent images first
    
    return render_template('images.html', images=images, current_year=CURRENT_YEAR)

@app.route('/plate_images/<filename>')
def serve_image(filename):
    """Serves individual plate image files from the PLATE_IMAGES_DIR."""
    return send_from_directory(PLATE_IMAGES_DIR, filename)

@app.route('/tickets')
def view_tickets():
    """Renders a list of generated ticket images (PNGs)."""
    tickets = []
    if os.path.exists(TICKETS_DIR):
        for filename in os.listdir(TICKETS_DIR):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')): # Changed to look for images
                tickets.append(filename)
        tickets.sort(reverse=True)
    
    return render_template('tickets.html', tickets=tickets, current_year=CURRENT_YEAR)

@app.route('/tickets/<filename>')
def serve_ticket(filename):
    """Serves individual ticket image files from the TICKETS_DIR."""
    return send_from_directory(TICKETS_DIR, filename)

@app.route('/reports')
def view_reports():
    """Renders a page with various aggregate reports."""
    conn = get_db_connection()
    cursor = conn.cursor()

    daily_report_query = """
    SELECT
        strftime('%Y-%m-%d', entry_time) AS report_date,
        COUNT(CASE WHEN status = 'parked' OR status = 'exited' THEN id END) AS entries,
        COUNT(CASE WHEN status = 'exited' THEN id END) AS exits,
        SUM(CASE WHEN status = 'exited' THEN cost ELSE 0 END) AS total_revenue
    FROM parking_records
    WHERE entry_time >= strftime('%Y-%m-%d %H:%M:%S', date('now', '-30 days')) -- Last 30 days
    GROUP BY report_date
    ORDER BY report_date DESC;
    """
    daily_reports = cursor.execute(daily_report_query).fetchall()

    conn.close()
    return render_template('reports.html',
                           daily_reports=daily_reports,
                           current_year=CURRENT_YEAR)

@app.route('/live_feed_page') # A new route to show the live feed HTML page
def live_feed_page():
    return render_template('live_feed.html', current_year=CURRENT_YEAR)

@app.route('/live_feed/latest_frame.jpg')
def serve_latest_frame():
    """Serves the latest camera frame snapshot."""
    # Add cache-busting headers to ensure the browser always gets the new image
    response = send_from_directory(LIVE_FEED_DIR, 'latest_frame.jpg', mimetype='image/jpeg')
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# --- Run the Flask Application ---
if __name__ == '__main__':
    app.run(debug=True,port=5002)