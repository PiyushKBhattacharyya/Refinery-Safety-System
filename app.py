from flask import Flask, jsonify
import sqlite3

app = Flask(__name__)

def get_violations():
    conn = sqlite3.connect('violations.db')
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, area, ppe_missing, height_level, image_path FROM violations")
    rows = cursor.fetchall()
    conn.close()
    # Convert to list of dicts
    return [
        {
            "timestamp": row[0],
            "area": row[1],
            "ppe_missing": row[2],
            "height_level": row[3],
            "image_path": row[4]
        } for row in rows
    ]

@app.route("/violations")
def violations_api():
    data = get_violations()
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, jsonify, send_from_directory
import csv

app = Flask(__name__)

@app.route('/')
def serve_dashboard():
    return send_from_directory('.', 'dashboard.html')  # serve the HTML file

@app.route('/violations')
def get_violations():
    violations = []
    with open('violations.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            violations.append({
                'timestamp': row['timestamp'],
                'area': row['area'],
                'ppe_missing': row['ppe_missing'],
                'height_level': float(row['height_level']),
                'image_path': row['image_path']
            })
    return jsonify(violations)

if __name__ == '__main__':
    app.run(debug=True)
