import os
import csv
import random
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

os.makedirs('data', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/log_data', methods=['POST'])
def log_data():
    data = request.json
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'data/tracking_{timestamp}.csv'
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'x', 'y', 'dx', 'dy', 'acceleration', 'click']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        # Add question metadata to each row
        question_data = data.get('question_data', {})
        for entry in data.get('data', []):
            full_entry = {**entry, **question_data}
            writer.writerow(full_entry)
    
    return jsonify({"status": "success"})

@app.route('/get_question')
def get_question():
    questions = [
        "Do you like pizza?",
        "Is the sky blue?",
        "Do you enjoy reading?",
        "Are cats better than dogs?",
        "Do you like coffee?",
        "Have you traveled recently?",
        "Do you enjoy coding?",
        "Is chocolate delicious?",
        "Are you a morning person?",
        "Do you like risky adventures?",
        "Do you like to be around people?",
        "Are you the life of the party?",
        "Do you like to play chess?",
        "Is fashion important to you?",
        "Do you enjoy your own company?",
        "Do you like to read?",
        "Are you interested in sports?",
        "Do you like to learn new things?",
        "Do you like to sing?",
        "Are you shy?",
        "Have you ever told a lie?",
        "Do you have any hobbies?",
        "Could you speak in front of a crowd?",
        "Do you like a good debate?",
        "Do you avoid conflict at all costs?",
        "Are you humble?",
        "Does skydiving sound like fun?",
        "Do you like scary movies?",
        "Do you think you know more than most people?",
        "Is your advice always good?",
        "Would you like to travel to another country?",
        "Are you a homebody?",
        "Are you creative?",
        "Do you push yourself to achieve?",
        "Do you like to do as little as possible to get by?",
        "Is the beach your happy place?",
        "Are you an animal lover?",
        "Are you afraid of heights?",
        "Do you like pizza?",
        "Do you like sushi?",
        "Can you cook a gourmet meal?",
        "Do you like a clean house?",
        "Do you like to shop?",
        "Are you a collector of anything?",
        "Do you like spending time with kids?",
        "Would you camp in a tent?",
        "Do you like being in big cities?",
        "Could you live in a rural area?"
    ]
    
    return jsonify({
        "question": random.choice(questions)
    })

if __name__ == '__main__':
    app.run(debug=True)