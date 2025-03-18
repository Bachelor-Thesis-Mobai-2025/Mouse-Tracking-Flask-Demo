import os
import csv
import math
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Secret key for session management
app.secret_key = os.urandom(32)

# Make sure these directories exist
os.makedirs('data/deceptive', exist_ok=True)
os.makedirs('data/truthful', exist_ok=True)

# Instructions for different phases
INSTRUCTIONS = {
    "start": "Thank you for taking time answering the following questions. We ask that you answer these next 10 "
             "questions truthfully.",
    "switch": "For the next 10 questions we ask that you answer the questions deceitfully (Lie).",
    "end": "You have completed all questions. Thank you for your participation!"
}

QUESTIONS = [
    # Truthful Yes Answers: Ground truth should be YES
    "Are you currently located in Norway?",
    "Are you currently located in the county: Innlandet?",
    "Are you currently located in Gjovik?",
    "Are you currently located at NTNU?",
    "Are you currently a Student at NTNU?",
    # Truthful No Answers: Ground truth should be NO
    "Are you currently located in the USA?",
    "Are you currently located in the state: California?",
    "Are you currently located in Berkeley?",
    "Are you currently located at UC, Berkeley?",
    "Are you currently a student at UC, Berkeley?",
    # Skipped to show instructions for deception
    "",
    # Deceitful Yes Answers: Ground truth should be NO
    "Are you currently located in Australia?",
    "Are you currently located in the state: Victoria?",
    "Are you currently located in Melbourne?",
    "Are you currently located at RMIT?",
    "Are you currently employed as a Professor at RMIT?",
    # Deceitful No Answers: Ground truth should be YES
    "Are you currently located in Norway?",
    "Are you currently located in the county: Innlandet?",
    "Are you currently located in Gjovik?",
    "Are you currently located at NTNU?",
    "Are you currently a Student at NTNU?",
]

# Define the ground truth for each question (1 for yes, 0 for no)
GROUND_TRUTH = [
    # Truthful Yes (first 5)
    1, 1, 1, 1, 1,
    # Truthful No (next 5)
    0, 0, 0, 0, 0,
    # Deceptive Yes (next 5)
    0, 0, 0, 0, 0,
    # Deceptive No (last 5)
    1, 1, 1, 1, 1
]


@app.route('/')
def index():
    # Reset question counter when starting a new session
    session['question_index'] = -1
    return render_template('index.html')


@app.route('/log_data', methods=['POST'])
def log_data():
    data = request.json
    question_index = session.get('question_index', 0) - 1  # -1 because we increment before getting next question

    # Get user's answer (1 for yes, 0 for no)
    user_answer = data.get('answer', None)
    if user_answer is None:
        return jsonify({"status": "error", "message": "No answer provided"})

    # Reduce the question index to prevent Out-of-Bounds
    if session.get('question_index', 0) > 10:
        question_index -= 1

    # Get ground truth for this question
    ground_truth = GROUND_TRUTH[question_index]

    # Determine if the answer was truthful
    is_truthful = user_answer == ground_truth

    # Determine answer direction (yes/no)
    answer_suffix = "yes" if user_answer == 1 else "no"

    # Select folder based on truthfulness
    subfolder = 'truthful' if is_truthful else 'deceptive'

    # Create filename with timestamp and answer suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'data/{subfolder}/tracking_{timestamp}_{answer_suffix}.csv'

    # Process mouse tracking data to add derived metrics
    mouse_data = data.get('data', [])
    processed_data = process_mouse_data(mouse_data)

    # Save to CSV
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'x', 'y', 'dx', 'dy', 'velocity', 'velocity_variability',
                      'curvature', 'decision_path_efficiency',
                      'final_decision_path_efficiency', 'changes_of_mind', 'click']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for entry in processed_data:
            writer.writerow(entry)

    return jsonify({"status": "success"})


def process_mouse_data(mouse_data):
    """Process mouse data to add velocity variability, curvature, and path efficiency metrics."""
    if not mouse_data:
        return []

    processed_data = []
    window_size = 5  # Look at 5 points for velocity variability

    # First pass: basic processing and ensure velocity field
    for i, entry in enumerate(mouse_data):
        # Create a new dict to avoid modifying the original
        new_entry = entry.copy()
        processed_data.append(new_entry)

    # Second pass: calculate velocity variability in a sliding window
    for i in range(len(processed_data)):
        # Get a window of velocities
        start_idx = max(0, i - window_size + 1)
        window = [processed_data[j].get('velocity', 0) for j in range(start_idx, i + 1)]

        # Calculate variability (standard deviation)
        if len(window) > 1:
            mean_vel = sum(window) / len(window)
            variance = sum((v - mean_vel)**2 for v in window) / len(window)
            std_dev = math.sqrt(variance)
            processed_data[i]['velocity_variability'] = std_dev
        else:
            processed_data[i]['velocity_variability'] = 0

    # Calculate curvature for each point
    if len(processed_data) >= 3:
        # Extract x, y coordinates
        points_x = [entry['x'] for entry in processed_data]
        points_y = [entry['y'] for entry in processed_data]

        # Calculate curvature
        curvatures = calculate_curvature(points_x, points_y)

        # Add to each entry
        for i, entry in enumerate(processed_data):
            if i < len(curvatures):
                entry['curvature'] = curvatures[i]
            else:
                entry['curvature'] = 0
    else:
        # For very short sequences
        for entry in processed_data:
            entry['curvature'] = 0

    # Calculate enhanced path efficiency metrics using decision analysis
    path_metrics = decision_path_analysis(processed_data)

    # Add the path efficiency metrics to each entry
    for entry in processed_data:
        entry['decision_path_efficiency'] = path_metrics['decision_path_efficiency']
        entry['final_decision_path_efficiency'] = path_metrics['final_decision_path_efficiency']
        entry['changes_of_mind'] = path_metrics['changes_of_mind']

    return processed_data


def calculate_curvature(x, y):
    """Calculate curvature at each point of a trajectory."""
    # First derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)

    # Second derivatives
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5

    # Replace any NaN or inf values with 0
    curvature = np.nan_to_num(curvature)

    return curvature.tolist()


def calculate_path_efficiency(x, y):
    """Calculate the ratio of direct distance to actual path length."""
    if len(x) < 2:
        return 1.0

    # Direct distance (straight line from start to end)
    direct_distance = math.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2)

    # Actual path length (sum of all segments)
    path_length = 0
    for i in range(1, len(x)):
        segment_length = math.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2)
        path_length += segment_length

    # Avoid division by zero
    if path_length == 0:
        return 1.0

    # Efficiency ratio (1.0 means perfectly straight path)
    return direct_distance / path_length


def decision_path_analysis(mouse_data):
    """Analyze the path based on decision-making process, accounting for changes of mind."""
    click_indices = [i for i, entry in enumerate(mouse_data) if entry['click'] == 1]

    results = {
        'decision_path_efficiency': 1.0,
        'final_decision_path_efficiency': 1.0,
        'changes_of_mind': 0
    }

    if not click_indices:
        return results  # No clicks detected

    # Since we no longer log Next button clicks, each click is a decision click
    if len(click_indices) >= 1:
        final_decision_idx = click_indices[-1]  # Last click is the final decision

        # Count changes of mind (number of clicks minus 1)
        results['changes_of_mind'] = len(click_indices) - 1

        # Calculate path efficiency up to first decision
        if click_indices[0] > 0:
            first_decision_points_x = [entry['x'] for entry in mouse_data[:click_indices[0]+1]]
            first_decision_points_y = [entry['y'] for entry in mouse_data[:click_indices[0]+1]]
            results['decision_path_efficiency'] = calculate_path_efficiency(
                first_decision_points_x, first_decision_points_y
            )

        # Calculate path efficiency up to final decision
        final_decision_points_x = [entry['x'] for entry in mouse_data[:final_decision_idx+1]]
        final_decision_points_y = [entry['y'] for entry in mouse_data[:final_decision_idx+1]]
        results['final_decision_path_efficiency'] = calculate_path_efficiency(
            final_decision_points_x, final_decision_points_y
        )

    return results


@app.route('/get_question')
def get_question():
    # Get current question index
    question_index = session.get('question_index', -1)

    # Check if we need to show instructions
    if question_index == -1:
        # Starting instructions

        # Increment for next time
        session['question_index'] = question_index + 1
        return jsonify({
            "isInstruction": True,
            "instruction": INSTRUCTIONS["start"],
            "questionNumber": 0,
            "totalQuestions": len(QUESTIONS)
        })
    elif question_index == 10:
        # Switch to deceptive mode instructions

        # Increment for next time
        session['question_index'] = question_index + 1
        return jsonify({
            "isInstruction": True,
            "instruction": INSTRUCTIONS["switch"],
            "questionNumber": 10,
            "totalQuestions": len(QUESTIONS)
        })
    elif question_index >= len(QUESTIONS):
        # End of experiment
        return jsonify({
            "isInstruction": True,
            "instruction": INSTRUCTIONS["end"],
            "complete": True
        })

    # Return the current question
    current_question = QUESTIONS[question_index]

    # Increment for next time
    session['question_index'] = question_index + 1

    return jsonify({
        "isInstruction": False,
        "question": current_question,
        "questionNumber": question_index + 1,
        "totalQuestions": len(QUESTIONS)
    })


if __name__ == '__main__':
    app.run(debug=True)
