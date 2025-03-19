import os
import csv
import math
import numpy as np
import random
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
    # Reset session data when starting a new session
    session['question_count'] = 0
    session['asked_truthful_indices'] = []
    session['asked_deceptive_indices'] = []
    session['phase'] = 'truthful'
    return render_template('index.html')


@app.route('/log_data', methods=['POST'])
def log_data():
    data = request.json

    # Get the last asked question index from session
    last_question_index = session.get('last_question_index', 0)

    # Get user's answer (1 for yes, 0 for no)
    user_answer = data.get('answer', None)
    if user_answer is None:
        return jsonify({"status": "error", "message": "No answer provided"})

    # Get ground truth for this question
    ground_truth = GROUND_TRUTH[last_question_index]

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


def get_random_question_index(phase):
    """Get a random question index that hasn't been asked yet for the current phase."""
    # Define the range of indices based on the phase
    if phase == 'truthful':
        # For truthful phase, use questions 0-9 (first 10 questions)
        available_indices = list(range(10))
        already_asked = session.get('asked_truthful_indices', [])
    else:  # 'deceptive' phase
        # For deceptive phase, use questions 11-20 (skip the empty question at index 10)
        available_indices = list(range(11, 21))
        already_asked = session.get('asked_deceptive_indices', [])

    # Filter out questions that have already been asked
    remaining_indices = [idx for idx in available_indices if idx not in already_asked]

    # If we've asked all questions in this phase, return None
    if not remaining_indices:
        return None

    # Select a random question from the remaining ones
    return random.choice(remaining_indices)


@app.route('/get_question')
def get_question():
    # Get current question count
    question_count = session.get('question_count', 0)
    phase = session.get('phase', 'truthful')

    # Check if we need to show instructions
    if question_count == 0:
        # Starting instructions - first time
        session['question_count'] = 1
        session['phase'] = 'truthful'
        session['asked_truthful_indices'] = []
        session['asked_deceptive_indices'] = []

        return jsonify({
            "isInstruction": True,
            "instruction": INSTRUCTIONS["start"],
            "questionNumber": 0,
            "totalQuestions": 20  # Total of 20 questions (10 truthful + 10 deceptive)
        })
    elif question_count == 10:
        # Switch to deceptive mode instructions
        session['phase'] = 'deceptive'
        session['question_count'] = 11

        return jsonify({
            "isInstruction": True,
            "instruction": INSTRUCTIONS["switch"],
            "questionNumber": 10,
            "totalQuestions": 20
        })
    elif question_count >= 20:
        # End of experiment
        return jsonify({
            "isInstruction": True,
            "instruction": INSTRUCTIONS["end"],
            "complete": True
        })

    # Get a random question index for the current phase
    question_index = get_random_question_index(phase)

    # Store the last question index for logging purposes
    session['last_question_index'] = question_index

    # Add this question to the list of asked questions
    if phase == 'truthful':
        asked_indices = session.get('asked_truthful_indices', [])
        asked_indices.append(question_index)
        session['asked_truthful_indices'] = asked_indices
    else:  # 'deceptive' phase
        asked_indices = session.get('asked_deceptive_indices', [])
        asked_indices.append(question_index)
        session['asked_deceptive_indices'] = asked_indices

    # Increment question count
    session['question_count'] = question_count + 1

    # Return the selected question
    return jsonify({
        "isInstruction": False,
        "question": QUESTIONS[question_index],
        "questionNumber": question_count + 1,
        "totalQuestions": 20
    })


if __name__ == '__main__':
    app.run(debug=True)
