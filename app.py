import os
import csv
import random
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import openai

# Configure your OpenAI key (only used if is_bot == False)
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)

# For session-based memory (if you keep memory logic):
app.secret_key = os.getenv("FLASK_API_KEY")

# Make sure these directories exist
os.makedirs('data/bot', exist_ok=True)
os.makedirs('data/human', exist_ok=True)

FALLBACK_QUESTIONS = [
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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/log_data', methods=['POST'])
def log_data():
    data = request.json
    is_bot = data.get('is_bot', False)
    subfolder = 'bot' if is_bot else 'human'

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'data/{subfolder}/tracking_{timestamp}.csv'

    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'x', 'y', 'dx', 'dy', 'acceleration', 'click']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for entry in data.get('data', []):
            writer.writerow(entry)

    return jsonify({"status": "success"})


@app.route('/get_question')
def get_question():
    """
    If `is_bot == true` (from the query param), return a random fallback
    yes/no question. Otherwise, use OpenAI to generate a question
    (with memory, or no memory if you prefer).
    """
    is_bot_str = request.args.get('bot', 'false')
    is_bot = (is_bot_str.lower() == 'true')

    if is_bot:
        # Disable OpenAI for the bot, just pick a random fallback question
        question_text = random.choice(FALLBACK_QUESTIONS)
        return jsonify({"question": question_text})
    else:
        # Example: Use OpenAI with memory or your custom logic
        # (Shown is a minimal example. You can adapt or remove memory if desired.)
        asked_questions = session.get('asked_questions', [])
        asked_text = "\n".join(f"- {q}" for q in asked_questions)
        # Prompt the model not to repeat any questions
        system_prompt = (
            "You are a helpful assistant that ONLY returns yes/no questions. "
            "Return exactly one short yes/no question, and nothing else."
            "The questions should be personal or about the surroundings of the person answering the "
            "question."
            "Examples of questions are:"
            "Do you like pizza?"
            "Is the sky blue?"
            "Do you enjoy reading?"
            "Are cats better than dogs?"
            "Do you like coffee?"
            "Have you traveled recently?"
            "Do you enjoy coding?"
            "Is chocolate delicious?"
            "Are you a morning person?"
            "Do you like risky adventures?"
            "Do you like to be around people?"
            "Are you the life of the party?"
            "Do you like to play chess?"
            "Is fashion important to you?"
            "Do you enjoy your own company?"
            "Do you like to read?"
            "Are you interested in sports?"
            "Do you like to learn new things?"
            "Do you like to sing?"
            "Are you shy?"
            "Have you ever told a lie?"
            "Do you have any hobbies?"
            "Could you speak in front of a crowd?"
            "Do you like a good debate?"
            "Do you avoid conflict at all costs?"
            "Are you humble?"
            "Does skydiving sound like fun?"
            "Do you like scary movies?"
            "Do you think you know more than most people?"
            "Is your advice always good?"
            "Would you like to travel to another country?"
            "Are you a homebody?"
            "Are you creative?"
            "Do you push yourself to achieve?"
            "Do you like to do as little as possible to get by?"
            "Is the beach your happy place?"
            "Are you an animal lover?"
            "Are you afraid of heights?"
            "Do you like pizza?"
            "Do you like sushi?"
            "Can you cook a gourmet meal?"
            "Do you like a clean house?"
            "Do you like to shop?"
            "Are you a collector of anything?"
            "Do you like spending time with kids?"
            "Would you camp in a tent?"
            "Do you like being in big cities?"
            "Could you live in a rural area?"
            "The user has already been asked these questions:\n"
            f"{asked_text}\n"
            "Please provide a NEW yes/no question that is not in the above list."
        )

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_prompt}],
                max_tokens=50,
                temperature=0.7,
                n=1
            )
            question_text = response.choices[0].message.content.strip()
        except Exception as e:
            question_text = random.choice(FALLBACK_QUESTIONS)
            print(f"OpenAI API error: {e}")

        # Update session memory
        if question_text:
            asked_questions.append(question_text)
            session['asked_questions'] = asked_questions

        return jsonify({"question": question_text})


if __name__ == '__main__':
    app.run(debug=True)
