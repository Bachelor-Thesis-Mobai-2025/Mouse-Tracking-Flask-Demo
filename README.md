# Mouse-Tracking-Flask-Demo

A **Flask**-based web app that presents yes/no questions to a user, logs mouse movements/clicks, and optionally simulates a “bot” that moves a **fake mouse pointer** around the webpage and automatically answers questions. It can also generate questions from **OpenAI** or fall back to a random set of yes/no questions.

## Table of Contents

1. [Project Description](#project-description)
2. [Technologies & Requirements](#technologies--requirements)
3. [Setup & Installation](#setup--installation)
4. [Running the Application](#running-the-application)
5. [Application Architecture](#application-architecture)
6. [Usage Guide](#usage-guide)
    - [Human Mode](#human-mode)
    - [Bot Mode](#bot-mode)
    - [Pause & Random Start Toggle](#pause--random-start-toggle)
7. [Bot Simulation Details](#bot-simulation-details)
8. [OpenAI Integration](#openai-integration-optional)
9. [FAQ & Troubleshooting](#faq--troubleshooting)

---

## Project Description

The **Mouse-Tracking-Flask-Demo** is a simple data collection demo that serves yes/no questions to users. Each time a question is presented, the application logs **mouse movement** and **click** data into CSV files. You can run it in two modes:

1. **Human Mode**: The user manually moves the mouse and clicks.
2. **Bot Mode**: A **fake pointer** (red circle) automatically navigates to the **Yes** or **No** button, clicks, then clicks **Next Question**.

When in **Bot Mode**, you can also **pause** the bot, enable/disable random starting positions for each cycle, and observe the generated CSV data under separate folders (`data/bot/` vs. `data/human/`).

Optionally, the server can fetch newly generated **yes/no** questions from the **OpenAI API** when in Human Mode, while the Bot Mode uses a fallback list.

---

## Technologies & Requirements

- **Python 3.12+**
- **Flask** (for the web server)
- **Flask-CORS** (optional, if cross-origin requests are needed)
- **OpenAI** (optional, only if you enable GPT-based question generation)
- **Matplotlib** / **Pandas** (optional, if you want to analyze or visualize the CSV data)

### Folder Structure

```
.
├── data
│   ├── bot
│   └── human
├── templates
│   └── index.html
├── app.py
├── display.py
├── README.md
└── .gitignore
```

- **`app.py`**: The main Flask server file.
- **`display.py`**: Python script to display plot of the tracking data.
- **`templates/index.html`**: The front-end user interface with bot/human logic.
- **`data/`**: Folder to store CSV logs (divided into `bot/` and `human/`).

---

## Setup & Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/Bachelor-Thesis-Mobai-2025/Mouse-Tracking-Flask-Demo
   cd Mouse-Tracking-Flask-Demo
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install required dependencies**:
   ```bash
   pip install flask flask-cors openai pandas matplotlib
   ```
   > **Note**: If you don’t plan on using OpenAI or advanced analysis, you can skip installing `openai`, `pandas`, or `matplotlib`.

4. **(Optional) Set up your OpenAI API key**:
    - If you want GPT-generated questions, set your key as an environment variable:
      ```bash
      export OPENAI_API_KEY="your-secret-key"
      ```
    - On Windows Command Prompt:
      ```bash
      set OPENAI_API_KEY="your-secret-key"
      ```

5. **Create the data folders** if not already present:
   ```bash
   mkdir -p data/bot data/human
   ```

---

## Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```
   By default, Flask will run on <http://127.0.0.1:5000> with `debug=True`.  
   (You can also run `flask run` if you prefer the Flask CLI.)

2. **Open** your browser and navigate to <http://127.0.0.1:5000>.

3. **Interact** with the UI to test the human/bot modes.

---

## Application Architecture

### `app.py` (Backend)

- **Routes**:
    - `GET /`: Renders `index.html`.
    - `POST /log_data`: Receives mouse tracking data (JSON) from the front end.
        - Saves CSV to either `data/bot/` or `data/human/`, based on `is_bot`.
    - `GET /get_question`: Returns a yes/no question.
        - If `?bot=true`, picks from a fallback list.
        - Otherwise, can optionally call OpenAI for fresh yes/no questions or from a static set.

### `index.html` (Frontend)

- **Top-left panel**:
    - **Bot** toggle (checkbox)
    - **Random Start** toggle (checkbox, only enabled if bot is paused, and **Bot** Toggle is checked)
    - **Start** / **Pause** buttons
- **Main container**:
    - **Timer**: Shows how long the user/bot has spent on the current question.
    - **Question**: Dynamically updated from the server.
    - **Yes/No buttons**: The user (or bot) clicks one.
    - **Next Question button**: Fetches a new question, logs data, etc.

- **JavaScript** logic includes:
    - Tracking real mouse movements (`mousemove`) for human logs.
    - A **fake pointer** `<div id="bot-pointer">` simulating the bot’s “mouse” with incremental movements.
    - **Bot** logic that moves the pointer in small steps, clicks Yes/No, then Next.
    - **Pause** functionality (stops the bot’s cycle, re-enables toggles).
    - **Random Start** toggle, which decides if each bot cycle begins at a random screen location.

---

## Usage Guide

### Human Mode

1. **Make sure “Bot” is unchecked** in the top-left panel.
3. **Answer questions** manually by moving the real mouse, clicking **Yes** or **No**, then **Next Question**.
4. Each time you click **Next Question**, the app sends your mouse data to the server, logging it in `data/human/tracking_YYYYMMDD_HHMMSS.csv`.

### Bot Mode

1. **Check the “Bot”** checkbox.
2. Ensure **Random Start** is set as you like (default: On).
3. **Click “Start.”** The fake pointer (red circle) will begin moving around the screen, automatically picking Yes or No, then pressing Next.
4. **Pause** the bot at any time. This stops the cycle and **re-enables** the toggles.
5. With the bot running, data is saved in `data/bot/tracking_YYYYMMDD_HHMMSS.csv`.

### Pause & Random Start Toggle

- **Pause** stops the bot’s current run. Once paused:
    - The “Bot” checkbox is re-enabled so you can uncheck it and become a human again.
    - The **“Random Start”** toggle is also enabled (only if Bot is selected and paused).
- **Random Start** toggle:
    - **On** (checked): The bot pointer does an initial move to a random location each cycle before selecting Yes/No.
    - **Off** (unchecked): The bot pointer skips the random move and goes straight to the Yes/No button.

---

## Bot Simulation Details

- **Fake Pointer**: A red circle that moves in small increments to simulate realistic mouse movement.
- **Jitter**: Each movement step has a small random offset, avoiding perfectly straight lines.
- **Timing**: After each cycle (Yes/No → Next), the bot waits 1 second, then repeats, until **MAX_ITERATIONS** (1000 by default) or until paused.
- **Real vs. Fake**: Because of browser security, the real OS mouse pointer cannot be moved by JavaScript. This red circle is purely a front-end simulation.

---

## OpenAI Integration (Optional)

If you configured your **OpenAI** API key, the app can fetch GPT-generated yes/no questions in **Human Mode**. In `app.py`, we do:

```python
@app.route('/get_question')
def get_question():
    # if is_bot: pick fallback
    # else: call OpenAI or fallback
    ...
```

1. **Set** `openai.api_key = os.getenv("OPENAI_API_KEY")`.
2. In your front-end, we pass `botMode` as a query parameter. If `botMode` is `false`, we use GPT; if `true`, fallback.

### Potential issues with OpenAI:
- **Rate Limits**: Generating too many questions rapidly could exceed your API quota.
- **Prompt Quality**: Customize the prompt so GPT returns only yes/no questions.

---

## FAQ & Troubleshooting

1. **Why doesn’t the mouse pointer physically move on my screen?**
    - Browsers can’t move the real OS cursor for security reasons. The red dot is a *simulated pointer* inside the webpage.

2. **Where do I see the CSV logs?**
    - In your `data/bot/` or `data/human/` folders, named `tracking_YYYYMMDD_HHMMSS.csv`.

3. **Why can’t I toggle “Bot” while it’s running?**
    - The code prevents toggling mid-run to avoid confusion with partial data logs.

4. **Why is my console showing an OpenAI error or fallback question?**
    - If the OpenAI call fails or you didn’t set your API key, we use a random fallback question.

5. **How do I reduce jitter or speed up the bot?**
    - Modify the `movePointerSmoothlyTo` function in `index.html`. You can reduce the number of steps or the random offsets.

---