# Mouse Tracking Experiment Web App

A Flask-based web application for conducting a mouse tracking study with multiple responsive layouts and detailed data analysis.

## Features

- **Dynamic Layout System**: 5 unique, responsive CSS layouts
- **Mouse Tracking**: Captures detailed mouse movement and interaction data
- **Experimental Design**:
    - Truthful vs. Deceptive response tracking
    - Location-based questioning
    - Comprehensive data logging

## Project Structure

```
.
├── data
│   ├── deceptive/    # Logs for deceptive responses
│   └── truthful/     # Logs for truthful responses
├── images/           # Layout preview images
│   ├── layout-1.png
│   ├── layout-2.png
│   └── ...
├── static/           # Static assets
│   ├── layout-1.css
│   ├── layout-2.css
│   └── script.js
├── templates/
│   └── index.html
├── app.py            # Flask backend
├── display.py        # Data visualization script
└── README.md
```

## Layouts Showcase

The application randomly selects from 5 unique layouts for each experiment session:

| Layout   | Preview                           | Description                         |
|----------|-----------------------------------|-------------------------------------|
| Layout 1 | ![Layout 1](/images/layout-1.png) | Vertical, clean modern design       |
| Layout 2 | ![Layout 2](/images/layout-2.png) | Horizontal, split-screen approach   |
| Layout 3 | ![Layout 3](/images/layout-3.png) | Innovative corner-positioned layout |
| Layout 4 | ![Layout 4](/images/layout-4.png) | Modern diagonal layout              |
| Layout 5 | ![Layout 5](/images/layout-5.png) | Fully responsive clean layout       |

## Key Components

### Backend (`app.py`)
- Manages experiment flow
- Handles question generation
- Logs mouse tracking data
- Supports truthful and deceptive response phases

### Data Logging
- Captures detailed mouse tracking metrics:
    - Timestamp
    - X, Y coordinates
    - Velocity
    - Path efficiency
    - Curvature
    - Decision path analysis

### Visualization (`display.py`)
- Generates comprehensive visualizations of mouse tracking data
- Compares truthful vs. deceptive response patterns
- Creates:
    - 2D trajectory plots
    - 3D trajectory visualization
    - Velocity comparisons
    - Path efficiency analysis

## Technologies Used

- **Backend**: Flask
- **Frontend**: HTML, CSS, JavaScript
- **Data Analysis**:
    - Pandas
    - NumPy
    - Matplotlib

## Setup and Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Mouse-Tracking-Flask-Demo
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install flask pandas numpy matplotlib
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

## Running the Experiment

1. Open `http://localhost:5000` in your browser
2. Follow on-screen instructions
3. Answer questions truthfully and then in a deceptive manner
4. Mouse movements are automatically tracked and logged

## Data Analysis

After running the experiment:
1. Generated data will be in `data/truthful/` and `data/deceptive/`
2. Run `display.py` to generate visualization plots
   ```bash
   python display.py
   ```