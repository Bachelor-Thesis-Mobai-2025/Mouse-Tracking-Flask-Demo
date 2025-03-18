let mouseData = [];
let currentQuestion = null;
let questionStartTime = null;
let selectedAnswer = null;

// Grab references
const yesBtn = document.getElementById('yes-btn');
const noBtn = document.getElementById('no-btn');
const nextBtn = document.getElementById('next-btn');
const questionElement = document.getElementById('question');

// Fixed-rate sampling at 100Hz (10ms)
const samplingInterval = 1000 / 100; // 10ms for 100Hz

// Current mouse position
let currentMouseX = 0;
let currentMouseY = 0;
let lastRecordedX = 0;  // Changed variable name for clarity
let lastRecordedY = 0;  // Changed variable name for clarity

const locationEmbeds = {
    // Norway locations
    "Norway": "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d7115129.1005240865!2d7.202227919814455!3d64.19044446471571!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x461268458f4de5bf%3A0xa1b03b9db864d02b!2sNorge!5e0!3m2!1sno!2sno!4v1742225138899!5m2!1sno!2sno",
    "Gjovik": "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d31680.92327955288!2d10.676129899999999!3d60.798508!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x4641da7f7d25d825%3A0xc834e9351bd371f1!2sGj%C3%B8vik!5e0!3m2!1sno!2sno!4v1711064304382!5m2!1sno!2sno",
    "Innlandet": "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d1999520.267789213!2d10.49408859677242!3d61.26669287075171!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x4613202593a930b7%3A0x4613202593a931b5!2sInnlandet!5e0!3m2!1sno!2sno!4v1711064362211!5m2!1sno!2sno",
    "NTNU": "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d1947.0953831666425!2d10.679612413300815!3d60.78973439317518!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x4641da14f48bc6d1%3A0x15d7b34504988672!2zTlROVSBww6UgR2rDuHZpaw!5e0!3m2!1sno!2sno!4v1742225884390!5m2!1sno!2sno",

    // United States locations
    "USA": "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d24314527.50780894!2d-102.58805748934844!3d40.13885294325144!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x54eab584e432360b%3A0x1c3bb99243deb742!2sUSA!5e0!3m2!1sno!2sno!4v1742225936787!5m2!1sno!2sno",
    "Berkeley": "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d50660.14164260444!2d-122.30440033250237!3d37.87223135536998!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x80857c3254417ca3%3A0x1f0ce75c7cfefe47!2sBerkeley%2C%20CA%2C%20USA!5e0!3m2!1sno!2sno!4v1711064428032!5m2!1sno!2sno",
    "California": "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d12443468.11099838!2d-122.97043661879774!3d36.778261015535724!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x80859a6d00690021%3A0x4a501367f076adff!2sCalifornien%2C%20USA!5e0!3m2!1sno!2sno!4v1711064461141!5m2!1sno!2sno",
    "UC Berkeley": "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3149.532814939858!2d-122.25804328751286!3d37.87121830648002!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x808f7718c522d7c1%3A0xda8034ea3b6b3289!2sUniversity%20of%20California!5e0!3m2!1sno!2sno!4v1742226174344!5m2!1sno!2sno",

// Australian locations
    "Australia": "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d34368450.22012237!2d134.48828173214755!3d-25.27433291133309!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x2b2bfd076787c5df%3A0x538267a1955b1352!2sAustralia!5e0!3m2!1sno!2sno!4v1711064593220!5m2!1sno!2sno",
    "Melbourne": "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d402590.52635753667!2d144.72282398041585!3d-37.971563335846504!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x6ad646b5d2ba4df7%3A0x4045675218ccd90!2sMelbourne%20Victoria%2C%20Australia!5e0!3m2!1sno!2sno!4v1742226652455!5m2!1sno!2sno",
    "Victoria": "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3285885.6074694144!2d140.45994787418059!3d-36.46063347963703!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x6ad4314b7e18954f%3A0x5a4efce2be829534!2sVictoria%2C%20Australia!5e0!3m2!1sno!2sno!4v1742226621015!5m2!1sno!2sno",
    "RMIT": "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3152.2194357913054!2d144.96135831278156!3d-37.808328833612855!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x6ad642cb0a2ff0fb%3A0xed6e6acedcefb31c!2sRMIT%20University!5e0!3m2!1sno!2sno!4v1742224523627!5m2!1sno!2sno"
};

// Update current position on mouse move
document.addEventListener('mousemove', function(event) {
    // Just update the current position
    currentMouseX = event.clientX;
    currentMouseY = event.clientY;
});

// Sample at fixed rate
setInterval(function() {
    const currentTime = Date.now();

    // Calculate dx and dy from last recorded position
    const dx = currentMouseX - lastRecordedX;
    const dy = currentMouseY - lastRecordedY;

    // Displacement (pixels)
    const displacement = Math.sqrt(dx**2 + dy**2);

    // Time elapsed (seconds)
    const timeElapsed = samplingInterval / 1000;

    // Velocity (pixels per second)
    const velocity = displacement / timeElapsed;

    const entry = {
        timestamp: currentTime,
        x: currentMouseX,
        y: currentMouseY,
        dx: dx,
        dy: dy,
        velocity: velocity,
        click: 0
    };

    // Only update the last recorded position AFTER recording the entry
    lastRecordedX = currentMouseX;
    lastRecordedY = currentMouseY;

    mouseData.push(entry);
}, samplingInterval);

// Timer
function updateTimer() {
    if (questionStartTime) {
        const elapsed = Math.floor((Date.now() - questionStartTime) / 1000);
        document.getElementById('timer').textContent = `Time on question: ${elapsed}s`;
    }
}
setInterval(updateTimer, 1000);

// Log a click
function logClick() {

    const currentTime = Date.now();

    // Calculate dx and dy from last recorded position
    const dx = currentMouseX - lastRecordedX;
    const dy = currentMouseY - lastRecordedY;

    // Displacement (pixels)
    const displacement = Math.sqrt(dx**2 + dy**2);

    // Time elapsed (seconds)
    const timeElapsed = samplingInterval / 1000;

    // Velocity (pixels per second)
    const velocity = displacement / timeElapsed;

    const entry = {
        timestamp: currentTime,
        x: currentMouseX,
        y: currentMouseY,
        dx: dx,
        dy: dy,
        velocity: velocity,
        click: 1
    };
    mouseData.push(entry);
}

// Save data
function saveData() {
    // Don't save if no answer selected
    if (selectedAnswer === null) {
        console.warn("No answer selected, not saving data");
        return;
    }

    fetch('/log_data', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            data: mouseData,
            answer: selectedAnswer // 1 for yes, 0 for no
        })
    })
        .then(response => response.json())
        .then(data => {
            if (data.status !== "success") {
                console.error("Failed to save data:", data.message);
            }
        })
        .catch(error => {
            console.error("Error saving data:", error);
        });

    // Reset data and selected answer
    mouseData = [];
    selectedAnswer = null;
}

// Fetch question
function fetchQuestion() {
    const url = `/get_question`;
    fetch(url)
        .then(response => response.json())
        .then(data => {
            if (data.complete) {
                // Experiment is complete
                questionElement.textContent = data.instruction;
                // Hide the answer buttons and next button
                yesBtn.style.display = 'none';
                noBtn.style.display = 'none';
                nextBtn.style.display = 'none';
                // Hide map container
                document.getElementById('map-container').style.display = 'none';
                return;
            }

            // Check if this is an instruction screen
            if (data.isInstruction) {
                // Display instruction
                questionElement.textContent = data.instruction;
                // Hide answer buttons for instructions
                yesBtn.style.display = 'none';
                noBtn.style.display = 'none';
                // Show next button to continue
                nextBtn.style.display = 'block';
                // Hide map container
                document.getElementById('map-container').style.display = 'none';
                // Reset timer
                questionStartTime = Date.now();
                return;
            }

            // Normal question display
            currentQuestion = data.question;
            questionElement.textContent = data.question;

            // Check if we should show a map for this question
            const mapContainer = document.getElementById('map-container');
            const mapIframe = document.getElementById('location-map');
            let mapFound = false;

            // Check each location against the question
            for (const [location, embedUrl] of Object.entries(locationEmbeds)) {
                if (data.question.includes(location)) {
                    mapIframe.src = embedUrl;
                    mapContainer.style.display = 'block';
                    mapFound = true;
                    break;
                }
            }

            // Hide map if no matching location
            if (!mapFound) {
                mapContainer.style.display = 'none';
            }

            // Reset button states
            yesBtn.classList.remove('selected');
            noBtn.classList.remove('selected');

            // Reset timer
            questionStartTime = Date.now();

            // Make sure buttons are visible
            yesBtn.style.display = 'block';
            noBtn.style.display = 'block';
            nextBtn.style.display = 'block';

            // Reset selected answer
            selectedAnswer = null;
        })
        .catch(error => {
            console.error("Error fetching question:", error);
        });
}

// Human clicks
yesBtn.addEventListener('click', () => {
    if (!yesBtn.classList.contains('selected')) {
        selectedAnswer = 1; // Yes = 1
        logClick();
        yesBtn.classList.add('selected');
        noBtn.classList.remove('selected');
    }
});

noBtn.addEventListener('click', () => {
    if (!noBtn.classList.contains('selected')) {
        selectedAnswer = 0; // No = 0
        logClick();
        noBtn.classList.add('selected');
        yesBtn.classList.remove('selected');
    }
});

// Existing shake animation function (if not already defined)
function shakeElement(element) {
    element.classList.add('shake-animation');
    setTimeout(() => {
        element.classList.remove('shake-animation');
    }, 500);
}

// CSS for shake animation (can be added to your stylesheet)
const shakeStyles = `
@keyframes shake {
    0%, 100% { transform: translateX(0); }
    10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
    20%, 40%, 60%, 80% { transform: translateX(5px); }
}

.shake-animation {
    animation: shake 0.5s ease-in-out;
}
`;

// Add styles to the document if not already present
const styleSheet = document.createElement("style");
styleSheet.type = "text/css";
styleSheet.innerText = shakeStyles;
document.head.appendChild(styleSheet);

// In your existing event listener
nextBtn.addEventListener('click', () => {
    if (yesBtn.style.display !== 'none' && noBtn.style.display !== 'none') {
        saveData();
    }
    if (yesBtn.classList.contains('selected')
        || noBtn.classList.contains('selected')
        || yesBtn.style.display === 'none'
        && noBtn.style.display === 'none') {
        fetchQuestion();
    } else {
        // Shake both buttons to indicate selection is needed
        shakeElement(yesBtn);
        shakeElement(noBtn);
    }
});

// Initial load
fetchQuestion();