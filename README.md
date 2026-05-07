# AI-Based Driver Drowsiness Detection System

A simple real-time driver drowsiness detection project made using Python, OpenCV, and MediaPipe.  
The system monitors the driver's eyes through a webcam and gives an alert when signs of sleep or fatigue are detected.

## Features

- Real-time face and eye tracking
- Detects eye closure using Eye Aspect Ratio (EAR)
- Alarm alert when driver becomes drowsy
- Hazard light simulation
- Simulated vehicle speed reduction
- Adaptive calibration for different users
- Simple and lightweight project

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy

## Project Working

1. Webcam captures the driver's face
2. Facial landmarks are detected using MediaPipe
3. Eye Aspect Ratio (EAR) is calculated
4. If eyes stay closed for a certain time, the system detects sleep
5. Alarm and safety simulation are triggered

## Installation

**Install the required libraries:**

## bash
```
pip install -r requirements.txt
```

## Run the Project
```
python main.py
```
Press Q to close the application.

## Future Improvements: 
 - Better performance in low light
 - Mobile application support
 - More accurate fatigue detection
 - Integration with real vehicle systems

## Authors:
Sarthak Rao
