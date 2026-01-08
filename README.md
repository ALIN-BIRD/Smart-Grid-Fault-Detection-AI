# 🛡️ Smart Grid Fault Detection & Autonomous Recovery using ML

![Python](https://img.shields.io/badge/AI-Scikit%20Learn-orange)
![MATLAB](https://img.shields.io/badge/Simulation-Simulink-blue)
![Status](https://img.shields.io/badge/Status-Prototype-green)

## 📌 Project Overview
This project is a **Real-Time Fault Detection and Protection System** for modern smart grids. It integrates a **Physics-Based Simulation (Simulink)** with an **External AI Controller (Python)** to predict power instability and prevent blackouts.

Unlike traditional protection systems (relays/fuses) that react *after* a fault occurs, this system uses **Machine Learning** to predict voltage collapse based on load trends and proactively reroutes power to a backup source, ensuring continuous uptime for critical infrastructure.

## 🎯 Key Objectives
* **Predictive Maintenance:** Use ML to forecast grid voltage stability in real-time.
* **Fault Detection:** Identify critical load anomalies (e.g., Short Circuits, Overloads).
* **Autonomous Restoration:** Automatically switch to a **Backup Generator** without human intervention during crises.
* **Hardware-in-the-Loop (HIL):** Demonstrate live communication between Python (Controller) and Simulink (Plant).

## 🛠️ Technology Stack
* **Simulation Environment:** MATLAB Simulink (Simscape Electrical).
* **Control Logic & Interface:** Python 3.10 (Tkinter, MATLAB Engine API).
* **Machine Learning:** Scikit-Learn (Linear Regression for Voltage Prediction).
* **Communication:** Inter-Process Communication (IPC) via `matlab.engine`.

## ⚙️ System Architecture
The system operates on a 100ms closed-loop cycle:

1.  **Data Acquisition:** Python extracts live `Resistance` (Load) and `Voltage` (Output) data from the Simulink model.
2.  **ML Inference:** The trained AI model (`vpp_brain.pkl`) predicts the **Steady-State Voltage** for the current load.
3.  **Decision Making:**
    * *Normal Operation:* Grid remains connected to Main Supply.
    * *Fault Detected:* If Predicted Voltage < 11.0V (or Load < 20Ω), the system flags a **Critical Fault**.
4.  **Actuation:** Python triggers the **Automatic Transfer Switch (ATS)** in Simulink to disconnect the Main Supply and engage the Backup Generator.

## 📂 Repository Structure
```text
Smart-Grid-Fault-Detection/
│
├── 📂 assets/                # Demo screenshots and diagrams
│   ├── dashboard_ui.png
│   └── simulink_circuit.png
│
├── 📂 src/                   # Source Code
│   ├── grid_controller.py    # Main Python Dashboard (Tkinter)
│   ├── train_model.py        # ML Training Script
│   └── debug_link.py         # Connection diagnostics tool
│
├── 📂 simulation/            # Physics Models
│   ├── Grid_Model.slx        # Simulink Circuit (Simscape)
│   └── ai_model.pkl          # Trained ML Brain
│
├── requirements.txt          # Python dependencies
└── README.md                 # Project Documentation
