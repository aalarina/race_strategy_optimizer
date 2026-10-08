# race_strategy_optimizer
A predictive system for pit-stop strategy and tyre wear

# 🏎️ F1 Tyre Degradation & Strategy Analysis

⚠️ **Work in Progress (WIP)**  
*This project is currently under active development. The core codebase and Jupyter notebooks are being built locally and will be pushed to this repository in stages as code refactoring is completed.*

---

## 📌 Project Overview & Core Idea
The goal of this project is to model Formula 1 **tyre degradation** and predict optimal pit stop windows using telemetry and timing data from the **FastF1** library.

### Key Objectives:
* **Advanced Feature Engineering:** Cleaning raw FIA timing data by building custom filters (isolating race starts/Lap 1, monitoring `TrackStatus`, and identifying precise out-laps and in-laps).
* **Degradation Modeling:** Implementing a **Fuel-Corrected Baseline** model that accounts for fuel burn (car weight loss) to isolate true, underlying tyre wear.
* **Strategy Analytics:** Detecting tyre performance "drop-off" points to forecast high-efficiency pit stop windows.

---

## 🗺️ Roadmap
- [x] Data acquisition and initial preprocessing via FastF1.
- [x] Custom anomaly filtering logic (`IsAccurate` expansion & `TrackStatus` tracking).
- [ ] Implement linear fuel-correction baseline — *Local, preparing for commit*.
- [ ] Visualize degradation curves across tire compounds (Soft/Medium/Hard) for top teams.
- [ ] Build a predictive model for optimal pit-window forecasting.
