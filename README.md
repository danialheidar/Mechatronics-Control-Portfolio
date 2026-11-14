# Mechatronics, Control & Robotics Portfolio
**Danial Heidar** — B.Sc. Student in Mechanical Engineering, University of Tehran[cite: 1]

This repository organizes my coursework and personal programming projects in **Control Systems**, **Robot Kinematics**, and **AI/Search Algorithms**.

---

## 1. Rotary Inverted Pendulum — Dynamic Modeling & Control
* **Project Report:** [`Control_Project_Report.pdf`](./Control_Project_Report.pdf) *(Group project; my primary role was dynamic modeling, stability analysis, and controller design)*[cite: 1, 2]
* **Overview:**
  * Derived nonlinear Euler-Lagrange equations of motion and linearized the system around the unstable upright equilibrium[cite: 7, 8].
  * Designed and evaluated **P, PI, PD, PID** controllers and a **Lead Compensator** in **MATLAB** using Root Locus, Bode, and Nyquist stability analyses[cite: 2, 16, 17].
  * Collaborated with teammates responsible for SolidWorks CAD modeling and Proteus/Arduino hardware simulation[cite: 17, 21].

---

## 2. Humanoid Robot Parallel Wrist Kinematics (Surena V)
* **Contribution Commit:** [surenaV_core — commit `a5a7379`](https://github.com/danialheidar/surenaV_core/commit/a5a73798365c1ac2ada1f31bd9f1f97b267616d5)
* **Overview:**
  * Implemented an updated geometric/analytical inverse kinematics solution for the Surena V parallel wrist mechanism.
  * Added boundary handling for **out-of-workspace target ranges** and selection logic to pick the optimal solution among **multiple analytical branches**.

---

## 3. AI & Algorithmic Programming Projects (Python)
* **Constraint-Propagation Sudoku Solver (`sudoku_solver_gui.py`):** Combines **Naked/Hidden Singles** inference over a `9×9×9` candidate matrix with **Minimum Remaining Values (MRV)** heuristic branching and iterative stack-based backtracking, wrapped in a **Pygame** GUI.
* **AlphaZero Practice Implementations (Backgammon, Fading Tic-Tac-Toe, Tic-Tac-Toe):** Self-play agents integrating Monte Carlo Tree Search (MCTS) with neural networks, adapted to stochastic dice transitions in Backgammon.
* **Neural Pong:** Real-time paddle control agent trained via neural networks.
