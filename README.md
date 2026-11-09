# AURA squad submission 6 - Mission 6 (The Autonomous Warp State Machine & Deliberation Loop)

A native-Python agent control stack: state machine, tool protocol with error boundary, MDP trajectory logging with discounted-return
evaluation, and Floyd cycle detection with halting guardrails for Ship Anomaly 06. Standard library only (Python 3.9+).

| Folder | Station | Objective | Main files |
|---|---|---|---|
| 01_Data_Specialist | Data Specialist | MDP trajectory logger + empirical discounted-return evaluator | mdp_logger.py, evaluation_demo.py, test_mdp_logger.py, sample_trajectories.jsonl |
| 02_Protocol_Specialist | Protocol Specialist | Tool Protocol schema validator + error-boundary dispatcher | tool_protocol.py, test_tool_protocol.py |
| 03_Security_Sentinel | Security Sentinel | Floyd cycle detector + halting guardrails (Ship Anomaly 06) | cycle_detector.py, anomaly_06_demo.py, test_cycle_detector.py |
| 04_Systems_Architect | Systems Architect | AgentStateMachine core + deliberation loop integration | agent_state_machine.py, deliberation_loop.py, ship_sim.py, policies.py, run_mission.py, state_diagram.dot, tests |
| (root) | Full squad | Integrated notebook, final report, file guide | ps06_solution.ipynb, final report (.docx), AURA_File_Guide_M6.xlsx |

Keep the four station folders side by side: the loop imports from all of them.

## Run
    cd 04_Systems_Architect && python -m unittest -v && python run_mission.py
    cd ../02_Protocol_Specialist && python -m unittest -v
    cd ../01_Data_Specialist && python -m unittest -v && python evaluation_demo.py
    cd ../03_Security_Sentinel && python -m unittest -v && python anomaly_06_demo.py
    jupyter notebook ps06_solution.ipynb

## Status and assumptions
- All code and tests were executed (61 unit tests); the numbers in the report are real outputs.
- The brief gave four objectives but no spec file or lecture notes were available when this was built (docs/weeks/week-06/problem_set/ps06.md was not accessible). The state set, tool schema format, reward values and the definition of "Ship Anomaly 06" (a runaway repeat-without-progress loop in warp control) are our design choices. Compare them with the spec and adjust.
- No LLM is called: scripted and random policies stand in for AURA's model, and the ship is a small deterministic simulator.
- Fill in the cadet names and the repository URL before submitting.
"# SQUAD_RUMELHART_MISSION-6" 
