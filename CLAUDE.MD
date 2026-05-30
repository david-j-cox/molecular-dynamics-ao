{\rtf1\ansi\ansicpg1252\cocoartf2870
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 HelveticaNeue-Bold;\f1\fnil\fcharset0 HelveticaNeue;\f2\fnil\fcharset0 .AppleSystemUIFontMonospaced-Regular;
}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
{\*\listtable{\list\listtemplateid1\listhybrid{\listlevel\levelnfc0\levelnfcn0\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{decimal\}.}{\leveltext\leveltemplateid1\'02\'00.;}{\levelnumbers\'01;}\fi-360\li720\lin720 }{\listname ;}\listid1}}
{\*\listoverridetable{\listoverride\listid1\listoverridecount0\ls1}}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab560
\pard\pardeftab560\partightenfactor0

\f0\b\fs40 \cf0 Behavioral Molecular Dynamics Simulation Engine\
\pard\pardeftab560\sa40\partightenfactor0

\fs32 \cf0 Project Goal\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 Build a Python simulation framework for modeling an organism interacting with an environment using an analogy to molecular dynamics.\
The organism is not represented as a single reinforcement-learning policy. Instead, it is represented as a distributed dynamical system composed of many response-capable units. Each unit has a current state, prior state, sensitivity to stimuli, response inertia, fatigue, and learning-history parameters. Environmental stimuli act as force fields. Learning history changes the direction and magnitude of the forces produced by those stimuli. Behavior emerges from the coupled dynamics of these units over time.\
The first version should be a proof-of-concept simulation, not a polished research package.\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 Core Concept\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 In molecular dynamics, particles move through space according to forces. In this project, behavioral units move through activation/state space according to stimulus functions.\
The organism contains many behavioral atoms. A behavioral atom is a small response-capable unit such as approach, withdraw, orient-left, orient-right, consume, pause, scan, press, vocalize, or attend. Each atom has a state vector. The environment presents sensory fields. The sensory fields exert forces on each behavioral atom. The forces are not fixed physical forces. They are history-dependent behavioral forces.\
The central update rule should be inspired by Verlet integration:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 x_i(t + dt) = 2*x_i(t) - x_i(t - dt) + (F_i(t) / m_i) * dt^2\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 where:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 x_i(t)        = current state of behavioral atom i\
x_i(t - dt)   = prior state of behavioral atom i\
F_i(t)        = current behavioral force acting on atom i\
m_i           = behavioral inertia, effort, or resistance-to-change parameter\
dt            = simulation timestep\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 The goal is to simulate behavior as trajectory propagation across a learned, changing stimulus-response force landscape.\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 Technical Stack\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 Use Python 3.11+.\
Use the following libraries:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 numpy\
scipy\
pandas\
matplotlib\
gymnasium\
pygame\
pydantic\
pytest\
ruff\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 Optional but desirable after the NumPy version works:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 jax\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 Do not start with PyTorch or a neural network policy. This project is not standard deep reinforcement learning. The first version should be mechanistic and inspectable.\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 Repository Structure\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 Create the following structure:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 behavioral-md/\
  README.md\
  CLAUDE.md\
  pyproject.toml\
  requirements.txt\
  src/\
    behavioral_md/\
      __init__.py\
      atoms.py\
      organism.py\
      forces.py\
      learning.py\
      environments/\
        __init__.py\
        gridworld.py\
      simulation.py\
      visualization.py\
      metrics.py\
      config.py\
  scripts/\
    run_demo.py\
    run_extinction_demo.py\
    run_generalization_demo.py\
  tests/\
    test_atoms.py\
    test_forces.py\
    test_organism.py\
    test_environment.py\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 First Milestone\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 Build a working simulation with:\
\pard\pardeftab560\pardirnatural\partightenfactor0
\ls1\ilvl0\cf0 {\listtext	0.	}A simple 2D Gymnasium environment.\
{\listtext	0.	}One organism.\
{\listtext	0.	}A distributed behavioral-atom engine.\
{\listtext	0.	}Stimulus fields for food, light, danger, and neutral cues.\
{\listtext	0.	}A force calculation system.\
{\listtext	0.	}A Verlet-style integration system.\
{\listtext	0.	}A response-selection/emission system.\
{\listtext	0.	}A learning-history update rule.\
{\listtext	0.	}A visual animation of the organism moving through the environment.\
{\listtext	0.	}Logged data showing atom activations, emitted responses, sensory inputs, reinforcer contact, and force magnitudes over time.\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 Environment Design\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 Create a custom Gymnasium environment called 
\fs24 BehavioralFieldEnv
\fs26 .\
The environment should be a 2D continuous or grid-based arena. Start with a grid because it is easier to debug.\
The environment should contain:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 organism position\
food location\
danger location\
light source location\
neutral cue location\
barriers or walls\
episode timestep\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 Observation should include sensory information available to the organism, not full omniscient state unless explicitly configured.\
Observation dictionary:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 \{\
    "position": np.ndarray shape (2,),\
    "food_vector": np.ndarray shape (2,),\
    "danger_vector": np.ndarray shape (2,),\
    "light_vector": np.ndarray shape (2,),\
    "cue_vector": np.ndarray shape (2,),\
    "food_intensity": float,\
    "danger_intensity": float,\
    "light_intensity": float,\
    "cue_intensity": float,\
    "last_consequence": float,\
\}\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 Actions should be discrete at first:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 0 = no-op\
1 = move up\
2 = move down\
3 = move left\
4 = move right\
5 = consume\
6 = freeze/pause\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 The environment should implement:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 reset(seed=None, options=None)\
step(action)\
render()\
close()\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 The 
\fs24 step()
\fs26  method should return:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 observation, reward, terminated, truncated, info\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 Rewards are not to be used as a standard RL training signal. In this framework, the reward is a consequence event that updates learning history.\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 Organism Design\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 Create an 
\fs24 Organism
\fs26  class.\
The organism owns:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 behavioral atoms\
sensory transform\
force calculator\
integrator\
learning history\
response emitter\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 The organism should expose:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 organism.reset(observation)\
organism.step(observation)\
organism.emit_action()\
organism.update_history(observation, action, consequence, info)\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 The organism should not use a policy network.\
The organism should make behavior emerge from atom activations. For example, movement action can be computed from the relative activation of movement atoms:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 move_up activation\
move_down activation\
move_left activation\
move_right activation\
pause activation\
consume activation\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 The highest activation can determine the emitted action in the first version. Later, implement probabilistic action emission using softmax over activation.\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 Behavioral Atom Design\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 Create a 
\fs24 BehavioralAtom
\fs26  class.\
Each atom should have:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 name: str\
state: np.ndarray\
previous_state: np.ndarray\
mass: float\
fatigue: float\
readiness: float\
baseline_activation: float\
sensitivity: dict[str, float]\
history_weights: dict[str, float]\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 Minimum atom set:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 approach_food\
avoid_danger\
approach_light\
orient_to_cue\
move_up\
move_down\
move_left\
move_right\
consume\
pause\
explore\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 Each atom\'92s state can start as a single activation value. Design the code so it can later support multi-dimensional state vectors.\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 Force Model\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 Create a force calculator.\
The behavioral force acting on atom 
\fs24 i
\fs26  should be:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 F_i(t) = sensory_drive_i + history_drive_i + motivational_drive_i + coupling_drive_i - fatigue_i\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 In code:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 force = (\
    sensory_drive\
    + history_drive\
    + motivational_drive\
    + coupling_drive\
    - fatigue_penalty\
)\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 Sensory Drive\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 Sensory drive should depend on current stimulus intensity and directional information.\
Example:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 sensory_drive = sensitivity_to_food * food_intensity\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 History Drive\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 History drive should depend on prior consequences.\
Example:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 history_drive = history_weight_for_food * food_intensity\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 This is the key conceptual move. The same stimulus can exert different behavioral force depending on learning history.\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 Motivational Drive\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 Add a simple motivational state variable such as hunger.\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 hunger: float\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 Hunger should increase over time and decrease after food contact. Hunger should amplify food-related force.\
Example:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 motivational_drive = hunger * food_intensity * food_history_weight\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 Coupling Drive\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 Behavioral atoms should influence one another.\
Example:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 approach_food excites move_toward_food\
avoid_danger inhibits consume\
pause inhibits movement\
explore excites movement under low stimulus control\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 Implement this as a coupling matrix:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 coupling_matrix: np.ndarray shape (n_atoms, n_atoms)\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 The coupling drive for all atoms is:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 coupling_drive = coupling_matrix @ atom_activations\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 Verlet Integration\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 Create an integrator function:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 def verlet_update(current_state, previous_state, force, mass, dt):\
    return 2 * current_state - previous_state + (force / mass) * dt**2\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 Clip activations to a reasonable range, such as:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 activation_min = -10.0\
activation_max = 10.0\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 After updating the atom state:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 previous_state = current_state\
current_state = new_state\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 Learning Rule\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 Implement a simple history update rule.\
When the organism contacts food after an action, increase the relevant history weights.\
When the organism contacts danger after an action, increase avoidance-related weights and decrease approach-related weights.\
Use an eligibility trace so recent atoms/actions are more affected than older ones.\
Eligibility trace:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 eligibility_i[t] = decay * eligibility_i[t-1] + current_activation_i\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 History update:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 history_weight_i += learning_rate * consequence * eligibility_i\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 For aversive consequences, 
\fs24 consequence
\fs26  can be negative.\
Add bounds:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 history_weight_min = -5.0\
history_weight_max = 5.0\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 Demonstration Experiments\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 Build three scripts.\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 1. Acquisition Demo\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 File:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 scripts/run_demo.py\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 Purpose:\
Show that the organism learns to approach food.\
Initial condition:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 Food exists in the arena.\
Food has weak initial control.\
Contact with food strengthens approach and consume atoms.\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 Expected output:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 Over trials, latency to food decreases.\
Approach-food activation increases.\
Movement trajectories become more direct.\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 2. Extinction Demo\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 File:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 scripts/run_extinction_demo.py\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 Purpose:\
Show extinction-like dynamics.\
Initial condition:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 Organism has already learned to approach food.\
Food is visible but no longer produces the consequence.\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 Expected output:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 Approach persists initially.\
Approach weakens over repeated non-reinforced contacts.\
Exploration or pausing increases.\
Possible extinction burst if activation temporarily increases after non-reinforcement.\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 3. Generalization Demo\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 File:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 scripts/run_generalization_demo.py\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 Purpose:\
Show stimulus generalization.\
Initial condition:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 Train with one cue value.\
Test with similar and dissimilar cue values.\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 Represent cue similarity as distance in a simple stimulus space.\
Example:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 similarity = np.exp(-beta * distance)\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 Expected output:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 Responding is strongest to the trained cue.\
Responding decreases smoothly as cue distance increases.\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 Data Logging\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 Log every timestep to a pandas DataFrame.\
Columns:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 episode\
timestep\
x\
y\
action\
reward\
terminated\
truncated\
food_intensity\
danger_intensity\
light_intensity\
cue_intensity\
hunger\
atom_name\
atom_activation\
atom_force\
atom_history_weight\
atom_fatigue\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 It is acceptable to produce long-format logs where each timestep has one row per atom.\
Save logs to:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 outputs/logs/\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 Visualization\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 Create simple visual outputs.\
Minimum visualizations:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 trajectory plot\
activation-over-time plot\
force-over-time plot\
history-weight-over-time plot\
latency-to-food across episodes\
generalization gradient\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 Save figures to:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 outputs/figures/\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 If possible, create a simple Pygame render showing:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 arena\
organism\
food\
danger\
light\
cue\
current action\
top active atom\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 Testing Requirements\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 Write tests for:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 Verlet update computes expected value.\
Atom state updates correctly.\
Force calculation returns correct shape.\
Learning update changes history weights in the expected direction.\
Environment reset returns valid observation.\
Environment step returns Gymnasium-compliant tuple.\
Organism emits a valid action.\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 Use 
\fs24 pytest
\fs26 .\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 Coding Standards\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 Use type hints.\
Use dataclasses or Pydantic models for configuration.\
Use clear modular files.\
Avoid hidden global state.\
Use random seeds for reproducibility.\
Create a 
\fs24 SimulationConfig
\fs26  object that controls:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 dt\
n_episodes\
max_steps\
learning_rate\
eligibility_decay\
activation_min\
activation_max\
history_weight_min\
history_weight_max\
grid_size\
render_mode\
seed\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 Design Principles\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 Do not build a black-box RL agent.\
Do not train a policy network.\
Do not use reward maximization language as the main explanation.\
Treat reward as consequence.\
Treat action as emitted behavior.\
Treat observation as sensory stimulation.\
Treat history weights as learning history.\
Treat atom activation as response strength or response tendency.\
Treat mass as behavioral inertia, effort, or resistance to change.\
Treat force as the current effective behavioral pull of a stimulus context given learning history.\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 README Content\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 The README should explain:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 This project models organism-environment interaction using a molecular-dynamics-inspired framework.\
The organism is a distributed dynamical system.\
Stimuli exert history-dependent behavioral forces.\
Behavior emerges through integration of many coupled response units.\
Gymnasium is used only as the environment interface.\
The organism is not a standard RL policy.\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\fs26 \cf0 Include installation instructions:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 python -m venv .venv\
source .venv/bin/activate\
pip install -r requirements.txt\
pip install -e .\
pytest\
python scripts/run_demo.py\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 Acceptance Criteria\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 The first complete version is successful when:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 The repo installs locally.\
All tests pass.\
The demo environment runs.\
The organism moves in the arena.\
The organism contacts food.\
Learning history changes over episodes.\
Latency to food decreases across acquisition episodes.\
Extinction demo shows decreasing approach after non-reinforcement.\
Generalization demo produces a smooth gradient.\
Figures and logs are saved to disk.\
Code is modular enough to extend.\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 Future Extensions\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 Do not implement these in the first pass unless the basic system already works.\
Future extensions:\
\pard\pardeftab560\slleading20\partightenfactor0

\f2\fs24 \cf0 Continuous action space\
Continuous 2D arena\
JAX vectorization\
Many organisms in parallel\
PettingZoo multi-agent version\
Response effort schedules\
Concurrent schedules\
Matching law simulations\
Stimulus equivalence / relational responding\
Verbal behavior atoms\
Multiple sensory modalities\
Hierarchical atoms: muscle-level, response-level, operant-level\
Parameter recovery\
Comparison to standard RL agents\
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 Build Order\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 Follow this order.\
First, create the package structure and configuration.\
Second, implement the gridworld Gymnasium environment.\
Third, implement 
\fs24 BehavioralAtom
\fs26 .\
Fourth, implement force calculation.\
Fifth, implement Verlet integration.\
Sixth, implement the organism.\
Seventh, connect organism and environment in a simulation loop.\
Eighth, add logging.\
Ninth, add plots.\
Tenth, add tests.\
Eleventh, add the three demo scripts.\
Do not optimize prematurely. Make the first version readable, inspectable, and scientifically interpretable.\
}