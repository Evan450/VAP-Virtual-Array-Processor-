Virtual Array Processor Quick-Start Guide

Note: For more details on the script’s features and inner workings, please refer the Manual (README), and that this quick-start guide assumes that you already know how to clone and/or download this script off-of GitHub.

1. Prerequisites

Python 3: Ensure you have Python 3 installed.
Dependencies: Install required packages using pip. For example:

pip install numpy pyyaml psutil

If you plan to use GPU processing, also install:

pip install cupy GPUtil aiofiles

Optional: If you’re not using GPU features, the script will fall back to CPU computations.

2. Running the Script

Direct Execution:

From the command line, run:

python VAP.py

This will start the interactive menu.

3. Using the Interactive Menus

Main Menu:

Start Computation: Runs all tasks and displays a few results.

Settings: Adjust parameters such as number of tasks, tasks per chunk, compute mode (CPU, GPU, or hybrid), output file, and task complexity.

Manage Tasks: Add, remove, or modify tasks interactively.

Auto-Generation:

If no tasks are present, VAP automatically generates a set of tasks based on the configuration (default: 50,000 tasks in mixed complexity).

4. Configuration Files

Configuration:

The script uses a YAML file (vap_config.yaml) for settings. If the file exists, it loads your preferences; otherwise, it uses default values.

Checkpointing & Caching:

Processed tasks and results are saved in the checkpoints and cache directories respectively. This helps resume interrupted computations.

5. Getting Results

After computation, the results are displayed and (if enabled) written to an output file (default: Output.txt).
