Virtual Array Processor Manual

Note: For quick setup and some execution instructions, see the Quick-Start Guide.

1. Introduction

Virtual Array Processor (VAP) v5.4b is a Python script designed to process a large array of arithmetic tasks. It supports both CPU and GPU computations (when available) and includes: error handling, checkpointing, logging, and resource monitoring. This script is ideal for both quick computations and detailed analysis of processing performance.

2. Installation and Setup

2.1 Prerequisites

Python 3: The script runs on Python 3.

Required Libraries:
Core: numpy, pyyaml, psutil
Async File I/O: aiofiles 
(optional; if not installed, caching/checkpointing remains functional but asynchronous file operations won’t be used)

GPU Support: cupy, GPUtil (if GPU processing is desired)

2.2 Installation Steps

Install Python 3 and up.

Install the dependencies:

pip install numpy pyyaml psutil

pip install aiofiles (Optional for async file operations)

pip install cupy GPUtil (Optional for GPU processing)

Download VAP.py: Place it in your working directory.

(Optional) Configure your environment:
Create or edit vap_config.yaml if you wish to override default settings.

3. Script Architecture and Features

3.1 Configuration Management

VAPConfig Data Class:
Holds parameters such as:

number_of_tasks: Total tasks to process.

tasks_per_chunk: Tasks processed in one batch.

compute_mode: Options include “CPU”, “GPU”, or “CPU+GPU” (hybrid).

Other settings include log directories, checkpoint intervals, and GPU memory limits.

ConfigManager:
Loads and saves configurations to vap_config.yaml.

3.2 Logging Setup

LogManager:
Sets up a logger with:

Error and Info Handlers: Uses rotating file handlers to manage log sizes.

Console Handler: Outputs warnings and higher-level messages to the screen.

Log Files:

Stored in the logs directory (e.g., vap_error.log, vap_info.log).

3.3 Task Management

Task Definition:
Each task is an arithmetic operation (e.g., add, sub, mul, etc.) defined in the Task data class.

Task Generation:
Tasks can be generated automatically based on a selected complexity mode (“simple”, “mixed”, “complex”).

Task Cache & Checkpointer:
TaskCache: Caches results to avoid recomputation.

Checkpointer: Saves processed chunks to enable recovery in case of errors or interruptions.

3.4 Resource Monitoring and Progress Tracking

ResourceMonitor:
Continuously monitors CPU, memory, and (if available) GPU usage in a background thread. It calculates metrics such as mean, max, min, and standard deviation.

ProgressTracker:
Displays live progress updates (tasks processed, elapsed time, and ETA).

3.5 Arithmetic Operations and Processing Engine

CPU Vectorized Operations:
Uses NumPy and Python’s math functions to process operations on arrays.

GPU Vectorized Operations:
Utilizes CuPy for efficient processing when GPU is available. For operations not supported on the GPU (or when in CPU-only mode), the script defaults to CPU processing.

ProcessingEngine:
Splits tasks into chunks based on available system resources.

Processes chunks asynchronously, with support for retries in case of errors.

Offers three modes:
CPU-only

GPU-only

Hybrid (CPU+GPU)

Integrates checkpointing and result caching.

3.6 User Interface and Controller

VAPController:
Bridges the configuration, task management, and processing engine. Provides methods to update settings, manage tasks, and start computations.

Interactive Menus:

The script offers several menus:
Main Menu: Allows you to: Start computation, edit settings, or for managing tasks.

Settings Menu: Adjust parameters such as: the number of tasks, tasks per chunk, compute mode, output file, and task complexity.

Task Management Menu: Add, remove, modify, or display tasks.

4. Running the Script: Detailed Workflow

Initialization:
The script starts in main(), where it loads configurations and sets up logging.

It attempts to restore a previous checkpoint and automatiaclly generates tasks if none exist.

Main Menu Interaction:
Choose Start Computation to process tasks.

Adjust settings or manage tasks via the interactive menus.

Computation Phase:
The ProcessingEngine divides tasks into chunks, monitors resources, and processes each chunk concurrently.

Errors are handled with retries and logging (and if nessasary, a shutdown of the script).

Output:
Computation results are shown in the terminal.

If an output file is set, results are written to that file (CSV format).

5. Troubleshooting and Tips

GPU Not Available:
If GPU libraries aren’t installed or no GPU is detected, select CPU mode in the settings.

Checkpoints & Caching:
In the event of an unexpected shutdown, check the checkpoints and cache directories to resume or review processed tasks.

Logging:
Review log files in the logs directory for detailed error messages if something goes wrong.

Modifying Configurations:
You can modify the vap_config.yaml manually to override defaults or change performance-related settings.

6. Additional References

Source Code:
Refer to the inline comments in VAP.py for further insights into each module and function.

Final Notes:
Versioning: This manual applies to VAP version 5.4b. Future updates might introduce new features or modifications, please note that I will attempt to keep this manual as up-to-date as possible though, I am only a one-man-army, so please do not expect it to be soon.

Support & Contributions: Contributions, bug reports, and/or feature requests are welcome. Please open an issue or submit a pull request on GitHub.
