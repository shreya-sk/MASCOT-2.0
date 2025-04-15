import subprocess
import time
import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging with timestamp in filename
timestamp = datetime.now().strftime("%d%m_%H%M")
log_file = f"logs/absa_training_results_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def run_command(cmd, description=None):
    if description:
        logging.info(f"TASK: {description}")
    
    logging.info(f"Running command: {cmd}")
    start_time = time.time()
    
    # Use Popen for real-time output
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # Line buffered
    )
    
    # Process output in real-time
    for line in process.stdout:
        line = line.strip()
        if line:
            logging.info(f"  {line}")
            print(line)
    
    # Wait for process to complete
    process.wait()
    
    duration = time.time() - start_time
    duration_str = f"{duration:.2f} seconds" if duration < 300 else f"{duration/60:.2f} minutes"
    
    logging.info(f"Command completed in {duration_str} with status {process.returncode}")
    return process.returncode == 0
# =====================================================================
# BLOCK 1: MEMORY CONSTRAINED MODEL (Commented out by default)
# =====================================================================
# This is the most memory-efficient configuration (recommended for 8GB machines)
# Uncomment the following lines to run this block

"""
logging.info("BLOCK 1: MEMORY CONSTRAINED MODEL (ULTRA-LIGHTWEIGHT)")
logging.info("This is the most memory-efficient configuration, recommended for 8GB machines")

run_command(
    "python train.py --dataset rest15 --config memory_constrained --device cpu --single_phase",
    "Training ultra-lightweight model (memory constrained)"
)

run_command(
    "python evaluate.py --model checkpoints/ultra-lightweight-absa_rest15_single_best.pt --dataset rest15 --mode all",
    "Evaluating ultra-lightweight model"
)

run_command(
    "python dia.py --model checkpoints/ultra-lightweight-absa_rest15_single_best.pt",
    "Running diagnostics on ultra-lightweight model"
)

run_command(
    "python predict.py --model checkpoints/ultra-lightweight-absa_rest15_single_best.pt --text 'The food was delicious but the service was slow.'",
    "Testing ultra-lightweight model on sample sentence"
)
"""

# =====================================================================
# BLOCK 2: SINGLE PHASE DEFAULT MODEL (Active by default)
# =====================================================================
# This configuration balances performance and memory usage

logging.info("BLOCK 2: DEFAULT MODEL WITH SINGLE PHASE (BALANCED)")
logging.info("This configuration balances performance and memory usage")

run_command(
    "python train.py --dataset rest15 --device cpu --single_phase",
    "Training balanced model with single phase"
)

run_command(
    "python evaluate.py --model checkpoints/improved-absa_rest15_single_best.pt --dataset rest15 --mode all",
    "Evaluating balanced model"
)

run_command(
    "python dia.py --model checkpoints/improved-absa_rest15_single_best.pt",
    "Running diagnostics on balanced model"
)

run_command(
    "python predict.py --model checkpoints/improved-absa_rest15_single_best.pt --text 'The food was delicious but the service was slow.'",
    "Testing balanced model on sample sentence"
)

# =====================================================================
# BLOCK 3: DEBERTA MODEL WITH TWO-PHASE TRAINING (Commented out by default)
# =====================================================================
# This configuration may exceed memory limits on 8GB machines
# Uncomment the following lines to run this block

"""
logging.info("BLOCK 3: DEBERTA MODEL WITH TWO-PHASE TRAINING (HIGHEST PERFORMANCE)")
logging.info("This configuration may exceed memory limits on 8GB machines")

run_command(
    "python train.py --dataset rest15 --model microsoft/deberta-v3-small --batch_size 4 --gradient_accumulation_steps 8 --device cpu",
    "Training DeBERTa model with two-phase approach"
)

run_command(
    "python evaluate.py --model checkpoints/improved-absa_rest15_extraction_best.pt --dataset rest15 --mode all",
    "Evaluating DeBERTa model (extraction phase)"
)

run_command(
    "python evaluate.py --model checkpoints/improved-absa_rest15_generation_best.pt --dataset rest15 --mode all",
    "Evaluating DeBERTa model (generation phase)"
)

run_command(
    "python dia.py --model checkpoints/improved-absa_rest15_generation_best.pt",
    "Running diagnostics on DeBERTa model"
)

run_command(
    "python predict.py --model checkpoints/improved-absa_rest15_generation_best.pt --text 'The food was delicious but the service was slow.'",
    "Testing DeBERTa model on sample sentence"
)
"""

# =====================================================================
# BLOCK 4: TESTING ON DIFFERENT DATASETS (Commented out by default)
# =====================================================================
# Run predictions on different datasets to compare performance
# Uncomment the following lines to run this block

"""
logging.info("BLOCK 4: TESTING ON DIFFERENT DATASETS")
logging.info("Running predictions on different datasets to compare performance")

# Use your best model here
best_model = "checkpoints/improved-absa_rest15_single_best.pt"

run_command(
    f"python evaluate.py --model {best_model} --dataset rest14 --mode all",
    "Evaluating model on REST14 dataset"
)

run_command(
    f"python evaluate.py --model {best_model} --dataset rest16 --mode all",
    "Evaluating model on REST16 dataset"
)

run_command(
    f"python evaluate.py --model {best_model} --dataset laptop14 --mode all",
    "Evaluating model on LAPTOP14 dataset"
)
"""

logging.info("ALL TRAINING AND EVALUATION COMPLETED")
logging.info(f"Full log is available in {log_file}")