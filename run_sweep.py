# run_sweep.py
import wandb
import subprocess

# Define your sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'overall_f1', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'min': 0.00001, 'max': 0.0001, 'distribution': 'log_uniform_values'},
        'batch_size': {'values': [8, 16, 32]},
        'hidden_size': {'values': [768, 1024]},
        'dropout': {'min': 0.1, 'max': 0.3}
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="stella-absa")

# Define the training function
def train():
    run = wandb.init()
    
    # Get hyperparameters from wandb
    config = wandb.config
    
    # Run the training script with the current hyperparameters
    cmd = [
        "python", "train.py",
        "--learning_rate", str(config.learning_rate),
        "--batch_size", str(config.batch_size),
        "--hidden_size", str(config.hidden_size),
        "--dropout", str(config.dropout)
    ]
    
    subprocess.run(cmd)

# Run the sweep
wandb.agent(sweep_id, train, count=10)  # Run 10 trials