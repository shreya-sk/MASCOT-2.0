# train.py
"""
Enhanced Training Script with Complete Few-Shot Learning Implementation
2024-2025 Breakthrough Features Integration
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
import logging
import os
from datetime import datetime
from tqdm import tqdm

# Import your existing components
from src.utils.config import LLMABSAConfig, create_high_performance_config
from src.data.dataset import ABSADataset
from src.data.preprocessor import ABSAPreprocessor
from src.models.enhanced_absa_model import EnhancedABSAModel, create_enhanced_absa_model
from src.training.few_shot_trainer import FewShotABSATrainer, FewShotDatasetAdapter
from src.training.trainer import ABSATrainer
from src.training.metrics import ABSAMetrics
from src.utils.logger import setup_logger

from transformers import AutoTokenizer, get_linear_schedule_with_warmup


def parse_arguments():
    """Parse command line arguments with few-shot learning options"""
    parser = argparse.ArgumentParser(description='Enhanced ABSA Training with Few-Shot Learning')
    
    # Dataset and model arguments
    parser.add_argument('--dataset', type=str, default='rest15', 
                       choices=['rest14', 'rest15', 'rest16', 'laptop14', 'mams'],
                       help='Dataset to use for training')
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-base',
                       help='Pretrained model to use')
    parser.add_argument('--config_type', type=str, default='high_performance',
                       choices=['balanced', 'high_performance', 'memory_constrained', 'publication_ready'],
                       help='Configuration preset to use')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length')
    
    # 2024-2025 breakthrough features
    parser.add_argument('--use_contrastive', action='store_true', default=True,
                       help='Enable contrastive learning')
    parser.add_argument('--use_few_shot', action='store_true', default=True,
                       help='Enable few-shot learning (NEW)')
    parser.add_argument('--use_implicit', action='store_true', default=True,
                       help='Enable implicit detection')
    parser.add_argument('--use_instruction', action='store_true', default=True,
                       help='Enable instruction following')
    
    # Few-shot specific arguments (NEW)
    parser.add_argument('--few_shot_k', type=int, default=5,
                       help='K-shot for few-shot learning')
    parser.add_argument('--episodes_per_epoch', type=int, default=100,
                       help='Number of few-shot episodes per epoch')
    parser.add_argument('--adaptation_steps', type=int, default=5,
                       help='Domain adaptation steps')
    parser.add_argument('--meta_learning_rate', type=float, default=0.01,
                       help='Meta-learning rate for few-shot learning')
    
    # Few-shot method selection
    parser.add_argument('--use_drp', action='store_true', default=True,
                       help='Enable Dual Relations Propagation')
    parser.add_argument('--use_afml', action='store_true', default=True,
                       help='Enable Aspect-Focused Meta-Learning')
    parser.add_argument('--use_cd_alphn', action='store_true', default=True,
                       help='Enable Cross-Domain Aspect Label Propagation')
    parser.add_argument('--use_ipt', action='store_true', default=True,
                       help='Enable Instruction Prompt-based Few-Shot')
    
    # Training modes
    parser.add_argument('--training_mode', type=str, default='hybrid',
                       choices=['standard', 'few_shot_only', 'hybrid'],
                       help='Training mode: standard, few-shot only, or hybrid')
    parser.add_argument('--domain_adaptation', action='store_true',
                       help='Enable cross-domain adaptation training')
    
    # Advanced options
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (fewer epochs, smaller data)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--eval_every', type=int, default=1,
                       help='Evaluate every N epochs')
    
    # Hardware optimization
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Use gradient checkpointing to save memory')
    
    return parser.parse_args()


def create_config_from_args(args):
    """Create configuration from arguments"""
    # Start with preset configuration
    if args.config_type == 'high_performance':
        config = create_high_performance_config()
    else:
        config = LLMABSAConfig()
    
    # Update with command line arguments
    config.model_name = args.model_name
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.learning_rate
    config.max_seq_length = args.max_length
    
    # 2024-2025 breakthrough features
    config.use_contrastive_learning = args.use_contrastive
    config.use_few_shot_learning = args.use_few_shot
    config.use_implicit_detection = args.use_implicit
    config.use_instruction_following = args.use_instruction
    
    # Few-shot specific configuration (NEW)
    config.few_shot_k = args.few_shot_k
    config.episodes_per_epoch = args.episodes_per_epoch
    config.adaptation_steps = args.adaptation_steps
    config.meta_learning_rate = args.meta_learning_rate
    
    # Few-shot method selection
    config.use_drp = args.use_drp
    config.use_afml = args.use_afml
    config.use_cd_alphn = args.use_cd_alphn
    config.use_ipt = args.use_ipt
    
    # Hardware optimization
    config.use_fp16 = args.mixed_precision
    config.use_gradient_checkpointing = args.gradient_checkpointing
    
    # Debug mode adjustments
    if args.debug:
        config.num_epochs = min(3, config.num_epochs)
        config.episodes_per_epoch = min(20, config.episodes_per_epoch)
        config.debug_mode = True
        print("🐛 Debug mode enabled - reduced epochs and episodes")
    
    # Set experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = args.training_mode
    config.experiment_name = f"enhanced_absa_{args.dataset}_{mode_suffix}_{timestamp}"
    
    return config


def load_datasets(config, tokenizer, args):
    """Load training, validation, and test datasets"""
    print(f"📊 Loading {args.dataset} dataset...")
    
    # Initialize preprocessor
    preprocessor = ABSAPreprocessor(
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        use_instruction_following=config.use_instruction_following
    )
    
    datasets = {}
    dataloaders = {}
    
    # Load standard datasets
    for split in ['train', 'dev', 'test']:
        dataset_path = f"Dataset/aste/{args.dataset}/{split}.txt"
        
        if os.path.exists(dataset_path):
            dataset = ABSADataset(
                file_path=dataset_path,
                preprocessor=preprocessor,
                config=config
            )
            datasets[split] = dataset
            
            # Create dataloaders
            batch_size = config.batch_size if split == 'train' else config.batch_size * 2
            shuffle = split == 'train'
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=2,
                pin_memory=True
            )
            dataloaders[split] = dataloader
            
            print(f"   {split}: {len(dataset)} samples")
        else:
            print(f"   Warning: {dataset_path} not found")
    
    return datasets, dataloaders


def setup_training_environment(args):
    """Setup training environment and logging"""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/enhanced_absa_{args.dataset}_{args.training_mode}_{timestamp}.log"
    logger = setup_logger(log_file)
    
    return device, logger


def create_few_shot_datasets(datasets, config):
    """Convert standard datasets to few-shot format"""
    print("🔄 Converting datasets for few-shot learning...")
    
    few_shot_datasets = {}
    
    for split, dataset in datasets.items():
        adapter = FewShotDatasetAdapter(dataset, config)
        few_shot_dataset = adapter.convert_to_few_shot_format()
        few_shot_datasets[split] = few_shot_dataset
        
        print(f"   {split}: {few_shot_dataset.features.size(0)} samples converted")
    
    return few_shot_datasets


def train_standard_mode(model, dataloaders, config, device, logger):
    """Train using standard ABSA approach"""
    print("🎯 Training in Standard Mode...")
    
    trainer = ABSATrainer(
        model=model,
        config=config,
        device=device,
        logger=logger
    )
    
    best_model = trainer.train(
        train_dataloader=dataloaders['train'],
        val_dataloader=dataloaders.get('dev'),
        test_dataloader=dataloaders.get('test')
    )
    
    return best_model


def train_few_shot_mode(model, few_shot_datasets, config, tokenizer, device, logger):
    """Train using few-shot learning approach"""
    print("🎯 Training in Few-Shot Mode...")
    
    # Initialize few-shot trainer
    few_shot_trainer = FewShotABSATrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    # Training loop
    best_metrics = {'f1_mean': 0.0}
    
    for epoch in range(config.num_epochs):
        print(f"\n📚 Epoch {epoch+1}/{config.num_epochs}")
        
        # Train epoch
        train_metrics = few_shot_trainer.train_few_shot_epoch(
            train_dataset=few_shot_datasets['train'],
            val_dataset=few_shot_datasets.get('dev')
        )
        
        # Evaluation
        if 'dev' in few_shot_datasets:
            val_metrics = few_shot_trainer.evaluate_few_shot(
                few_shot_datasets['dev'], 
                num_episodes=50
            )
            
            print(f"Validation: F1={val_metrics['f1_mean']:.4f}±{val_metrics['f1_std']:.4f}")
            
            # Save best model
            if val_metrics['f1_mean'] > best_metrics['f1_mean']:
                best_metrics = val_metrics
                checkpoint_path = f"checkpoints/{config.experiment_name}_best_few_shot.pt"
                few_shot_trainer.save_few_shot_model(checkpoint_path)
                print(f"💾 New best model saved: F1={val_metrics['f1_mean']:.4f}")
        
        # Test evaluation
        if 'test' in few_shot_datasets and (epoch + 1) % config.eval_interval == 0:
            test_metrics = few_shot_trainer.evaluate_few_shot(
                few_shot_datasets['test'], 
                num_episodes=100
            )
            print(f"Test: F1={test_metrics['f1_mean']:.4f}±{test_metrics['f1_std']:.4f}")
    
    return model


def train_hybrid_mode(model, dataloaders, few_shot_datasets, config, tokenizer, device, logger):
    """Train using hybrid approach (standard + few-shot)"""
    print("🎯 Training in Hybrid Mode...")
    
    # Phase 1: Standard training with few-shot integration
    print("\n📚 Phase 1: Hybrid Training")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=len(dataloaders['train']) * config.num_epochs
    )
    
    best_f1 = 0.0
    
    for epoch in range(config.num_epochs):
        print(f"\n📚 Epoch {epoch+1}/{config.num_epochs}")
        
        # Training phase
        model.train()
        epoch_losses = []
        
        # Prepare few-shot support data for this epoch
        support_data, _, domain_ids = _sample_few_shot_support(few_shot_datasets['train'], config)
        
        pbar = tqdm(dataloaders['train'], desc="Training")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with few-shot support
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                few_shot_support_data=support_data,
                domain_ids=domain_ids[:batch['input_ids'].size(0)],  # Match batch size
                training=True
            )
            
            loss = outputs['loss']
            
            # Backward pass
            if config.use_fp16:
                # Mixed precision training
                from torch.cuda.amp import autocast, GradScaler
                scaler = GradScaler()
                with autocast():
                    loss.backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            
            epoch_losses.append(loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Avg': f"{np.mean(epoch_losses):.4f}",
                'LR': f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        # Validation
        if 'dev' in dataloaders:
            val_metrics = evaluate_hybrid_model(
                model, dataloaders['dev'], few_shot_datasets['dev'], 
                config, device
            )
            
            val_f1 = val_metrics['fusion_f1']
            print(f"Validation F1: {val_f1:.4f}")
            print(f"Standard F1: {val_metrics['standard_f1']:.4f}")
            print(f"Few-shot F1: {val_metrics['few_shot_f1']:.4f}")
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                checkpoint_path = f"checkpoints/{config.experiment_name}_best_hybrid.pt"
                model.save_enhanced_model(checkpoint_path)
                print(f"💾 New best hybrid model saved: F1={val_f1:.4f}")
        
        # Test evaluation
        if 'test' in dataloaders and (epoch + 1) % config.eval_interval == 0:
            test_metrics = evaluate_hybrid_model(
                model, dataloaders['test'], few_shot_datasets['test'], 
                config, device
            )
            print(f"Test F1: {test_metrics['fusion_f1']:.4f}")
    
    return model


def _sample_few_shot_support(few_shot_dataset, config):
    """Sample support data for few-shot learning"""
    support_data, _ = few_shot_dataset.sample_episode(
        k_shot=config.few_shot_k,
        num_query=5
    )
    
    # Simulate domain IDs
    domain_ids = torch.randint(0, 3, (support_data['features'].size(0) + 100,))
    
    return support_data, _, domain_ids


def evaluate_hybrid_model(model, dataloader, few_shot_dataset, config, device):
    """Evaluate hybrid model performance"""
    model.eval()
    
    all_standard_preds = []
    all_few_shot_preds = []
    all_fusion_preds = []
    all_labels = []
    
    # Sample support data for evaluation
    support_data, _, domain_ids = _sample_few_shot_support(few_shot_dataset, config)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                few_shot_support_data=support_data,
                domain_ids=domain_ids[:batch['input_ids'].size(0)],
                training=False
            )
            
            # Collect predictions
            all_standard_preds.extend(outputs['standard_predictions'].argmax(dim=-1).cpu().numpy())
            all_fusion_preds.extend(outputs['predictions'].argmax(dim=-1).cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            
            if outputs['few_shot_predictions'] is not None:
                all_few_shot_preds.extend(outputs['few_shot_predictions'].argmax(dim=-1).cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import f1_score, accuracy_score
    
    metrics = {
        'standard_f1': f1_score(all_labels, all_standard_preds, average='macro'),
        'standard_accuracy': accuracy_score(all_labels, all_standard_preds),
        'fusion_f1': f1_score(all_labels, all_fusion_preds, average='macro'),
        'fusion_accuracy': accuracy_score(all_labels, all_fusion_preds)
    }
    
    if all_few_shot_preds:
        metrics.update({
            'few_shot_f1': f1_score(all_labels, all_few_shot_preds, average='macro'),
            'few_shot_accuracy': accuracy_score(all_labels, all_few_shot_preds)
        })
    
    return metrics


def train_with_domain_adaptation(model, datasets, config, tokenizer, device, logger):
    """Train with cross-domain adaptation"""
    print("🌍 Training with Domain Adaptation...")
    
    # Use first dataset as source, others as targets
    dataset_names = list(datasets.keys())
    source_dataset = datasets[dataset_names[0]]
    target_datasets = [datasets[name] for name in dataset_names[1:]]
    
    few_shot_trainer = FewShotABSATrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    # Convert to few-shot format
    source_few_shot = FewShotDatasetAdapter(source_dataset, config).convert_to_few_shot_format()
    target_few_shot = [
        FewShotDatasetAdapter(target, config).convert_to_few_shot_format()
        for target in target_datasets
    ]
    
    # Domain adaptation training
    adapted_model = few_shot_trainer.train_with_domain_adaptation(
        source_dataset=source_few_shot,
        target_datasets=target_few_shot,
        epochs=config.num_epochs
    )
    
    return adapted_model


def main():
    """Main training function with few-shot learning integration"""
    print("="*80)
    print("🚀 ENHANCED ABSA TRAINING WITH FEW-SHOT LEARNING")
    print("="*80)
    
    # Parse arguments
    args = parse_arguments()
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Setup environment
    device, logger = setup_training_environment(args)
    
    # Print configuration
    print(f"\n📋 Training Configuration:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Model: {config.model_name}")
    print(f"   Training Mode: {args.training_mode}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Learning Rate: {config.learning_rate}")
    
    print(f"\n🎯 Enabled Features:")
    print(f"   ✅ Contrastive Learning: {config.use_contrastive_learning}")
    print(f"   {'✅' if config.use_few_shot_learning else '❌'} Few-Shot Learning: {config.use_few_shot_learning}")
    print(f"   {'✅' if config.use_implicit_detection else '❌'} Implicit Detection: {config.use_implicit_detection}")
    print(f"   {'✅' if config.use_instruction_following else '❌'} Instruction Following: {config.use_instruction_following}")
    
    if config.use_few_shot_learning:
        print(f"\n🔬 Few-Shot Methods:")
        print(f"   {'✅' if config.use_drp else '❌'} DRP (Dual Relations Propagation): {config.use_drp}")
        print(f"   {'✅' if config.use_afml else '❌'} AFML (Aspect-Focused Meta-Learning): {config.use_afml}")
        print(f"   {'✅' if config.use_cd_alphn else '❌'} CD-ALPHN (Cross-Domain Propagation): {config.use_cd_alphn}")
        print(f"   {'✅' if config.use_ipt else '❌'} IPT (Instruction Prompt Few-Shot): {config.use_ipt}")
    
    # Load tokenizer
    print(f"\n🔤 Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    datasets, dataloaders = load_datasets(config, tokenizer, args)
    
    if not datasets:
        print("❌ No datasets loaded. Please check dataset paths.")
        return
    
    # Create enhanced model
    print(f"\n🏗️ Creating Enhanced ABSA Model...")
    model = create_enhanced_absa_model(config, device)
    
    # Convert datasets for few-shot learning if needed
    few_shot_datasets = {}
    if config.use_few_shot_learning and args.training_mode in ['few_shot_only', 'hybrid']:
        few_shot_datasets = create_few_shot_datasets(datasets, config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"📥 Resuming from checkpoint: {args.resume}")
        try:
            model = EnhancedABSAModel.load_enhanced_model(args.resume, config, device)
            print("✅ Checkpoint loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return
    
    # Training based on mode
    try:
        if args.training_mode == 'standard':
            trained_model = train_standard_mode(model, dataloaders, config, device, logger)
        
        elif args.training_mode == 'few_shot_only':
            trained_model = train_few_shot_mode(
                model, few_shot_datasets, config, tokenizer, device, logger
            )
        
        elif args.training_mode == 'hybrid':
            trained_model = train_hybrid_mode(
                model, dataloaders, few_shot_datasets, config, tokenizer, device, logger
            )
        
        # Domain adaptation training
        if args.domain_adaptation:
            print("\n🌍 Starting Domain Adaptation Training...")
            # For domain adaptation, load multiple datasets
            multi_datasets = {}
            for dataset_name in ['rest14', 'rest15', 'laptop14']:
                try:
                    dataset_path = f"Dataset/aste/{dataset_name}/train.txt"
                    if os.path.exists(dataset_path):
                        preprocessor = ABSAPreprocessor(tokenizer, config.max_seq_length)
                        dataset = ABSADataset(dataset_path, preprocessor, config)
                        multi_datasets[dataset_name] = dataset
                except Exception as e:
                    print(f"Warning: Could not load {dataset_name}: {e}")
            
            if len(multi_datasets) > 1:
                trained_model = train_with_domain_adaptation(
                    trained_model, multi_datasets, config, tokenizer, device, logger
                )
        
        # Final evaluation and save
        print("\n🎯 Final Evaluation...")
        if 'test' in dataloaders:
            if args.training_mode == 'hybrid' and few_shot_datasets:
                final_metrics = evaluate_hybrid_model(
                    trained_model, dataloaders['test'], 
                    few_shot_datasets['test'], config, device
                )
                print(f"Final Test Results:")
                print(f"   Standard F1: {final_metrics['standard_f1']:.4f}")
                print(f"   Few-Shot F1: {final_metrics.get('few_shot_f1', 'N/A')}")
                print(f"   Fusion F1: {final_metrics['fusion_f1']:.4f}")
            else:
                # Standard evaluation
                test_metrics = ABSAMetrics.evaluate_model(
                    trained_model, dataloaders['test'], device
                )
                print(f"Final Test F1: {test_metrics['macro_f1']:.4f}")
        
        # Save final model
        final_checkpoint_path = f"checkpoints/{config.experiment_name}_final.pt"
        trained_model.save_enhanced_model(final_checkpoint_path)
        print(f"💾 Final model saved to: {final_checkpoint_path}")
        
        # Print performance summary
        print("\n📊 Performance Summary:")
        performance_metrics = trained_model.get_performance_metrics()
        for component, improvement in performance_metrics['expected_improvements'].items():
            print(f"   📈 {component}: {improvement}")
        
        print(f"\n🎉 Training completed successfully!")
        print(f"   Experiment: {config.experiment_name}")
        print(f"   Best model: {final_checkpoint_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()