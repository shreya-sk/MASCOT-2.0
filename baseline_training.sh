#!/bin/bash

# Simple Overnight Training - All 4 GRADIENT Datasets
echo "🌙 Starting Overnight Training for All 4 Datasets"
echo "Started: $(date)"
echo "Estimated total time: ~12 hours"
echo "================================================"

# Create directories
mkdir -p experiments/baselines
mkdir -p logs

# Datasets to train
datasets=("rest14" "rest15" "rest16" "laptop14")

# Start time tracking
start_time=$(date +%s)

# Train each dataset
for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}
    current_num=$((i + 1))
    
    echo ""
    echo "🚀 [$current_num/4] Starting $dataset training"
    echo "Time: $(date)"
    echo "Expected duration: ~3 hours"
    echo "----------------------------------------"
    
    # Run training
    python train.py \
        --config research \
        --dataset $dataset \
        --num_epochs 25 \
        --output_dir experiments/baselines/$dataset \
        2>&1 | tee logs/${dataset}_training.log
    
    # Check if successful
    if [ $? -eq 0 ]; then
        echo "✅ $dataset completed successfully!"
        
        # Show quick results
        if [ -f "experiments/baselines/$dataset/training_results.json" ]; then
            echo "📊 Quick results:"
            grep -E "best_f1|aspect_f1|opinion_f1|sentiment_f1|triplet_f1" experiments/baselines/$dataset/training_results.json | head -5
        fi
    else
        echo "❌ $dataset training failed!"
        echo "Check log: logs/${dataset}_training.log"
    fi
    
    echo "[$current_num/4] $dataset done. Moving to next..."
    echo ""
done

# Calculate total time
end_time=$(date +%s)
total_seconds=$((end_time - start_time))
hours=$((total_seconds / 3600))
minutes=$(((total_seconds % 3600) / 60))

echo ""
echo "🎉 ALL TRAINING COMPLETED!"
echo "Total time: ${hours}h ${minutes}m"
echo "Finished: $(date)"
echo "========================="

# Create summary
echo ""
echo "📊 FINAL RESULTS SUMMARY:"
echo "========================="

for dataset in "${datasets[@]}"; do
    echo ""
    echo "$dataset:"
    echo "--------"
    
    if [ -f "experiments/baselines/$dataset/training_results.json" ]; then
        # Extract key metrics
        best_f1=$(grep '"best_f1"' experiments/baselines/$dataset/training_results.json | sed 's/.*: *\([0-9.]*\).*/\1/')
        echo "✅ Status: COMPLETED"
        echo "   Best F1: $best_f1"
        echo "   Model: experiments/baselines/$dataset/best_model.pt"
        echo "   Results: experiments/baselines/$dataset/training_results.json"
        echo "   Log: logs/${dataset}_training.log"
    else
        echo "❌ Status: FAILED"
        echo "   Check log: logs/${dataset}_training.log"
    fi
done

echo ""
echo "🎯 NEXT STEPS:"
echo "1. Check individual results in experiments/baselines/"
echo "2. Review training logs in logs/"
echo "3. Start cross-domain experiments"
echo "4. Run ablation studies"
echo ""
echo "✅ Ready for ACL/EMNLP 2025 experiments!"