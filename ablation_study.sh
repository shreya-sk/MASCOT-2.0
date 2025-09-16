#!/bin/bash

# Simple Ablation Study - GRADIENT Components
echo "🔬 Starting Ablation Study for ACL/EMNLP 2025"
echo "Started: $(date)"
echo "Estimated total time: ~6 hours"
echo "=============================================="

# Create directories
mkdir -p experiments/ablation
mkdir -p logs/ablation

# Ablation configurations to test
configs=(
    "baseline:No gradient reversal, no orthogonal constraints"
    "gradient_only:Only gradient reversal enabled"
    "orthogonal_only:Only orthogonal constraints enabled"
    "no_implicit:Full model without implicit detection"
    "no_fusion:Full model without multi-granularity fusion"
    "full_model:Complete GRADIENT model"
)

# Start time tracking
start_time=$(date +%s)

# Train each ablation configuration
for i in "${!configs[@]}"; do
    config_info=${configs[$i]}
    config_name="${config_info%%:*}"
    config_desc="${config_info##*:}"
    current_num=$((i + 1))
    total=${#configs[@]}
    
    echo ""
    echo "🚀 [$current_num/$total] Training: $config_name"
    echo "Description: $config_desc"
    echo "Time: $(date)"
    echo "Expected duration: ~1 hour"
    echo "----------------------------------------"
    
    # Create simple config override
    export ABLATION_CONFIG=$config_name
    
    # Run training with ablation config
    python train.py \
        --config research \
        --dataset rest14 \
        --num_epochs 15 \
        --output_dir experiments/ablation/$config_name \
        2>&1 | tee logs/ablation/${config_name}_training.log
    
    # Check if successful
    if [ $? -eq 0 ]; then
        echo "✅ $config_name completed successfully!"
        
        # Show quick results if available
        if [ -f "experiments/ablation/$config_name/training_results.json" ]; then
            echo "📊 Quick results:"
            grep -E "best_f1|triplet_f1" experiments/ablation/$config_name/training_results.json | head -2
        fi
    else
        echo "❌ $config_name training failed!"
        echo "Check log: logs/ablation/${config_name}_training.log"
    fi
    
    echo "[$current_num/$total] $config_name done. Moving to next..."
    echo ""
done

# Now test cross-domain performance
echo ""
echo "🎯 Testing Cross-Domain Performance"
echo "=================================="

target_datasets=("laptop14" "rest15" "rest16")

for config_info in "${configs[@]}"; do
    config_name="${config_info%%:*}"
    model_path="experiments/ablation/$config_name/best_model.pt"
    
    # Only test if model exists
    if [ -f "$model_path" ]; then
        echo ""
        echo "Testing $config_name on cross-domain datasets..."
        
        for target in "${target_datasets[@]}"; do
            echo "  🎯 $config_name → $target"
            
            python test.py \
                --model_path "$model_path" \
                --dataset "$target" \
                --output_dir "experiments/ablation/$config_name/eval_$target" \
                --confidence 0.6 \
                2>&1 | tee logs/ablation/${config_name}_${target}_test.log
            
            if [ $? -eq 0 ]; then
                # Extract triplet F1 for quick comparison
                if [ -f "experiments/ablation/$config_name/eval_$target/test_results_$target.json" ]; then
                    triplet_f1=$(grep '"triplet_f1"' "experiments/ablation/$config_name/eval_$target/test_results_$target.json" | sed 's/.*: *\([0-9.]*\).*/\1/')
                    echo "    ✅ Triplet F1: $triplet_f1"
                fi
            else
                echo "    ❌ Testing failed"
            fi
        done
    else
        echo "⚠️ Skipping $config_name (model not found)"
    fi
done

# Calculate total time
end_time=$(date +%s)
total_seconds=$((end_time - start_time))
hours=$((total_seconds / 3600))
minutes=$(((total_seconds % 3600) / 60))

echo ""
echo "🎉 ABLATION STUDY COMPLETED!"
echo "Total time: ${hours}h ${minutes}m"
echo "Finished: $(date)"
echo "=========================="

# Create results summary
echo ""
echo "📊 ABLATION RESULTS SUMMARY:"
echo "============================"

# Create CSV header
echo "Configuration,Component_Description,Rest14_F1,Laptop14_F1,Rest15_F1,Rest16_F1,Avg_CrossDomain_F1" > results/ablation_summary.csv

for config_info in "${configs[@]}"; do
    config_name="${config_info%%:*}"
    config_desc="${config_info##*:}"
    
    echo ""
    echo "$config_name ($config_desc):"
    echo "$(printf '%.0s-' {1..50})"
    
    # Get source domain performance (rest14)
    rest14_f1="N/A"
    if [ -f "experiments/ablation/$config_name/training_results.json" ]; then
        rest14_f1=$(grep '"best_f1"' "experiments/ablation/$config_name/training_results.json" | sed 's/.*: *\([0-9.]*\).*/\1/' || echo "N/A")
        echo "✅ Source (rest14): $rest14_f1"
    else
        echo "❌ Source (rest14): FAILED"
    fi
    
    # Get cross-domain performance
    laptop14_f1="N/A"
    rest15_f1="N/A" 
    rest16_f1="N/A"
    
    if [ -f "experiments/ablation/$config_name/eval_laptop14/test_results_laptop14.json" ]; then
        laptop14_f1=$(grep '"triplet_f1"' "experiments/ablation/$config_name/eval_laptop14/test_results_laptop14.json" | sed 's/.*: *\([0-9.]*\).*/\1/' || echo "N/A")
        echo "   → laptop14: $laptop14_f1"
    else
        echo "   → laptop14: FAILED"
    fi
    
    if [ -f "experiments/ablation/$config_name/eval_rest15/test_results_rest15.json" ]; then
        rest15_f1=$(grep '"triplet_f1"' "experiments/ablation/$config_name/eval_rest15/test_results_rest15.json" | sed 's/.*: *\([0-9.]*\).*/\1/' || echo "N/A")
        echo "   → rest15: $rest15_f1"
    else
        echo "   → rest15: FAILED"
    fi
    
    if [ -f "experiments/ablation/$config_name/eval_rest16/test_results_rest16.json" ]; then
        rest16_f1=$(grep '"triplet_f1"' "experiments/ablation/$config_name/eval_rest16/test_results_rest16.json" | sed 's/.*: *\([0-9.]*\).*/\1/' || echo "N/A")
        echo "   → rest16: $rest16_f1"
    else
        echo "   → rest16: FAILED"
    fi
    
    # Calculate average cross-domain F1 (if all values are numeric)
    avg_f1="N/A"
    if [[ "$laptop14_f1" =~ ^[0-9.]+$ ]] && [[ "$rest15_f1" =~ ^[0-9.]+$ ]] && [[ "$rest16_f1" =~ ^[0-9.]+$ ]]; then
        avg_f1=$(echo "scale=4; ($laptop14_f1 + $rest15_f1 + $rest16_f1) / 3" | bc -l 2>/dev/null || echo "N/A")
    fi
    
    echo "   Average Cross-Domain: $avg_f1"
    
    # Add to CSV
    echo "$config_name,\"$config_desc\",$rest14_f1,$laptop14_f1,$rest15_f1,$rest16_f1,$avg_f1" >> results/ablation_summary.csv
done

echo ""
echo "📁 RESULTS FILES:"
echo "================"
echo "Main summary: results/ablation_summary.csv"
echo "Training logs: logs/ablation/"
echo "Model checkpoints: experiments/ablation/"
echo ""

echo "🎯 NEXT STEPS FOR ACL/EMNLP 2025:"
echo "1. Open results/ablation_summary.csv in Excel/sheets"
echo "2. Create Table 3 (Ablation Study Results)"
echo "3. Calculate component contributions"
echo "4. Run statistical significance tests"
echo "5. Create ablation plots for Figure 4"
echo ""

echo "✅ Ablation study completed!"
echo "Ready for paper analysis!"