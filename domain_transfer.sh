#!/bin/bash

# Cross-Domain Transfer Evaluation - GRADIENT ABSA (FIXED VERSION)
echo "🔄 Starting Cross-Domain Transfer Evaluation"
echo "Started: $(date)"
echo "Testing models trained on source domains against target domains"
echo "================================================================"

# Create directories
mkdir -p experiments/cross_domain
mkdir -p logs/cross_domain

# Define source→target pairs using arrays (compatible with all bash versions)
# Format: "source:target"
domain_pairs=(
    "rest14:laptop14"
    "laptop14:rest14"
    "rest15:rest16"
    "rest16:rest15"
    "rest14:rest15"
    "rest14:rest16"
)

# Start time tracking
start_time=$(date +%s)
pair_count=0
total_pairs=${#domain_pairs[@]}

echo "📊 Cross-domain pairs to evaluate: $total_pairs"
echo ""

# Evaluate each source→target pair
for pair in "${domain_pairs[@]}"; do
    # Extract source and target from pair
    source="${pair%%:*}"
    target="${pair##*:}"
    
    pair_count=$((pair_count + 1))
    pair_name="${source}_to_${target}"
    
    echo "🎯 [$pair_count/$total_pairs] Evaluating: $source → $target"
    echo "Time: $(date)"
    echo "Model: experiments/baselines/$source/best_model.pt"
    echo "Test data: $target dataset"
    echo "----------------------------------------"
    
    # Check if source model exists
    if [ ! -f "experiments/baselines/$source/best_model.pt" ]; then
        echo "❌ Source model not found: experiments/baselines/$source/best_model.pt"
        echo "   Make sure baseline training completed for $source"
        echo ""
        continue
    fi
    
    # Run cross-domain evaluation
    python test.py \
        --model_path experiments/baselines/$source/best_model.pt \
        --dataset $target \
        --output_dir experiments/cross_domain/$pair_name \
        --confidence 0.6 \
        --wandb \
        2>&1 | tee logs/cross_domain/${pair_name}_evaluation.log
    
    # Check if evaluation succeeded
    if [ $? -eq 0 ]; then
        echo "✅ $pair_name evaluation completed successfully!"
        
        # Show quick results if available
        if [ -f "experiments/cross_domain/$pair_name/test_results_${target}.json" ]; then
            echo "📊 Cross-domain results:"
            grep -E "f1|precision|recall" experiments/cross_domain/$pair_name/test_results_${target}.json | head -3
        fi
    else
        echo "❌ $pair_name evaluation failed!"
        echo "Check log: logs/cross_domain/${pair_name}_evaluation.log"
    fi
    
    echo "[$pair_count/$total_pairs] $pair_name done. Moving to next..."
    echo ""
done

# Calculate total time
end_time=$(date +%s)
total_seconds=$((end_time - start_time))
hours=$((total_seconds / 3600))
minutes=$(((total_seconds % 3600) / 60))

echo ""
echo "🎉 ALL CROSS-DOMAIN EVALUATIONS COMPLETED!"
echo "Total time: ${hours}h ${minutes}m"
echo "Finished: $(date)"
echo "=================================="

# Create comprehensive summary
echo ""
echo "📊 CROSS-DOMAIN TRANSFER SUMMARY:"
echo "=================================="

# Create summary table header
echo ""
printf "%-20s %-12s %-12s %-12s\n" "Source→Target" "Aspect F1" "Opinion F1" "Triplet F1"
echo "----------------------------------------------------------------"

# Process results for each pair
for pair in "${domain_pairs[@]}"; do
    source="${pair%%:*}"
    target="${pair##*:}"
    pair_name="${source}_to_${target}"
    
    printf "%-20s " "$source→$target"
    
    if [ -f "experiments/cross_domain/$pair_name/test_results_${target}.json" ]; then
        # Extract F1 scores (adjust grep patterns based on your JSON structure)
        aspect_f1=$(grep -o '"aspect_f1"[^,]*' experiments/cross_domain/$pair_name/test_results_${target}.json | grep -o '[0-9.]*' | head -1)
        opinion_f1=$(grep -o '"opinion_f1"[^,]*' experiments/cross_domain/$pair_name/test_results_${target}.json | grep -o '[0-9.]*' | head -1)
        triplet_f1=$(grep -o '"triplet_f1"[^,]*' experiments/cross_domain/$pair_name/test_results_${target}.json | grep -o '[0-9.]*' | head -1)
        
        # If primary metrics not found, try alternative names
        if [ -z "$aspect_f1" ]; then
            aspect_f1=$(grep -o '"f1"[^,]*' experiments/cross_domain/$pair_name/test_results_${target}.json | grep -o '[0-9.]*' | head -1)
        fi
        
        printf "%-12s %-12s %-12s\n" "${aspect_f1:-N/A}" "${opinion_f1:-N/A}" "${triplet_f1:-N/A}"
    else
        printf "%-12s %-12s %-12s\n" "FAILED" "FAILED" "FAILED"
    fi
done

echo ""
echo "📁 DETAILED RESULTS LOCATIONS:"
echo "==============================="
for pair in "${domain_pairs[@]}"; do
    source="${pair%%:*}"
    target="${pair##*:}"
    pair_name="${source}_to_${target}"
    echo "$source→$target:"
    echo "  Results: experiments/cross_domain/$pair_name/"
    echo "  Log: logs/cross_domain/${pair_name}_evaluation.log"
    echo ""
done

echo "🎯 NEXT STEPS FOR PAPER:"
echo "1. Compare cross-domain results with single-domain baselines"
echo "2. Calculate transfer improvement percentages"
echo "3. Run statistical significance tests"
echo "4. Create tables for ACL/EMNLP 2025 submission"
echo ""
echo "✅ Ready for ablation studies and paper writing!"

# Generate performance comparison matrix for paper
echo ""
echo "📊 CREATING PAPER-READY PERFORMANCE MATRIX..."
echo "============================================="

# Create a CSV file for easy import into LaTeX/Excel
echo "Source,Target,Aspect_F1,Opinion_F1,Triplet_F1,Transfer_Type" > experiments/cross_domain/performance_matrix.csv

for pair in "${domain_pairs[@]}"; do
    source="${pair%%:*}"
    target="${pair##*:}"
    pair_name="${source}_to_${target}"
    
    # Determine transfer type
    if [[ "$source" == rest* && "$target" == rest* ]]; then
        transfer_type="Within_Restaurant"
    elif [[ "$source" == laptop* && "$target" == laptop* ]]; then
        transfer_type="Within_Laptop"
    elif [[ "$source" == rest* && "$target" == laptop* ]]; then
        transfer_type="Restaurant_to_Laptop"
    elif [[ "$source" == laptop* && "$target" == rest* ]]; then
        transfer_type="Laptop_to_Restaurant"
    else
        transfer_type="Cross_Domain"
    fi
    
    if [ -f "experiments/cross_domain/$pair_name/test_results_${target}.json" ]; then
        aspect_f1=$(grep -o '"aspect_f1"[^,]*' experiments/cross_domain/$pair_name/test_results_${target}.json | grep -o '[0-9.]*' | head -1)
        opinion_f1=$(grep -o '"opinion_f1"[^,]*' experiments/cross_domain/$pair_name/test_results_${target}.json | grep -o '[0-9.]*' | head -1)
        triplet_f1=$(grep -o '"triplet_f1"[^,]*' experiments/cross_domain/$pair_name/test_results_${target}.json | grep -o '[0-9.]*' | head -1)
        
        echo "$source,$target,${aspect_f1:-N/A},${opinion_f1:-N/A},${triplet_f1:-N/A},$transfer_type" >> experiments/cross_domain/performance_matrix.csv
    else
        echo "$source,$target,FAILED,FAILED,FAILED,$transfer_type" >> experiments/cross_domain/performance_matrix.csv
    fi
done

echo "✅ Performance matrix saved to: experiments/cross_domain/performance_matrix.csv"
echo "   Ready for import into LaTeX tables or analysis scripts!"