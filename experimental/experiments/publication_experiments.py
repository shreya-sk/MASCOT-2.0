# src/experiments/publication_experiments.py
import torch
import numpy as np
import random
import json
import os
from typing import Dict, List, Any
from dataclasses import dataclass
import wandb
from transformers import set_seed

@dataclass
class PublicationExperimentConfig:
    """Configuration for publication-ready experiments"""
    
    # Experiment metadata
    experiment_name: str = "unified-generative-absa-2025"
    paper_title: str = "Unified Generative Framework for Aspect-Based Sentiment Analysis with Advanced Contrastive Learning"
    
    # Model configurations to compare
    model_configs: List[str] = None
    
    # Datasets for comprehensive evaluation
    datasets: List[str] = None
    
    # Evaluation settings
    cross_validation_folds: int = 5
    statistical_significance_tests: bool = True
    
    # Few-shot settings
    few_shot_k_values: List[int] = None
    few_shot_n_ways: List[int] = None
    
    # Cross-domain evaluation
    source_domains: List[str] = None
    target_domains: List[str] = None
    
    # Reproducibility
    random_seeds: List[int] = None
    
    def __post_init__(self):
        if self.model_configs is None:
            self.model_configs = [
                'unified_generative',
                'unified_generative_with_contrastive',
                'few_shot_meta_learning',
                'instruction_based',
                'baseline_pipeline'
            ]
            
        if self.datasets is None:
            self.datasets = [
                'laptop14', 'rest14', 'rest15', 'rest16',
                'mams',  # Multi-aspect multi-sentiment
                'acos'   # Aspect-category-opinion-sentiment
            ]
            
        if self.few_shot_k_values is None:
            self.few_shot_k_values = [1, 3, 5, 10]
            
        if self.few_shot_n_ways is None:
            self.few_shot_n_ways = [3, 5]  # 3-way (POS/NEU/NEG), 5-way with categories
            
        if self.source_domains is None:
            self.source_domains = ['rest14', 'rest15']
            
        if self.target_domains is None:
            self.target_domains = ['laptop14', 'rest16']
            
        if self.random_seeds is None:
            self.random_seeds = [42, 1337, 2023, 2024, 2025]

class PublicationExperimentRunner:
    """
    Comprehensive experimental framework for publication-ready results
    """
    
    def __init__(self, config: PublicationExperimentConfig):
        self.config = config
        self.results = {}
        self.statistical_results = {}
        
        # Initialize wandb for experiment tracking
        wandb.init(
            project="absa-publication-experiments",
            name=config.experiment_name,
            config=config.__dict__
        )
        
    def run_all_experiments(self):
        """Run comprehensive experimental evaluation"""
        print(f"Starting publication experiments: {self.config.paper_title}")
        
        # 1. Main performance comparison
        print("\n=== Main Performance Comparison ===")
        self.main_performance_experiments()
        
        # 2. Few-shot learning evaluation
        print("\n=== Few-Shot Learning Evaluation ===")
        self.few_shot_experiments()
        
        # 3. Cross-domain robustness
        print("\n=== Cross-Domain Robustness ===")
        self.cross_domain_experiments()
        
        # 4. Ablation studies
        print("\n=== Ablation Studies ===")
        self.ablation_experiments()
        
        # 5. Implicit sentiment detection
        print("\n=== Implicit Sentiment Detection ===")
        self.implicit_sentiment_experiments()
        
        # 6. Statistical significance testing
        print("\n=== Statistical Significance Testing ===")
        self.statistical_significance_tests()
        
        # 7. Generate publication tables and figures
        print("\n=== Generating Publication Materials ===")
        self.generate_publication_materials()
        
        return self.results
    
    def main_performance_experiments(self):
        """Main performance comparison across all models and datasets"""
        for model_config in self.config.model_configs:
            for dataset in self.config.datasets:
                print(f"Evaluating {model_config} on {dataset}")
                
                # Run with multiple seeds for statistical significance
                seed_results = []
                for seed in self.config.random_seeds:
                    result = self._run_single_experiment(model_config, dataset, seed)
                    seed_results.append(result)
                
                # Aggregate results
                aggregated = self._aggregate_seed_results(seed_results)
                
                # Store results
                key = f"{model_config}_{dataset}"
                self.results[key] = aggregated
                
                # Log to wandb
                wandb.log({
                    f"{key}_f1_mean": aggregated['f1_mean'],
                    f"{key}_f1_std": aggregated['f1_std'],
                    f"{key}_accuracy_mean": aggregated['accuracy_mean'],
                    f"{key}_accuracy_std": aggregated['accuracy_std']
                })
    
    def few_shot_experiments(self):
        """Few-shot learning evaluation"""
        few_shot_results = {}
        
        for k in self.config.few_shot_k_values:
            for n_way in self.config.few_shot_n_ways:
                for dataset in self.config.datasets:
                    print(f"Few-shot: {k}-shot {n_way}-way on {dataset}")
                    
                    # Test our few-shot models
                    for model in ['few_shot_drp', 'few_shot_afml', 'few_shot_instruction']:
                        seed_results = []
                        for seed in self.config.random_seeds:
                            result = self._run_few_shot_experiment(
                                model, dataset, k, n_way, seed
                            )
                            seed_results.append(result)
                        
                        aggregated = self._aggregate_seed_results(seed_results)
                        key = f"{model}_{dataset}_{k}shot_{n_way}way"
                        few_shot_results[key] = aggregated
        
        self.results['few_shot'] = few_shot_results
    
    def cross_domain_experiments(self):
        """Cross-domain robustness evaluation"""
        cross_domain_results = {}
        
        for source in self.config.source_domains:
            for target in self.config.target_domains:
                if source != target:
                    print(f"Cross-domain: {source} -> {target}")
                    
                    # Test different transfer approaches
                    for approach in ['fine_tune', 'domain_adaptation', 'meta_learning']:
                        seed_results = []
                        for seed in self.config.random_seeds:
                            result = self._run_cross_domain_experiment(
                                source, target, approach, seed
                            )
                            seed_results.append(result)
                        
                        aggregated = self._aggregate_seed_results(seed_results)
                        key = f"{approach}_{source}_to_{target}"
                        cross_domain_results[key] = aggregated
        
        self.results['cross_domain'] = cross_domain_results
    
    def ablation_experiments(self):
        """Ablation studies to understand component contributions"""
        ablation_components = [
            'contrastive_learning',
            'instruction_tuning',
            'few_shot_meta_learning',
            'unified_generation',
            'implicit_detection'
        ]
        
        ablation_results = {}
        
        for component in ablation_components:
            for dataset in self.config.datasets:
                print(f"Ablation: removing {component} on {dataset}")
                
                # Test with and without component
                for setting in ['with_component', 'without_component']:
                    seed_results = []
                    for seed in self.config.random_seeds:
                        result = self._run_ablation_experiment(
                            component, dataset, setting, seed
                        )
                        seed_results.append(result)
                    
                    aggregated = self._aggregate_seed_results(seed_results)
                    key = f"{component}_{dataset}_{setting}"
                    ablation_results[key] = aggregated
        
        self.results['ablation'] = ablation_results
    
    def implicit_sentiment_experiments(self):
        """Specialized evaluation for implicit sentiment detection"""
        implicit_results = {}
        
        # Create datasets with implicit annotations
        implicit_datasets = self._create_implicit_datasets()
        
        for dataset_name, dataset in implicit_datasets.items():
            print(f"Implicit sentiment evaluation on {dataset_name}")
            
            # Test models specifically designed for implicit detection
            for model in ['unified_generative_implicit', 'instruction_implicit', 'contrastive_implicit']:
                seed_results = []
                for seed in self.config.random_seeds:
                    result = self._run_implicit_experiment(model, dataset, seed)
                    seed_results.append(result)
                
                aggregated = self._aggregate_seed_results(seed_results)
                key = f"{model}_{dataset_name}"
                implicit_results[key] = aggregated
        
        self.results['implicit'] = implicit_results
    
    def statistical_significance_tests(self):
        """Perform statistical significance testing"""
        if not self.config.statistical_significance_tests:
            return
        
        print("Computing statistical significance...")
        
        # Compare our best model against baselines
        our_model = 'unified_generative_with_contrastive'
        baselines = ['baseline_pipeline', 'bert_baseline', 'roberta_baseline']
        
        for dataset in self.config.datasets:
            for baseline in baselines:
                our_key = f"{our_model}_{dataset}"
                baseline_key = f"{baseline}_{dataset}"
                
                if our_key in self.results and baseline_key in self.results:
                    significance = self._compute_statistical_significance(
                        self.results[our_key],
                        self.results[baseline_key]
                    )
                    
                    self.statistical_results[f"{our_model}_vs_{baseline}_{dataset}"] = significance
    
    def generate_publication_materials(self):
        """Generate tables and figures for publication"""
        
        # Generate main results table
        self._generate_main_results_table()
        
        # Generate few-shot results table
        self._generate_few_shot_table()
        
        # Generate cross-domain results table
        self._generate_cross_domain_table()
        
        # Generate ablation study table
        self._generate_ablation_table()
        
        # Generate statistical significance table
        self._generate_significance_table()
        
        # Generate figures
        self._generate_performance_figures()
        self._generate_few_shot_figures()
        
        print("Publication materials generated in ./publication_materials/")
    
    def _run_single_experiment(self, model_config, dataset, seed):
        """Run a single experiment configuration"""
        set_seed(seed)
        
        # This would integrate with your actual model training/evaluation
        # For now, returning mock results that follow the expected format
        
        # Mock results - replace with actual model evaluation
        mock_results = {
            'f1': np.random.normal(0.75, 0.05),  # Mock F1 score
            'accuracy': np.random.normal(0.80, 0.05),  # Mock accuracy
            'precision': np.random.normal(0.78, 0.05),
            'recall': np.random.normal(0.72, 0.05),
            'aspect_f1': np.random.normal(0.70, 0.05),
            'opinion_f1': np.random.normal(0.68, 0.05),
            'sentiment_f1': np.random.normal(0.82, 0.05)
        }
        
        return mock_results
    
    def _run_few_shot_experiment(self, model, dataset, k, n_way, seed):
        """Run few-shot experiment"""
        set_seed(seed)
        
        # Mock few-shot results - replace with actual evaluation
        base_performance = 0.60  # Base few-shot performance
        k_bonus = k * 0.05  # Performance improves with more shots
        
        mock_results = {
            'f1': np.random.normal(base_performance + k_bonus, 0.03),
            'accuracy': np.random.normal(base_performance + k_bonus + 0.05, 0.03)
        }
        
        return mock_results
    
    def _run_cross_domain_experiment(self, source, target, approach, seed):
        """Run cross-domain experiment"""
        set_seed(seed)
        
        # Mock cross-domain results - replace with actual evaluation
        domain_penalty = 0.15  # Performance drop for cross-domain
        approach_bonus = {'fine_tune': 0.0, 'domain_adaptation': 0.05, 'meta_learning': 0.08}
        
        base_performance = 0.70
        final_performance = base_performance - domain_penalty + approach_bonus[approach]
        
        mock_results = {
            'f1': np.random.normal(final_performance, 0.04),
            'accuracy': np.random.normal(final_performance + 0.05, 0.04)
        }
        
        return mock_results
    
    def _run_ablation_experiment(self, component, dataset, setting, seed):
        """Run ablation experiment"""
        set_seed(seed)
        
        # Mock ablation results - replace with actual evaluation
        component_contribution = {
            'contrastive_learning': 0.03,
            'instruction_tuning': 0.05,
            'few_shot_meta_learning': 0.04,
            'unified_generation': 0.06,
            'implicit_detection': 0.02
        }
        
        base_performance = 0.72
        if setting == 'with_component':
            final_performance = base_performance + component_contribution[component]
        else:
            final_performance = base_performance
        
        mock_results = {
            'f1': np.random.normal(final_performance, 0.02),
            'accuracy': np.random.normal(final_performance + 0.03, 0.02)
        }
        
        return mock_results
    
    def _run_implicit_experiment(self, model, dataset, seed):
        """Run implicit sentiment detection experiment"""
        set_seed(seed)
        
        # Mock implicit detection results - replace with actual evaluation
        implicit_difficulty = 0.20  # Implicit detection is harder
        model_bonus = {'unified_generative_implicit': 0.08, 'instruction_implicit': 0.06, 'contrastive_implicit': 0.05}
        
        base_performance = 0.60
        final_performance = base_performance + model_bonus.get(model, 0.0)
        
        mock_results = {
            'f1': np.random.normal(final_performance, 0.04),
            'accuracy': np.random.normal(final_performance + 0.05, 0.04),
            'implicit_f1': np.random.normal(final_performance - implicit_difficulty, 0.05),
            'explicit_f1': np.random.normal(final_performance + 0.10, 0.03)
        }
        
        return mock_results
    
    def _aggregate_seed_results(self, seed_results):
        """Aggregate results across multiple seeds"""
        aggregated = {}
        
        # Get all metric keys
        all_keys = set()
        for result in seed_results:
            all_keys.update(result.keys())
        
        # Compute mean and std for each metric
        for key in all_keys:
            values = [result[key] for result in seed_results if key in result]
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
            aggregated[f"{key}_values"] = values
        
        return aggregated
    
    def _compute_statistical_significance(self, our_results, baseline_results):
        """Compute statistical significance using t-test"""
        from scipy import stats
        
        our_f1_values = our_results['f1_values']
        baseline_f1_values = baseline_results['f1_values']
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(our_f1_values, baseline_f1_values)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(our_f1_values) - 1) * np.var(our_f1_values) + 
                             (len(baseline_f1_values) - 1) * np.var(baseline_f1_values)) / 
                            (len(our_f1_values) + len(baseline_f1_values) - 2))
        
        cohens_d = (np.mean(our_f1_values) - np.mean(baseline_f1_values)) / pooled_std
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'cohens_d': cohens_d,
            'effect_size_interpretation': self._interpret_effect_size(cohens_d)
        }
    
    def _interpret_effect_size(self, cohens_d):
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _create_implicit_datasets(self):
        """Create datasets with implicit sentiment annotations"""
        # Mock implicit datasets - replace with actual data processing
        implicit_datasets = {
            'rest14_implicit': [],
            'laptop14_implicit': [],
            'mixed_implicit': []
        }
        return implicit_datasets
    
    def _generate_main_results_table(self):
        """Generate main results table for publication"""
        os.makedirs("publication_materials", exist_ok=True)
        
        # Create LaTeX table
        latex_table = """
\\begin{table}[ht]
\\centering
\\caption{Main Performance Comparison on ABSA Datasets}
\\label{tab:main_results}
\\begin{tabular}{l|cccc|c}
\\toprule
\\textbf{Model} & \\textbf{Rest14} & \\textbf{Rest15} & \\textbf{Rest16} & \\textbf{Laptop14} & \\textbf{Avg} \\\\
\\midrule
"""
        
        # Add rows for each model
        for model in self.config.model_configs:
            row = f"{model.replace('_', ' ').title()}"
            f1_scores = []
            
            for dataset in ['rest14', 'rest15', 'rest16', 'laptop14']:
                key = f"{model}_{dataset}"
                if key in self.results:
                    f1_mean = self.results[key]['f1_mean']
                    f1_std = self.results[key]['f1_std']
                    row += f" & {f1_mean:.2f}±{f1_std:.2f}"
                    f1_scores.append(f1_mean)
                else:
                    row += " & -"
            
            # Add average
            if f1_scores:
                avg_f1 = np.mean(f1_scores)
                row += f" & \\textbf{{{avg_f1:.2f}}}"
            else:
                row += " & -"
            
            row += " \\\\\n"
            latex_table += row
        
        latex_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
        
        with open("publication_materials/main_results_table.tex", "w") as f:
            f.write(latex_table)
    
    def _generate_few_shot_table(self):
        """Generate few-shot learning results table"""
        latex_table = """
\\begin{table}[ht]
\\centering
\\caption{Few-Shot Learning Performance}
\\label{tab:few_shot_results}
\\begin{tabular}{l|cccc}
\\toprule
\\textbf{Method} & \\textbf{1-shot} & \\textbf{3-shot} & \\textbf{5-shot} & \\textbf{10-shot} \\\\
\\midrule
"""
        
        methods = ['few_shot_drp', 'few_shot_afml', 'few_shot_instruction']
        
        for method in methods:
            row = f"{method.replace('_', ' ').title()}"
            for k in [1, 3, 5, 10]:
                # Average across datasets and n_way settings
                scores = []
                for dataset in self.config.datasets:
                    for n_way in [3, 5]:
                        key = f"{method}_{dataset}_{k}shot_{n_way}way"
                        if key in self.results.get('few_shot', {}):
                            scores.append(self.results['few_shot'][key]['f1_mean'])
                
                if scores:
                    avg_score = np.mean(scores)
                    std_score = np.std(scores)
                    row += f" & {avg_score:.2f}±{std_score:.2f}"
                else:
                    row += " & -"
            
            row += " \\\\\n"
            latex_table += row
        
        latex_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
        
        with open("publication_materials/few_shot_table.tex", "w") as f:
            f.write(latex_table)
    
    def _generate_cross_domain_table(self):
        """Generate cross-domain robustness table"""
        latex_table = """
\\begin{table}[ht]
\\centering
\\caption{Cross-Domain Transfer Learning Results}
\\label{tab:cross_domain_results}
\\begin{tabular}{l|ccc}
\\toprule
\\textbf{Transfer Direction} & \\textbf{Fine-tune} & \\textbf{Domain Adapt} & \\textbf{Meta Learning} \\\\
\\midrule
"""
        
        for source in self.config.source_domains:
            for target in self.config.target_domains:
                if source != target:
                    row = f"{source} → {target}"
                    
                    for approach in ['fine_tune', 'domain_adaptation', 'meta_learning']:
                        key = f"{approach}_{source}_to_{target}"
                        if key in self.results.get('cross_domain', {}):
                            f1_mean = self.results['cross_domain'][key]['f1_mean']
                            f1_std = self.results['cross_domain'][key]['f1_std']
                            row += f" & {f1_mean:.2f}±{f1_std:.2f}"
                        else:
                            row += " & -"
                    
                    row += " \\\\\n"
                    latex_table += row
        
        latex_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
        
        with open("publication_materials/cross_domain_table.tex", "w") as f:
            f.write(latex_table)
    
    def _generate_ablation_table(self):
        """Generate ablation study table"""
        latex_table = """
\\begin{table}[ht]
\\centering
\\caption{Ablation Study Results}
\\label{tab:ablation_results}
\\begin{tabular}{l|c|c}
\\toprule
\\textbf{Component} & \\textbf{With Component} & \\textbf{Contribution} \\\\
\\midrule
"""
        
        components = [
            'contrastive_learning',
            'instruction_tuning', 
            'few_shot_meta_learning',
            'unified_generation',
            'implicit_detection'
        ]
        
        for component in components:
            # Average across datasets
            with_scores = []
            without_scores = []
            
            for dataset in self.config.datasets:
                with_key = f"{component}_{dataset}_with_component"
                without_key = f"{component}_{dataset}_without_component"
                
                if (with_key in self.results.get('ablation', {}) and 
                    without_key in self.results.get('ablation', {})):
                    with_scores.append(self.results['ablation'][with_key]['f1_mean'])
                    without_scores.append(self.results['ablation'][without_key]['f1_mean'])
            
            if with_scores and without_scores:
                avg_with = np.mean(with_scores)
                avg_without = np.mean(without_scores)
                contribution = avg_with - avg_without
                
                row = f"{component.replace('_', ' ').title()} & {avg_with:.3f} & +{contribution:.3f} \\\\\n"
                latex_table += row
        
        latex_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
        
        with open("publication_materials/ablation_table.tex", "w") as f:
            f.write(latex_table)
    
    def _generate_significance_table(self):
        """Generate statistical significance table"""
        if not self.statistical_results:
            return
            
        latex_table = """
\\begin{table}[ht]
\\centering
\\caption{Statistical Significance Test Results}
\\label{tab:significance_results}
\\begin{tabular}{l|c|c|c|c}
\\toprule
\\textbf{Comparison} & \\textbf{p-value} & \\textbf{Significant} & \\textbf{Cohen's d} & \\textbf{Effect Size} \\\\
\\midrule
"""
        
        for comparison, stats in self.statistical_results.items():
            p_val = stats['p_value']
            significant = "✓" if stats['significant'] else "✗"
            cohens_d = stats['cohens_d']
            effect_size = stats['effect_size_interpretation']
            
            row = f"{comparison.replace('_', ' ')} & {p_val:.4f} & {significant} & {cohens_d:.3f} & {effect_size} \\\\\n"
            latex_table += row
        
        latex_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
        
        with open("publication_materials/significance_table.tex", "w") as f:
            f.write(latex_table)
    
    def _generate_performance_figures(self):
        """Generate performance comparison figures"""
        import matplotlib.pyplot as plt
        
        # Main performance comparison bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = []
        f1_means = []
        f1_stds = []
        
        for model in self.config.model_configs:
            # Average F1 across datasets
            scores = []
            for dataset in self.config.datasets:
                key = f"{model}_{dataset}"
                if key in self.results:
                    scores.append(self.results[key]['f1_mean'])
            
            if scores:
                models.append(model.replace('_', ' ').title())
                f1_means.append(np.mean(scores))
                f1_stds.append(np.std(scores))
        
        x_pos = range(len(models))
        bars = ax.bar(x_pos, f1_means, yerr=f1_stds, capsize=5, 
                     color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        
        ax.set_xlabel('Models')
        ax.set_ylabel('F1 Score')
        ax.set_title('ABSA Performance Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('publication_materials/performance_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_few_shot_figures(self):
        """Generate few-shot learning figures"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        k_values = [1, 3, 5, 10]
        methods = ['few_shot_drp', 'few_shot_afml', 'few_shot_instruction']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, method in enumerate(methods):
            scores = []
            for k in k_values:
                # Average across datasets and n_way
                method_scores = []
                for dataset in self.config.datasets:
                    for n_way in [3, 5]:
                        key = f"{method}_{dataset}_{k}shot_{n_way}way"
                        if key in self.results.get('few_shot', {}):
                            method_scores.append(self.results['few_shot'][key]['f1_mean'])
                
                if method_scores:
                    scores.append(np.mean(method_scores))
                else:
                    scores.append(0)
            
            ax.plot(k_values, scores, 'o-', color=colors[i], 
                   label=method.replace('_', ' ').title(), linewidth=2, markersize=6)
        
        ax.set_xlabel('Number of Shots (k)')
        ax.set_ylabel('F1 Score')
        ax.set_title('Few-Shot Learning Performance')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xticks(k_values)
        
        plt.tight_layout()
        plt.savefig('publication_materials/few_shot_performance.pdf', dpi=300, bbox_inches='tight')
        plt.close()

# Usage example
def run_publication_experiments():
    """Main function to run all publication experiments"""
    
    # Configure experiments
    config = PublicationExperimentConfig(
        experiment_name="unified-generative-absa-2025",
        paper_title="Unified Generative Framework for Aspect-Based Sentiment Analysis with Advanced Contrastive Learning"
    )
    
    # Run experiments
    runner = PublicationExperimentRunner(config)
    results = runner.run_all_experiments()
    
    # Save complete results
    with open('publication_materials/complete_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("Publication experiments completed!")
    print("Results saved in ./publication_materials/")
    
    return results

if __name__ == "__main__":
    run_publication_experiments()