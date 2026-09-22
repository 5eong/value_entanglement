import sys
import os
import gc
import shutil
import torch
import subprocess
import argparse

from huggingface_hub.constants import HF_HUB_CACHE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_behavior import main as run_model_behavior
from activations import main as run_activations

MODELS = [
    # 'qwen-2.5-7b',
    # 'gemma-2-9b',
    # 'Mistral-Small-24B-Base-2501',
    'gemma-3-4b-it',
    'gemma-3-12b-it',
    'gemma-3-27b-it',
    'Qwen3-8B',
    'Qwen3-14B',
    'Qwen3-32B',
]

INTERVENTION_CONFIGS = [    
    {
        'use_scenarios': True,
        'steering_concept': 'moral',
        'eval_dataset': 'Dillion',
        'intervention_type': 'direction_ablation',
        'strength': 2.0,
        'vector_method': 'means'
    },
    {
        'use_scenarios': True,
        'steering_concept': 'moral',
        'eval_dataset': 'moralvalue68_morality',
        'intervention_type': 'direction_ablation',
        'strength': 2.0,
        'vector_method': 'means'
    },
    {
        'use_scenarios': True,
        'steering_concept': 'moral',
        'eval_dataset': 'moralvalue68_grammar',
        'intervention_type': 'direction_ablation',
        'strength': 2.0,
        'vector_method': 'means'
    },
    {
        'use_scenarios': True,
        'steering_concept': 'moral',
        'eval_dataset': 'moralvalue68_economic',
        'intervention_type': 'direction_ablation',
        'strength': 2.0,
        'vector_method': 'means'
    },
    {
        'use_scenarios': True,
        'steering_concept': 'moral',
        'eval_dataset': 'animals_size',
        'intervention_type': 'direction_ablation',
        'strength': 2.0,
        'vector_method': 'means'
    },
    {
        'use_scenarios': True,
        'steering_concept': 'moral',
        'eval_dataset': 'names_age',
        'intervention_type': 'direction_ablation',
        'strength': 2.0,
        'vector_method': 'means'
    },
    {
        'use_scenarios': True,
        'steering_concept': 'moral',
        'eval_dataset': 'states_temperature',
        'intervention_type': 'direction_ablation',
        'strength': 2.0,
        'vector_method': 'means'
    },
    {
        'use_scenarios': True,
        'steering_concept': 'moral',
        'eval_dataset': 'weather_wetness',
        'intervention_type': 'direction_ablation',
        'strength': 2.0,
        'vector_method': 'means'
    },
    {
        'use_scenarios': True,
        'steering_concept': 'grammar',
        'eval_dataset': 'Dillion',
        'intervention_type': 'direction_ablation',
        'strength': 2.0,
        'vector_method': 'means'
    },
    {
        'use_scenarios': True,
        'steering_concept': 'grammar',
        'eval_dataset': 'moralvalue68_morality',
        'intervention_type': 'direction_ablation',
        'strength': 2.0,
        'vector_method': 'means'
    },
    {
        'use_scenarios': True,
        'steering_concept': 'grammar',
        'eval_dataset': 'moralvalue68_economic',
        'intervention_type': 'direction_ablation',
        'strength': 2.0,
        'vector_method': 'means'
    },
    {
        'use_scenarios': True,
        'steering_concept': 'grammar',
        'eval_dataset': 'animals_size',
        'intervention_type': 'direction_ablation',
        'strength': 2.0,
        'vector_method': 'means'
    },
    {
        'use_scenarios': True,
        'steering_concept': 'grammar',
        'eval_dataset': 'names_age',
        'intervention_type': 'direction_ablation',
        'strength': 2.0,
        'vector_method': 'means'
    },
    {
        'use_scenarios': True,
        'steering_concept': 'grammar',
        'eval_dataset': 'states_temperature',
        'intervention_type': 'direction_ablation',
        'strength': 2.0,
        'vector_method': 'means'
    },
    {
        'use_scenarios': True,
        'steering_concept': 'grammar',
        'eval_dataset': 'weather_wetness',
        'intervention_type': 'direction_ablation',
        'strength': 2.0,
        'vector_method': 'means'
    },
    {
        'use_scenarios': True,
        'steering_concept': 'economic',
        'eval_dataset': 'Dillion',
        'intervention_type': 'direction_ablation',
        'strength': 2.0,
        'vector_method': 'means'
    },
    {
        'use_scenarios': True,
        'steering_concept': 'economic',
        'eval_dataset': 'moralvalue68_morality',
        'intervention_type': 'direction_ablation',
        'strength': 2.0,
        'vector_method': 'means'
    },
    {
        'use_scenarios': True,
        'steering_concept': 'economic',
        'eval_dataset': 'moralvalue68_economic',
        'intervention_type': 'direction_ablation',
        'strength': 2.0,
        'vector_method': 'means'
    },
    {
        'use_scenarios': True,
        'steering_concept': 'economic',
        'eval_dataset': 'animals_size',
        'intervention_type': 'direction_ablation',
        'strength': 2.0,
        'vector_method': 'means'
    },
    {
        'use_scenarios': True,
        'steering_concept': 'economic',
        'eval_dataset': 'names_age',
        'intervention_type': 'direction_ablation',
        'strength': 2.0,
        'vector_method': 'means'
    },
    {
        'use_scenarios': True,
        'steering_concept': 'economic',
        'eval_dataset': 'states_temperature',
        'intervention_type': 'direction_ablation',
        'strength': 2.0,
        'vector_method': 'means'
    },
    {
        'use_scenarios': True,
        'steering_concept': 'economic',
        'eval_dataset': 'weather_wetness',
        'intervention_type': 'direction_ablation',
        'strength': 2.0,
        'vector_method': 'means'
    },
]


def delete_hf_model_cache(model_path):
    cache_name = "models--" + model_path.replace("/", "--")
    cache_path = os.path.join(HF_HUB_CACHE, cache_name)
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path, ignore_errors=True)
        print(f"  removed disk cache: {cache_path}")
    else:
        print(f"  no disk cache at: {cache_path}")


def run_intervention_experiment(model, config):

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), '..', 'src', 'intervention.py'),
        '--model', model,
        '--use-scenarios', str(config['use_scenarios']).lower(),
        '--steering-concept', config['steering_concept'],
        '--eval-dataset', config['eval_dataset'],
        '--intervention-type', config['intervention_type'],
        '--strength', str(config['strength']),
        '--vector-method', config['vector_method']
    ]

    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))

    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Intervention failed with return code {result.returncode}")

    print(result.stdout)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run all experiments for multiple models')
    parser.add_argument('--run-interventions', type=bool, default=True,
                        help='Run intervention experiments (default: True)')
    parser.add_argument('--output-dir', type=str, default='./data/',
                        help='Output directory for results (default: ./data/)')

    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(__file__), args.output_dir)

    print("=" * 60)
    print(f"Running experiments for {len(MODELS)} models")
    print(f"  Interventions: {args.run_interventions}")
    print("=" * 60)

    for i, model_name in enumerate(MODELS, 1):
        print(f"\n{'#' * 60}")
        print(f"[{i}/{len(MODELS)}] MODEL: {model_name}")
        print(f"{'#' * 60}\n")

        model_name_full = {
            'qwen-2.5-7b': 'Qwen/Qwen2.5-7B',
            'qwen-2.5-7b-it': 'Qwen/Qwen2.5-7B-Instruct',
            'gemma-2-9b': 'google/gemma-2-9b',
            'gemma-2-9b-it': 'google/gemma-2-9b-it',
            'Mistral-Small-24B-Base-2501': 'mistralai/Mistral-Small-24B-Base-2501',
            'Mistral-Small-24B-Instruct-2501': 'mistralai/Mistral-Small-24B-Instruct-2501',
            'gemma-3-4b-it': 'google/gemma-3-4b-it',
            'gemma-3-12b-it': 'google/gemma-3-12b-it',
            'gemma-3-27b-it': 'google/gemma-3-27b-it',
            'Qwen3-8B': 'Qwen/Qwen3-8B',
            'Qwen3-14B': 'Qwen/Qwen3-14B',
            'Qwen3-32B': 'Qwen/Qwen3-32B',
        }.get(model_name, model_name)

        if args.run_interventions:
            for j, config in enumerate(INTERVENTION_CONFIGS, 1):
                print(f"\n>>> Experiment 3.{j}: Interventions")
                print(f"    Config: {config['steering_concept']} on {config['eval_dataset']}")
                print(f"    Method: {config['intervention_type']} (strength={config['strength']}, vector={config['vector_method']})")
                print("-" * 60)
                try:
                    run_intervention_experiment(model_name, config)
                    print(f"✓ Intervention {j} completed: {model_name}")
                except Exception as e:
                    print(f"✗ Intervention {j} failed: {model_name}")
                    print(f"Error: {str(e)}")
                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print("Cleared memory after intervention")
        else:
            print(f"\n>>> Experiment 3: Interventions [SKIPPED]")

        print(f"\n{'#' * 60}")
        print(f"Completed all experiments for: {model_name}")
        print(f"{'#' * 60}\n")

        delete_hf_model_cache(model_name_full)

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 60)
