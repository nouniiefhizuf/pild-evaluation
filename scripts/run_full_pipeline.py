#!/usr/bin/env python3
"""Run complete inference and evaluation pipeline."""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from physbench import load_dataset
from prompts import load_prompt
from inference import GPT4oWrapper, LlamaWrapper, QwenWrapper, DeepSeekWrapper
from evaluation import compute_pde, check_executability, compute_rouge_l, compute_bertscore


MODEL_WRAPPERS = {
    "gpt-4o": GPT4oWrapper,
    "llama-3.3-70b": LlamaWrapper,
    "qwen2.5-72b": QwenWrapper,
    "deepseek-v3": DeepSeekWrapper
}


def run_model(model_name: str, examples: list, prompt_levels: list, output_dir: Path):
    """Run inference for one model across all examples and prompt levels."""
    print(f"\n{'='*60}")
    print(f"Running {model_name}")
    print(f"{'='*60}")

    # Initialize model
    wrapper_cls = MODEL_WRAPPERS[model_name]
    model = wrapper_cls()

    model_dir = output_dir / model_name.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)

    for example in tqdm(examples, desc=f"{model_name} examples"):
        for level in prompt_levels:
            # Load and fill prompt
            prompt_template = load_prompt(level)
            prompt = prompt_template.fill(
                physics_problem_description=example["description"],
                initial_conditions=example["initial_conditions"],
                simulation_time=example["t_span"][1],
                time_step=0.01
            )

            # Generate code
            try:
                response = model.generate(prompt)
                code = model.extract_code(response)
            except Exception as e:
                print(f"ERROR generating {example['id']} with {level}: {e}")
                continue

            # Evaluate
            try:
                gt_fn = example.get("ground_truth_fn")
                pde = compute_pde(code, gt_fn, tuple(example["t_span"]), 
                                example["y0"], example["initial_conditions"])
                executable = check_executability(code)
                rouge = compute_rouge_l(code, example["reference_code"])
                bert = compute_bertscore(code, example["reference_code"])

                result = {
                    "example_id": example["id"],
                    "prompt_level": level,
                    "model": model_name,
                    "code": code,
                    "metrics": {
                        "pde": pde,
                        "execution_rate": 1.0 if executable else 0.0,
                        "rouge_l": rouge["rouge_l_f1"],
                        "bertscore": bert["bertscore_f1"],
                        "pass_at_1": 1.0 if (executable and pde < 0.15) else 0.0
                    }
                }

                # Save
                out_file = model_dir / f"{example['id']}_{level}.json"
                with open(out_file, "w") as f:
                    json.dump(result, f, indent=2)

            except Exception as e:
                print(f"ERROR evaluating {example['id']}: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(description="Run full evaluation pipeline")
    parser.add_argument("--models", nargs="+", default=["gpt-4o"],
                       choices=list(MODEL_WRAPPERS.keys()),
                       help="Models to evaluate")
    parser.add_argument("--prompt_levels", nargs="+", 
                       default=["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10"],
                       help="Prompt levels to test")
    parser.add_argument("--num_examples", type=int, default=1000,
                       help="Number of examples (max 1000)")
    parser.add_argument("--output_dir", default="results",
                       help="Output directory")
    args = parser.parse_args()

    # Load dataset
    print("Loading PhysBench-1K...")
    examples = load_dataset(num_examples=args.num_examples)
    print(f"Loaded {len(examples)} examples")

    # Run each model
    output_dir = Path(args.output_dir)
    for model in args.models:
        run_model(model, examples, args.prompt_levels, output_dir)

    print("\n" + "="*60)
    print("Pipeline Complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
