import asyncio
import os
import json
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional

import black
import tqdm

# Internal Imports
from gg_bench.utils.chat_completion import Message, UsageTracker, chat_completion_async
from gg_bench.utils.load_yaml import load_yaml
from gg_bench.utils.markdown import extract_python_code

import re

def extract_pddl_domain(text: str) -> Optional[str]:
    """
    Extracts the PDDL domain file content from a code block marked with ~~~pddl ... ~~~.
    """
    match = re.search(r"~~~pddl(.*?)~~~", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


async def generate_pddl_generator_and_save(
    game_idx: int,
    game_description: str,
    gym_env_code: str,
    prompt: str,
    model: str,
    max_completion_tokens: int,
    usage_tracker: UsageTracker,
    pbar: tqdm.tqdm,
) -> None:
    """
    Generate a PDDL-based level generator using the specified model and save it to a file.

    Args:
        game_idx (int): The index of the game being generated.
        game_description (str): The description of the game.
        gym_env_code (str): The generated Gym environment code.
        prompt (str): The instruction prompt for generating the PDDL level generator.
        model (str): The name of the model to use for chat completion.
        max_completion_tokens (int): The maximum number of tokens to generate.
        usage_tracker (UsageTracker): The usage tracker object.
        pbar (tqdm.tqdm): A progress bar to update after generation.

    Returns:
        None

    This function generates a PDDL level generator that is compatible with the
    provided Gym environment code, saves it to a file, and updates the progress bar.
    """
    # Replace placeholders in prompt
    prompt = prompt.replace("<GameDescription>", game_description)
    prompt = prompt.replace("<PythonCode>", gym_env_code)
    
    try:
        response = await chat_completion_async(
            model=model,
            messages=[Message(role="user", content=prompt)],
            max_completion_tokens=max_completion_tokens,
            usage_tracker=usage_tracker,
        )
    except Exception as e:
        print(f"Error generating PDDL generator for idx {game_idx}: {e}")
        return
    
    generator_code = extract_python_code(response.strip())
    if not generator_code:
        raise ValueError(f"No Python code found for PDDL generator idx {game_idx}")
    
    try:
        generator_code = black.format_str(generator_code, mode=black.FileMode())
    except:
        pass
    
    # Save the PDDL generator code
    generator_path = f"gg_bench/data/pddl_generators/generator_{game_idx}.py"
    with open(generator_path, "w") as f:
        f.write(generator_code)
    
    # Create PDDL domain directory for this game
    domain_dir = f"gg_bench/data/pddl_domains/{game_idx}"
    os.makedirs(domain_dir, exist_ok=True)
    
    # Extract and save the PDDL domain file if present
    domain_content = extract_pddl_domain(response)
    if domain_content:
        domain_path = f"{domain_dir}/domain.pddl"
        with open(domain_path, "w") as f:
            f.write(domain_content)
    else:
        print(f"Warning: No PDDL domain found for game {game_idx}")
    
    pbar.update(1)


def load_gym_environment_code(game_idx: int) -> Optional[str]:
    """
    Load the previously generated Gym environment code for a specific game.
    
    Args:
        game_idx (int): The index of the game.
        
    Returns:
        Optional[str]: The Gym environment code, or None if not found.
    """
    env_path = f"gg_bench/data/envs/env_{game_idx}.py"
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            return f.read()
    return None


async def generate_test_levels(
    game_idx: int,
    num_test_levels: int = 20,
    difficulties: List[str] = ["easy", "medium", "hard"]
) -> None:
    """
    Generate test levels using the created PDDL generator and validate compatibility
    with the Gym environment.
    
    Args:
        game_idx (int): The index of the game.
        num_test_levels (int): Number of test levels to generate per difficulty.
        difficulties (List[str]): List of difficulty levels to generate.
    """
    try:
        # Import the generated PDDL generator
        spec = importlib.util.spec_from_file_location(
            f"generator_{game_idx}", 
            f"gg_bench/data/pddl_generators/generator_{game_idx}.py"
        )
        generator_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator_module)
        
        # Try to import the corresponding Gym environment for validation
        gym_spec = importlib.util.spec_from_file_location(
            f"env_{game_idx}",
            f"gg_bench/data/envs/env_{game_idx}.py"
        )
        gym_module = importlib.util.module_from_spec(gym_spec)
        gym_spec.loader.exec_module(gym_module)
        
        # Create levels directory
        levels_dir = f"gg_bench/data/generated_levels/{game_idx}"
        os.makedirs(levels_dir, exist_ok=True)
        
        # Generate levels for each difficulty
        for difficulty in difficulties:
            diff_dir = os.path.join(levels_dir, difficulty)
            os.makedirs(diff_dir, exist_ok=True)
            
            generator = generator_module.PDDLLevelGenerator(difficulty=difficulty, seed=42)
            levels = generator.generate_batch(count=num_test_levels)
            
            successful_levels = 0
            for i, level_data in enumerate(levels):
                try:
                    level_path = os.path.join(diff_dir, f"level_{i:03d}")
                    generator.save_level(level_data, level_path)
                    
                    # Validate with PDDL planner
                    validation_result = generator.validate_level(level_data)
                    
                    # Test compatibility with Gym environment
                    try:
                        # This assumes the generator saves levels in a format
                        # that can be loaded by the Gym environment
                        test_env = gym_module.CustomEnv()  # Load test level
                        gym_compatible = True
                    except Exception as gym_error:
                        print(f"Gym compatibility error for level {i}: {gym_error}")
                        gym_compatible = False
                    
                    # Save metadata
                    metadata = {
                        "game_idx": game_idx,
                        "difficulty": difficulty,
                        "level_id": i,
                        "pddl_validation": validation_result,
                        "gym_compatible": gym_compatible,
                        "constraints": generator.get_difficulty_constraints(),
                        "generation_timestamp": str(asyncio.get_event_loop().time())
                    }
                    
                    with open(f"{level_path}_metadata.json", "w") as f:
                        json.dump(metadata, f, indent=2)
                    
                    if validation_result.get("solvable", False) and gym_compatible:
                        successful_levels += 1
                        
                except Exception as level_error:
                    print(f"Error generating level {i} for game {game_idx}: {level_error}")
                    continue
            
            print(f"Game {game_idx} - {difficulty}: {successful_levels}/{num_test_levels} levels generated successfully")
                    
    except Exception as e:
        print(f"Error generating test levels for game {game_idx}: {e}")


async def validate_generator_quality(game_idx: int) -> Dict:
    """
    Validate the quality of the generated PDDL generator.
    
    Args:
        game_idx (int): The index of the game.
        
    Returns:
        Dict: Quality metrics and validation results.
    """
    try:
        # Import generator
        spec = importlib.util.spec_from_file_location(
            f"generator_{game_idx}", 
            f"gg_bench/data/pddl_generators/generator_{game_idx}.py"
        )
        generator_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator_module)
        
        # Test generation for different difficulties
        quality_metrics = {
            "game_idx": game_idx,
            "generator_functional": True,
            "difficulties_supported": [],
            "generation_success_rate": {},
            "errors": []
        }
        
        for difficulty in ["easy", "medium", "hard"]:
            try:
                generator = generator_module.PDDLLevelGenerator(difficulty=difficulty, seed=123)
                test_levels = generator.generate_batch(count=5)
                
                successful = 0
                for level in test_levels:
                    try:
                        validation = generator.validate_level(level)
                        if validation.get("solvable", False):
                            successful += 1
                    except:
                        continue
                
                quality_metrics["difficulties_supported"].append(difficulty)
                quality_metrics["generation_success_rate"][difficulty] = successful / len(test_levels)
                
            except Exception as diff_error:
                quality_metrics["errors"].append(f"Difficulty {difficulty}: {str(diff_error)}")
        
        return quality_metrics
        
    except Exception as e:
        return {
            "game_idx": game_idx,
            "generator_functional": False,
            "error": str(e)
        }


async def main() -> None:
    """
    Main function to generate PDDL level generators for all games.
    """
    config = load_yaml("gg_bench/configs/generate_pddl_generators.yaml")

    prompt = config["prompt"]
    model = config["model"]
    max_completion_tokens = config["max_completion_tokens"]
    num_games = config["num_games"]
    generate_test_levels_flag = config.get("generate_test_levels", False)
    validate_quality_flag = config.get("validate_quality", True)
    only_validate_and_test = config.get("only_validate_and_test", False)

    # Create necessary directories
    for dir_name in ["pddl_generators", "pddl_domains", "generated_levels", "usage_trackers", "quality_reports"]:
        os.makedirs(f"gg_bench/data/{dir_name}", exist_ok=True)

    tasks = []
    usage_tracker = UsageTracker()
    
    # Find games that需要PDDL generators和有description和env
    game_indices = []
    if only_validate_and_test:
        # 只收集已存在的 generator
        for i in range(1, num_games + 1):
            if os.path.exists(f"gg_bench/data/pddl_generators/generator_{i}.py"):
                game_indices.append(i)
        if not game_indices:
            print("No existing PDDL generators found for validation/testing.")
            return
    else:
        for i in range(1, num_games + 1):
            if (not os.path.exists(f"gg_bench/data/pddl_generators/generator_{i}.py") and
                os.path.exists(f"gg_bench/data/descriptions/{i}.txt") and
                os.path.exists(f"gg_bench/data/envs/env_{i}.py")):
                game_indices.append(i)

        if not game_indices:
            print("No games found that need PDDL generator generation.")
            return

        print(f"Generating PDDL generators for {len(game_indices)} games...")

        with tqdm.tqdm(total=len(game_indices), desc="Generating PDDL generators") as pbar:
            for game_idx in game_indices:
                # Load game description
                with open(f"gg_bench/data/descriptions/{game_idx}.txt", "r") as f:
                    game_description = f.read().strip()
                
                # Load corresponding Gym environment code
                gym_env_code = load_gym_environment_code(game_idx)
                if not gym_env_code:
                    print(f"Warning: No Gym environment found for game {game_idx}, skipping...")
                    pbar.update(1)
                    continue

                tasks.append(
                    generate_pddl_generator_and_save(
                        game_idx=game_idx,
                        game_description=game_description,
                        gym_env_code=gym_env_code,
                        prompt=prompt,
                        model=model,
                        max_completion_tokens=max_completion_tokens,
                        usage_tracker=usage_tracker,
                        pbar=pbar,
                    )
                )
            
            await asyncio.gather(*tasks)

        print("All PDDL generators have been created and saved.")
    
    # Validate generator quality
    if validate_quality_flag:
        print("Validating generator quality...")
        quality_reports = []
        for game_idx in game_indices:
            quality_report = await validate_generator_quality(game_idx)
            quality_reports.append(quality_report)
        
        # Save quality report
        with open("gg_bench/data/quality_reports/pddl_generators_quality.json", "w") as f:
            json.dump(quality_reports, f, indent=2)
    
    # Generate test levels if requested
    if generate_test_levels_flag:
        print("Generating test levels...")
        with tqdm.tqdm(total=len(game_indices), desc="Generating test levels") as level_pbar:
            level_tasks = []
            for game_idx in game_indices:
                level_tasks.append(generate_test_levels(game_idx))
            await asyncio.gather(*level_tasks)
            level_pbar.update(len(game_indices))

    print("Usage tracker:")
    print(f"In Tokens: {usage_tracker.in_tokens}")
    print(f"Out Tokens: {usage_tracker.out_tokens}")
    print(f"Cost: {usage_tracker.cost}")

    usage_tracker.save("gg_bench/data/usage_trackers/pddl_generators.yaml")


if __name__ == "__main__":
    asyncio.run(main())