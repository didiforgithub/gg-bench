import subprocess
import os
import shutil
# Constants for PDDL tools
VAL_EXECUTABLE = "/Users/fancy/Desktop/project/autoenv/VAL/build/macos64/Release/bin/validate"  # Path to the PDDL validator executable (VAL)
FAST_DOWNWARD_SCRIPT_PATH = "/Users/fancy/Desktop/project/autoenv/downward/fast-downward.py"  # Path to the Fast Downward planner script


class PDDLValidator:
    def __init__(self, validator_executable=VAL_EXECUTABLE):
        self.validator_executable = validator_executable

    def validate_pddl(self, domain_filepath: str, problem_filepath: str) -> bool:
        print(f"\n--- Validating PDDL files ({self.validator_executable}) ---")
        command = [self.validator_executable, domain_filepath, problem_filepath]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                print("PDDL Validation Successful (according to VAL output pattern)!")
                print("VAL Output:\n", result.stdout)
                return True
            elif "Problem an Domain files validated successfully" in result.stdout:
                print("PDDL Validation Successful (according to VAL output pattern)!")
                print("VAL Output:\n", result.stdout)
                return True
            else:
                print("PDDL Validation Failed or Validator output indicates issues.")
                print("Return Code:", result.returncode)
                print("STDOUT:\n", result.stdout)
                print("STDERR:\n", result.stderr)
                return False

        except FileNotFoundError:
            print(f"Error: '{self.validator_executable}' command not found.")
            print("Please ensure VAL (or your chosen validator) is installed and in your PATH, or update VAL_EXECUTABLE.")
            return False
        except Exception as e:
            print(f"An error occurred during PDDL validation: {e}")
            return False

class PDDLPlanner:
    def __init__(self, planner_script_path=FAST_DOWNWARD_SCRIPT_PATH):
        self.planner_script_path = planner_script_path
        if not (os.path.isfile(self.planner_script_path) and os.access(self.planner_script_path, os.X_OK)):
            # If it's not a direct file path, check if it's in PATH
            if not shutil.which(self.planner_script_path):
                print(f"Warning: Planner script '{self.planner_script_path}' not found, not executable, or not in PATH.")
                self.is_configured = False
            else:
                self.is_configured = True # Found in PATH
        else:
            self.is_configured = True # Direct file path is valid

    def generate_plan(self, domain_filepath: str, problem_filepath: str,
                      plan_output_file="sas_plan") -> list[str] | None:
        if not self.is_configured:
            print("Skipping plan generation as Fast Downward planner is not configured correctly.")
            return None

        print(f"\n--- Generating plan using Fast Downward ({self.planner_script_path}) ---")
        
        # A common satisficing (non-optimal, but fast) configuration for Fast Downward
        # You might need to adjust python interpreter if fast-downward.py is a python script
        # and your system needs specific python version e.g. ['python3', self.planner_script_path, ...]
        command = [
            self.planner_script_path,
            domain_filepath,
            problem_filepath,
            "--search", "lazy_greedy([ff()], preferred=[ff()])" # A good general purpose planner
            # For optimal: "--search", "astar(lmcut())"
        ]
        
        # Clean up previous plan file if it exists
        if os.path.exists(plan_output_file):
            os.remove(plan_output_file)
        # Clean up other common FD output files
        for fd_out_file in ["output", "output.sas"]:
            if os.path.exists(fd_out_file):
                os.remove(fd_out_file)

        try:
            print(f"Executing command: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60) # 60s timeout

            print("Planner STDOUT:")
            print(result.stdout)
            print("Planner STDERR:")
            print(result.stderr)

            if os.path.exists(plan_output_file):
                print(f"Plan found! Output written to {plan_output_file}")
                with open(plan_output_file, 'r') as f:
                    plan = [line.strip() for line in f if line.strip() and not line.startswith(';')]
                
                # Clean up after successful run
                if os.path.exists(plan_output_file): os.remove(plan_output_file)
                if os.path.exists("output"): os.remove("output")
                if os.path.exists("output.sas"): os.remove("output.sas")
                if not plan: # File existed but was empty or only comments
                    print("Plan file was empty or contained no valid actions.")
                    return None
                return plan
            else:
                print("No plan file (sas_plan) was generated by Fast Downward.")
                if "Solution found." in result.stdout and result.returncode == 0:
                     print("Planner reported solution found, but sas_plan is missing. Check planner configuration/version.")
                elif "Task is provably unsolvable." in result.stdout or "Goal can be simplified to false." in result.stdout :
                    print("Planner reported task is unsolvable.")
                elif result.returncode != 0 :
                    print(f"Planner exited with error code: {result.returncode}")
                return None

        except FileNotFoundError:
            print(f"Error: Planner script '{self.planner_script_path}' not found.")
            print("Please ensure Fast Downward is installed and FAST_DOWNWARD_SCRIPT_PATH is set correctly.")
            return None
        except subprocess.TimeoutExpired:
            print(f"Error: Planner timed out after 60 seconds.")
            return None
        except Exception as e:
            print(f"An error occurred during plan generation: {e}")
            return None
