import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import tenacity
import sys

# Load environment variables from .env (mounted at /app/.env) --> only works with a github PAT
# load_dotenv("/app/.env")
# DAGSHUB_USERNAME = os.getenv('DAGSHUB_USERNAME')
# DAGSHUB_KEY = os.getenv('DAGSHUB_KEY')
# DAGSHUB_REPO = os.getenv('DAGSHUB_REPO')
# CLONE_DIR = Path(os.getenv('CLONE_DIR', '/app'))

# Workaround locally
# Dynamic repo root (parent of src/)
REPO_ROOT = Path(__file__).parent.parent
# Load environment variables from .env in repo root
load_dotenv(REPO_ROOT / ".env")
# Use CLONE_DIR from .env if set, else default to dynamic root
CLONE_DIR = Path(os.getenv('CLONE_DIR', REPO_ROOT))

def get_current_branch():
    try:
        # This returns "HEAD" in a detached head state, so check for that.
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        branch = result.stdout.strip()
        return branch if branch != 'HEAD' else None
    except Exception:
        return None

BRANCH_NAME = os.getenv('GIT_BRANCH') or get_current_branch() or 'main'


def check_files_exist():
    """Check if production_model.keras and models/metadata.txt exist."""
    model_path = CLONE_DIR / "model" / "production_model.keras"  # keep the prod_model in the root (current state)
    metadata_path = CLONE_DIR / "models" / "metadata.txt"
    
    model_exists = model_path.exists()
    metadata_exists = metadata_path.exists()
    
    print(f"Checking files: model exists={model_exists}, metadata exists={metadata_exists}")
    return model_exists, metadata_exists

def run_command(cmd, cwd=CLONE_DIR, check=False):
    """Run a command and handle errors consistently."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            check=check, 
            capture_output=True, 
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        if check:
            raise
        return None

@tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10), stop=tenacity.stop_after_attempt(3))
def git_push():
    """Try to push to Git with retry logic."""
    result = run_command(['git', 'push', 'origin', BRANCH_NAME])
    if result and result.returncode != 0:
        print(f"Git push failed: {result.stderr}")
        # Check if it's a non-fast-forward error
        if "non-fast-forward" in result.stderr or "fetch first" in result.stderr:
            print("Detected push conflict. Attempting selective merge of our files...")
            # We need to fetch first
            run_command(['git', 'fetch', 'origin', BRANCH_NAME], check=True)
            # Get the list of files we care about
            files_to_checkout = []
            dvc_file = CLONE_DIR / "models" / "production_model.keras.dvc"
            metadata_path = CLONE_DIR / "models" / "metadata.txt"
            if dvc_file.exists():
                files_to_checkout.append(str(dvc_file))
            if metadata_path.exists():
                files_to_checkout.append(str(metadata_path))
            
            # Stash our changes
            run_command(['git', 'stash'], check=True)
            
            # Checkout the specific files from origin
            for file in files_to_checkout:
                relative_path = Path(file).relative_to(CLONE_DIR)
                run_command(['git', 'checkout', f'origin/{BRANCH_NAME}', '--', str(relative_path)], check=True)
            
            # Pop our stash
            run_command(['git', 'stash', 'pop'], check=False)  # Don't check=True as it might fail if there are conflicts
            
            # Stage our files again
            run_command(['git', 'add'] + files_to_checkout, check=True)
            
            # Commit with the same message
            commit_msg = "Update model and metadata"
            run_command(['git', 'commit', '-m', commit_msg], check=False)
            
            # Try to push again - this time let it fail if there's a problem
            result = run_command(['git', 'push', 'origin', BRANCH_NAME], check=True)
        else:
            # If it's another kind of error, just raise it
            raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
    print("Pushed Git changes successfully ✅")

@tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10), stop=tenacity.stop_after_attempt(3))
def dvc_push():
    """Retry DVC push up to 3 times with exponential backoff."""
    result = run_command(["dvc", "push"], check=False)
    if result and result.returncode != 0:
        print(f"DVC push failed: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
    print("Pushed model to DVC remote ✅")

def main():
    """Stage and push changes to GitHub and DVC/DagsHub from host."""
    print("Starting git_dvc_update.py")
    print(f"Using repository root: {CLONE_DIR}")
    print(f"Using branch: {BRANCH_NAME}")
    
    try:
        # Check if we're in a Git repository
        result = run_command(['git', 'status'], check=False)
        if result and result.returncode != 0:
            print("Not in a Git repository. Exiting.")
            sys.exit(1)
        
        # Check if remote exists
        result = run_command(['git', 'remote', '-v'], check=False)
        if result and 'origin' not in result.stdout:
            print("No 'origin' remote found. Exiting.")
            sys.exit(1)
        
        # Check that the branch exists
        result = run_command(['git', 'branch', '--list', BRANCH_NAME], check=False)
        if result and BRANCH_NAME not in result.stdout:
            print(f"Branch {BRANCH_NAME} doesn't exist locally. Exiting.")
            sys.exit(1)
            
        # Check file existence
        model_exists, metadata_exists = check_files_exist()
        if not model_exists and not metadata_exists:
            print("No model or metadata files found, nothing to push")
            return
        
        # Stage model with DVC if it exists
        model_path = CLONE_DIR / "models" / "production_model.keras"
        dvc_file = CLONE_DIR / "models" / "production_model.keras.dvc"
        if model_exists:
            run_command(['dvc', 'add', str(model_path)], check=True)
            print(f"Added {model_path} to DVC")
        
        # Stage files with Git
        metadata_path = CLONE_DIR / "models" / "metadata.txt"
        files_to_add = []
        if dvc_file.exists():
            files_to_add.append(str(dvc_file))
        if metadata_exists:
            files_to_add.append(str(metadata_path))
        
        if files_to_add:
            run_command(['git', 'add'] + files_to_add, check=True)
            print(f"Staged files: {files_to_add}")
            
            # Commit changes
            commit_msg = "Update model and metadata"
            result = run_command(['git', 'commit', '-m', commit_msg], check=False)
            if result and result.returncode == 0:
                print("Committed changes")
            else:
                print("No changes to commit or commit failed")
                if "nothing to commit" in (result.stdout + result.stderr):
                    print("No changes to files detected")
                    return
            
            # Push to GitHub with retry and conflict handling
            git_push()
        
        # Push to DVC (relies on host config)
        if model_exists:
            dvc_push()
        
        print("Successfully completed git_dvc_update.py")
    
    except Exception as e:
        print(f"Error in git_dvc_update.py: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()