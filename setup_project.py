import os
from pathlib import Path

def create_project_structure():
    """Create the necessary project directories."""
    # We create the scripts folder as well, just in case
    directories = ['data/Train', 'data/Test', 'models', 'scripts']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    print("Setting up project structure...")
    create_project_structure()
    print("Project structure created successfully!")
