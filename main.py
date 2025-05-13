import os
import importlib.util
from typing import List
from utils.config import Config
from utils.preprocessing import process_dataset

def get_available_models() -> List[str]:
    """Get list of available models."""
    try:
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        return [f[:-3] for f in os.listdir(models_dir) 
                if f.endswith('.py') and f not in ['__init__.py', '__pycache__']]
    except Exception as e:
        print(f"Error getting available models: {str(e)}")
        return []

def load_and_run_model(model_name: str) -> bool:
    """Load and run selected model."""
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'models', f'{model_name}.py')
        if not os.path.exists(model_path):
            print(f"Model file {model_name}.py not found")
            return False
            
        spec = importlib.util.spec_from_file_location(model_name, model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'run'):
            return module.run()
        print(f"Error: {model_name}.py must have a 'run' function")
        return False
            
    except Exception as e:
        print(f"Error running model {model_name}: {str(e)}")
        return False

def main():
    try:
        if not os.path.exists(Config.PREPROCESSED_DATA_PATH):
            print("Preprocessing dataset...")
            process_dataset()
        
        available_models = get_available_models()
        print(available_models)
        if not available_models:
            print("No models found in the models directory.")
            return
        
        print("\nAvailable models:")
        for i, model in enumerate(available_models, 1):
            print(f"{i}. {model}")
        
        while True:
            try:
                choice = int(input("\nSelect a model number to run (or 0 to exit): "))
                if choice == 0:
                    return
                if 1 <= choice <= len(available_models):
                    selected_model = available_models[choice - 1]
                    print(f"\nRunning model: {selected_model}")
                    load_and_run_model(selected_model)
                    break
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
                
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
