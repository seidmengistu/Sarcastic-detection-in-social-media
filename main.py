# main.py
# Entry point for training or inference

import os
import importlib.util
from typing import List

def get_available_models() -> List[str]:
    """Get list of available model files from the models directory."""
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print("Created models directory. Please add your model files there.")
        return []
    
    model_files = [f[:-3] for f in os.listdir(models_dir) 
                  if f.endswith('.py') and f != '__init__.py']
    return model_files

def load_and_run_model(model_name: str):
    """Load and run the selected model."""
    model_path = os.path.join(os.path.dirname(__file__), 'models', f'{model_name}.py')
    
    try:
        spec = importlib.util.spec_from_file_location(model_name, model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'run'):
            module.run()
        else:
            print(f"Error: {model_name}.py must have a 'run' function")
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")

if __name__ == '__main__':
    available_models = get_available_models()
    
    if not available_models:
        print("No models found in the models directory.")
        exit()
    
    print("Available models:")
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model}")
    
    while True:
        try:
            choice = int(input("\nSelect a model number to run (or 0 to exit): "))
            if choice == 0:
                break
            if 1 <= choice <= len(available_models):
                selected_model = available_models[choice - 1]
                print(f"\nRunning model: {selected_model}")
                load_and_run_model(selected_model)
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
