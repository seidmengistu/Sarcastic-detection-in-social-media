try:
    import numpy as np
except ImportError:
    print("Error: NumPy is not installed. Please install it using: pip install numpy")
    raise

def run():
    try:
        print("Running sample model...")
        # Test numpy functionality
        test_array = np.array([1, 2, 3])
        print("NumPy is working correctly!")
        # Add your model's training or inference code here
    except Exception as e:
        print(f"Error in sample model: {str(e)}") 