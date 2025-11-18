import torch

# Load file (modify according to actual path)
data = torch.load('model_best.pth', map_location=torch.device('cpu'),weights_only=False)  # Force loading with CPU

# Check data type
print(f"File type: {type(data)}")

# If it's a state dictionary (most common case)
if isinstance(data, dict):
    print("\nState dictionary keys:")
    for key in data.keys():
        print(f"- {key} (shape: {data[key].shape if hasattr(data[key], 'shape') else 'non-tensor'})")

# If it's a complete model
elif isinstance(data, torch.nn.Module):
    print("\nModel structure:")
    print(data)

# Other data types
else:
    print("\nContent summary:")
    print(data)