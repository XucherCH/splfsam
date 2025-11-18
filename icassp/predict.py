import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms
#from model.train_utils.sam import SAM

from splfsam import getmodel

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def process_image(model, device, img_path, data_transform, threshold=0.5):
    assert os.path.exists(img_path), f"image file {img_path} dose not exists."
    
    origin_img = cv2.cvtColor(cv2.imread(img_path, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    h, w = origin_img.shape[:2]
    
    img = data_transform(origin_img)
    img = torch.unsqueeze(img, 0).to(device)  # [C, H, W] -> [1, C, H, W]

    with torch.no_grad():
        pred = model(img)[-1]
        pred = torch.squeeze(pred).to("cpu").numpy()  # [1, 1, H, W] -> [H, W]
        pred = cv2.resize(pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        pred_mask = np.where(pred > threshold, 255, 0).astype(np.uint8)  # Binary mask (0 and 255)
        
    return pred_mask

def main():
    weights_path = "model_best.pth"
    input_folder = "dataset/images/test"  # Folder containing images to process
    output_folder = "freadapter_output"  # Folder to save results
    threshold = 0.5

    os.makedirs(output_folder, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([512,512]),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    model = getmodel()
    weights = torch.load(weights_path, map_location='cpu', weights_only=False)
    if "model" in weights:
        model.load_state_dict(weights["model"])
    else:
        model.load_state_dict(weights)
    model.to(device)
    model.eval()

    # Initialize model
    img_height, img_width = 512, 512
    init_img = torch.zeros((1, 3, img_height, img_width), device=device)
    model(init_img)

    # Process all images in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    total_time = 0
    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        
        t_start = time_synchronized()
        pred_mask = process_image(model, device, img_path, data_transform, threshold)
        t_end = time_synchronized()
        
        process_time = t_end - t_start
        total_time += process_time
        print(f"Processed {img_file} in {process_time:.4f} seconds")
        
        # Save binary saliency map
        output_path = os.path.join(output_folder, f"mask_{os.path.splitext(img_file)[0]}.png")
        cv2.imwrite(output_path, pred_mask)
        
        # Optional: Display the result
        '''
        plt.imshow(pred_mask, cmap='gray')
        plt.title(img_file)
        plt.show()
        '''

    print(f"\nProcessed {len(image_files)} images in total")
    print(f"Total processing time: {total_time:.4f} seconds")
    print(f"Average time per image: {total_time/len(image_files):.4f} seconds")

if __name__ == '__main__':
    main()