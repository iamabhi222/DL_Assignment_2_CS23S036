import torch
from util_PART_A import prepare_data, CNN
import matplotlib.pyplot as plt
import numpy as np
import wandb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Function to make predictions using the trained model
def predict():
    # Prepare testing data
    _, test_loader ,_ = prepare_data(augmentation=False, batch_size=128)

    # Initialize and load trained model
    model = CNN(n_filters=32, filter_multiplier=2, dropout=0.5, batch_norm=False, dense_size=64, act_func='relu')
    model.to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # Lists to store images and their corresponding true and predicted labels
    sample_images = []
    true_labels = []
    predicted_labels = []

    # Counter to keep track of the number of images per class
    class_counts = {i: 0 for i in range(10)}
    label_names = {
        0: "Amphibia",
        1: "Animalia",
        2: "Arachnida",
        3: "Aves",
        4: "Fungi",
        5: "Insecta",
        6: "Mammalia",
        7: "Mollusca",
        8: "Plantae",
        9: "Reptilia"
    }

    # Perform inference on test data 
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
#----------------------------------------------------------------------------------------------------
            for image, true_label, predicted_label in zip(images, labels, predicted):
                if class_counts[true_label.item()] < 3:  # Select only 3 images per class
                    sample_images.append(np.transpose(image.cpu().numpy(), (1, 2, 0)))  # Transpose image from CHW to HWC format
                    true_labels.append(true_label.item())
                    predicted_labels.append(predicted_label.item())
                    class_counts[true_label.item()] += 1                
#----------------------------------------------------------------------------------------------------
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Create a 10x3 grid of sample images with labels
    fig, axes = plt.subplots(10, 3, figsize=(15, 30))
    for i in range(10):
        for j in range(3):
            index = i * 3 + j
            axes[i, j].imshow(sample_images[index])
            axes[i, j].set_title(f'True Label: {label_names[true_labels[index]]}\nPredicted Label: {label_names[predicted_labels[index]]}')
            axes[i, j].axis('off')

    # Log the grid of sample images to wandb
    wandb.log({"Sample Images": wandb.Image(plt)})

    # # Combine images and labels into a single plot and log to wandb
    # for i in range(len(sample_images)):
    #     # Create a combined plot of image and label
    #     combined_plot = wandb.Image(sample_images[i], caption=f'True Label: {label_names[true_labels[i]]}, Predicted Label: {label_names[predicted_labels[i]]}')
        
    #     # Log the combined plot to wandb
    #     wandb.log({"Sample Image": combined_plot})

    # Print accuracy of the model
    print('Test Accuracy: %d %%' % (100 * correct / total))

if __name__ == "__main__":
    wandb.init(project="DL-Assignment-2", entity="cs23s036")
    predict()



