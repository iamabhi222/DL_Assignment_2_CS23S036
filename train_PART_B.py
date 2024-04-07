import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse
from util_PART_B import modified_model
from util_PART_A import prepare_data
from wandb_config_PART_B import train_wandb
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sweep_config = {
    "name": "ResNet50 Sweep(Bayesian)",
    "description": "Tuning hyperparameters",
    'metric': {
      'name': 'val_categorical_accuracy',
      'goal': 'maximize'
  },
    "method": "bayes",
    "parameters": {
        "augment_data": {
            "values": [True, False]
        },
        "epochs": {
            "values": [5, 7, 10]
        },
        "lr": {
            "values": [0.01, 0.001, 0.0001]
        },
        "batch_size": {
            "values": [16, 32]
        }
    }
}

if os.path.exists('best_model.pth'):
    os.remove('best_model.pth')
    
# Function to train the CNN model
def train(config):
    wandb.init(project="DL-Assignment-2", entity="cs23s036", config=config)

    # Prepare training and testing data
    train_loader ,_, val_loader  = prepare_data(augmentation=config.augmentation, batch_size=config.batch_size)

    # Initialize the CNN model
    model = modified_model()
    model.to(device)    # gpu acc
    criterion = nn.CrossEntropyLoss()  # Define cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=config.lr)  # Initialize Adam optimizer

    # Training loop
    for epoch in range(config.epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for _, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()   # clears the gradient

            outputs = model(inputs) # runs one forward pass
            loss = criterion(outputs, labels)   # compute the loss based on cross-entropy
            loss.backward()     # backprop
            optimizer.step()

            running_loss += loss.item()

        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for _, data in enumerate(val_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs) # runs one forward pass

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)   # compute the loss based on cross-entropy

                val_loss += loss.item()        

        val_categorical_accuracy = (100 * correct / total)
        wandb.log({"epoch": epoch + 1,
                   "Training loss": running_loss / len(train_loader),
                   "Validation loss": val_loss / len(val_loader),
                   "val_categorical_accuracy": val_categorical_accuracy})
        print(f"Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}, val_categorical_accuracy: {val_categorical_accuracy}%")


    print('----------Finished Training-----------')

    # Save trained model
    torch.save(model.state_dict(), 'best_model.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet50 CNN')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--augmentation', type=bool, default=False, help='Data Augmentation')
    parser.add_argument('--epochs', type=int, default=7, help='Number of epochs')
    args = parser.parse_args()

    # train(args)   #------ Train the best model ------


    # creating the sweep
    sweep_id = wandb.sweep(sweep_config, project="custom", entity="-------wandb ID-------")
    wandb.agent(sweep_id, train_wandb, count=10)






