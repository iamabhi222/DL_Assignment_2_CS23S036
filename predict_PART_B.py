import torch
from util_PART_A import prepare_data
from util_PART_B import modified_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Function to make predictions using the trained model
def predict():
    # Prepare testing data
    _, test_loader ,_ = prepare_data(augmentation=False, batch_size=64)

    # Initialize and load trained model
    model = modified_model()
    model.to(device)
    model.load_state_dict(torch.load('best_model.pth'), strict=False)
    model.eval()

    # Perform inference on test data
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print accuracy of the model
    print('Test Accuracy: %d %%' % (100 * correct / total))

if __name__ == "__main__":
    predict()
