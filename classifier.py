import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random


def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def generate_dataset(train_size=5000, validation_size=1000):
    # label = 0
    # x1 = r * cos(t)
    # x2 = r * sin(t)
    # label = 1
    # x1 = (r + 5) * cos(t)
    # x2 = (r + 5) * sin(t)
    for i in range(train_size + validation_size):
        r = np.random.randn()
        t = np.random.uniform(0, 2 * np.pi)
        if i < train_size:
            with open("train.txt", "a") as f:
                f.write(f"{r * np.cos(t)} {r * np.sin(t)} 0\n")
        else:
            with open("validation.txt", "a") as f:
                f.write(f"{r * np.cos(t)} {r * np.sin(t)} 0\n")

    for i in range(train_size + validation_size):
        r = np.random.randn()
        t = np.random.uniform(0, 2 * np.pi)
        if i < train_size:
            with open("train.txt", "a") as f:
                f.write(f"{(r + 5) * np.cos(t)} {(r + 5) * np.sin(t)} 1\n")
        else:
            with open("validation.txt", "a") as f:
                f.write(f"{(r + 5) * np.cos(t)} {(r + 5) * np.sin(t)} 1\n")


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.dataset = []
        with open(file_path, "r") as f:
            for line in f:
                x1, x2, y = line.split()
                self.dataset.append([float(x1), float(x2), int(y)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return torch.tensor([data[0], data[1]], dtype=torch.float), torch.tensor([data[2]], dtype=torch.float)


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train():
    train_dataset = CustomDataset("train.txt")
    validation_dataset = CustomDataset("validation.txt")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False)
    model = MLP(2, 30, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_val_loss = np.inf
    no_improvement = 0
    train_losses = []
    val_losses = []
    for epoch in range(100):
        train_loss = 0
        val_loss = 0
        for i, batch in enumerate(train_dataloader):
            x, y = batch
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        with torch.no_grad():
            for i, batch in enumerate(validation_dataloader):
                x, y = batch
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()
        train_mean_loss = train_loss / len(train_dataset)
        val_mean_loss = val_loss / len(validation_dataset)
        train_losses.append(train_mean_loss)
        val_losses.append(val_mean_loss)
        print(f"Epoch: {epoch}, Train Loss: {train_mean_loss}, Validation Loss: {val_mean_loss}")
        if val_mean_loss < best_val_loss:
            no_improvement = 0
            best_val_loss = val_mean_loss
            torch.save(model.state_dict(), f"checkpoints/model_{epoch}.pt")
        else:
            no_improvement += 1
        if no_improvement >= 5:
            print("Early stopping at epoch", epoch)
            break

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig("training_validation_loss_task2.pdf", dpi=300)
    plt.show()


def plot_validation_and_model():
    # Plot the validation data
    validation_dataset = CustomDataset("validation.txt")
    x = [x for x in validation_dataset]
    x1_0 = [x[0][0] for x in x if x[1] == 0]
    x2_0 = [x[0][1] for x in x if x[1] == 0]
    x1_1 = [x[0][0] for x in x if x[1] == 1]
    x2_1 = [x[0][1] for x in x if x[1] == 1]
    color = ["blue", "orange"]
    shape = ["o", "x"]
    for i in range(2):
        plt.scatter(x1_0 if i == 0 else x1_1, x2_0 if i == 0 else x2_1, c=color[i], marker=shape[i], label=f"Class {i}")
    # Plot the model
    model = MLP(2, 30, 1)
    model.load_state_dict(torch.load("checkpoints/model_22.pt"))
    # Activation function is relu, so it is the line that separates the two classes
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            Z[i, j] = model(torch.tensor([X[i, j], Y[i, j]], dtype=torch.float)).item()
    plt.contour(X, Y, Z, levels=[0])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Validation Data and Decision Boundary")
    plt.savefig("validation_data_and_decision_boundary.pdf", dpi=300)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    set_seed(123)
    # generate_dataset()
    # train()
    plot_validation_and_model()