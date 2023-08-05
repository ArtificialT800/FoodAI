from functions import plot_loss_curves


##Building the Vision Model
class CNNModel(nn.Module):
  def __init__(self, input_units, output_units, hidden_units):
    super().__init__()
    self.block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_units, out_channels=hidden_units, kernel_size=2, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=2, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=1)
    )

    self.block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=2, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=2, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=1)
    )
    self.block_3 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=2, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=2, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=1)
    )

    self.block_4 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=2, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=2, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=1)
    )
    self.block_5 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=2, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=2, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=1)
    )


    self.Classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*229*229, out_features=output_units)

    )

  def forward(self, x):
    return self.Classifier(self.block_5(self.block_4(self.block_3(self.block_2(self.block_1(x))))))

model = CNNModel(input_units=3, hidden_units=16, output_units=len(class_names))
model = model.to(device)

#Loss function

loss_fn = nn.CrossEntropyLoss()

#Optimizer
"""
You could always experiment with other optimizers too, like SGD or Adagrad!

More optimizers at -> "https://pytorch.org/docs/stable/optim.html#algorithms"

"""
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=0.001)

##Training and testing Loop

torch.manual_seed(42)
torch.cuda.manual_seed(42)

start = timer()

EPOCHS = 150

results = {"train_loss": [],
          "train_acc": [],
          "test_loss": [],
          "test_acc": []
          }

for epoch in tqdm(range(EPOCHS)):
  #Training

  model.train()
  train_loss, train_acc = 0, 0
  for batch, (X, y) in enumerate(train_dataloader):
    X, y = X.to(device), y.to(device)

    y_pred = model(X)

    loss = loss_fn(y_pred, y)
    train_loss += loss

    acc = accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
    train_acc += acc

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

  train_loss /= len(train_dataloader)
  train_acc /= len(train_dataloader)

  #Testing
  model.eval()
  with torch.inference_mode():
    total_test_loss, total_test_acc = 0, 0
    for batch, (X_test, y_test) in enumerate(test_dataloader):
      X_test, y_test = X_test.to(device), y_test.to(device)

      test_pred = model(X_test)

      test_loss = loss_fn(test_pred, y_test)
      total_test_loss += test_loss

      test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))
      total_test_acc += test_acc

    total_test_loss /= len(test_dataloader)
    total_test_acc /= len(test_dataloader)

    results["train_loss"].append(train_loss.to("cpu"))
    results["train_acc"].append(train_acc)
    results["test_loss"].append(total_test_loss.to("cpu"))
    results["test_acc"].append(total_test_acc)

  print(f"Epochs: {epoch}| Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}% | Test Loss: {total_test_loss:.4f}, Test Accuracy: {total_test_acc:.4f}%")

if int(total_test_acc) >= 85:
  print(f"Test Accuracy is pretty good..\nCurrent Test Accuracy is: {total_test_acc:.3f}%")
  torch.save(obj=model.state_dict(), f='model.pt')

elif int(train_acc) >= 95:
  torch.save(obj=model.state_dict(), f='train_model.pt')
  print("Train Accuracy is good.. \nCurrent Test Accuracy is: {total_test_acc:.3f}%")

else:
  print("Accuracy isn't good enough...")
  print(f"Current Test Accuracy is: {total_test_acc:.3f}%")

plot_loss_curves(results=results)
