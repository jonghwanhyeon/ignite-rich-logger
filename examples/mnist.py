import torch
from ignite.engine import Engine, Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Accuracy, Loss
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, ToTensor

from ignite_rich_logger import ProgressBar

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.model = resnet18(num_classes=10)

        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        return self.model(input_values)


model = Net().to(device)

transform_data = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

train_dataloader = DataLoader(
    MNIST(download=True, root=".", transform=transform_data, train=True), batch_size=128, shuffle=True
)

test_dataloader = DataLoader(
    MNIST(download=True, root=".", transform=transform_data, train=False), batch_size=256, shuffle=False
)

optimizer = torch.optim.RMSprop(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

metrics = {"accuracy": Accuracy(), "loss": Loss(criterion)}

trainer = create_supervised_trainer(model, optimizer, criterion, device)
train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
test_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)


@trainer.on(Events.ITERATION_COMPLETED)
def update_train_metrics(trainer: Engine) -> None:
    metrics = {"loss": trainer.state.output}

    if len(optimizer.param_groups) == 1:
        metrics["lr"] = optimizer.param_groups[0]["lr"]
    else:
        for index, param_group in enumerate(optimizer.param_groups):
            metrics[f"lr_{index}"] = param_group["lr"]

    trainer.state.metrics = metrics


@trainer.on(Events.EPOCH_COMPLETED)
def evaluate_model(trainer: Engine) -> None:
    test_evaluator.run(test_dataloader)


progress = ProgressBar()
progress.attach(
    trainer,
    tag="train",
    metric_names="all",
)
progress.attach(train_evaluator, tag="train", metric_names="all")
progress.attach(test_evaluator, tag="test", metric_names="all")

trainer.run(train_dataloader, max_epochs=5)
