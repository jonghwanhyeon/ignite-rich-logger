import time

import torch
import torch.nn as nn
from ignite.engine import Engine, Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Loss, MeanAbsoluteError
from torch.utils.data import DataLoader, TensorDataset

from ignite_rich_logger import ProgressBar


def main() -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.linear = nn.Linear(16, 1)

        def forward(self, input_values: torch.Tensor) -> torch.Tensor:
            hidden_states = self.linear(input_values)
            time.sleep(0.075)  # slow inference
            return torch.sigmoid(hidden_states)

    model = Model()

    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.005)
    criterion = nn.L1Loss()
    metrics = {"mae": MeanAbsoluteError(), "loss": Loss(criterion)}

    train_dataset = TensorDataset(torch.randn((64, 16)), torch.rand((64, 1)))
    dev_dataset = TensorDataset(torch.randn((16, 16)), torch.rand((16, 1)))
    test_dataset = TensorDataset(torch.randn((32, 16)), torch.rand((32, 1)))

    train_dataloader = DataLoader(train_dataset, batch_size=2)
    dev_dataloader = DataLoader(dev_dataset, batch_size=2)
    test_dataloader = DataLoader(test_dataset, batch_size=2)

    trainer = create_supervised_trainer(model, optimizer, criterion, device="cpu")
    dev_evaluator = create_supervised_evaluator(model, metrics=metrics, device="cpu")
    test_evaluator = create_supervised_evaluator(model, metrics=metrics, device="cpu")

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
    def evaluate_model_using_dev_dataset(trainer: Engine) -> None:
        dev_evaluator.run(dev_dataloader)

    @trainer.on(Events.COMPLETED)
    def evaluate_model_test_dataset(trainer: Engine) -> None:
        test_evaluator.run(test_dataloader)

    progress = ProgressBar()
    progress.attach(trainer, tag="train", metric_names="all")
    progress.attach(dev_evaluator, tag="dev", metric_names="all")
    progress.attach(test_evaluator, tag="test", metric_names="all")

    trainer.run(train_dataloader, max_epochs=4)


if __name__ == "__main__":
    main()
