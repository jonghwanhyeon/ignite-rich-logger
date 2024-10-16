from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from ignite.engine import Engine, Events
from ignite.engine.events import CallableEventWithFilter
from ignite.handlers.base_logger import BaseLogger, BaseOutputHandler
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text


def _num_steps(event_name: Events | CallableEventWithFilter, engine: Engine) -> int | None:
    if event_name in (Events.ITERATION_STARTED, Events.ITERATION_COMPLETED):
        return engine.state.epoch_length
    elif event_name in (Events.EPOCH_STARTED, Events.EPOCH_COMPLETED):
        return engine.state.max_epochs

    raise ValueError(f"Event {event_name} is not supported")


@dataclass(frozen=True)
class Metric:
    name: str
    value: float | None

    def make_text(self) -> Text:
        if self.value is None:
            return Text.from_markup(f"[yellow]{self.name}[/yellow]=[i magenta]None[/i magenta]")

        if 10 <= self.value:
            format = ".2f"
        elif 0.0001 <= self.value < 10:
            format = ".4f"
        elif 0.000001 <= self.value < 0.0001:
            format = ".6f"
        else:
            format = "g"

        return Text.from_markup(f"[yellow]{self.name}[/yellow]=[cyan]{self.value:{format}}[/cyan]")


class Metrics(OrderedDict[str, float]):
    def __rich__(self) -> Text:
        return Text(" ").join([Metric(name, value).make_text() for name, value in self.items()])


class MetricsByTag(OrderedDict[str, Metrics]):
    def __getitem__(self, key: str) -> Metrics:
        if key not in self:
            self[key] = Metrics()
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: dict[str, float] | Metrics) -> None:
        if not isinstance(value, Metrics):
            value = Metrics(value)
        return super().__setitem__(key, value)

    def __rich__(self) -> Panel:
        if not self:
            return self._default_panel()

        grid = Table.grid(padding=(0, 2))
        grid.add_column(justify="right")
        grid.add_column()

        for tag, metrics in self.items():
            if not metrics:
                continue
            grid.add_row(f"[bold]{tag}[/bold]", metrics)

        return Panel.fit(grid)

    def _default_panel(self) -> Panel:
        return Panel.fit(Spinner("dots", "Waiting for logs..."))


class ProgressBar(BaseLogger):
    """
    A logger that displays a progress bar and metrics using the Rich library.
    """

    def __init__(self, persist: bool = False) -> None:
        self.live = Live(self)
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            "<",
            TimeRemainingColumn(),
        )
        self.metrics_by_tag = MetricsByTag()

        self.persist = persist

    def log_message(self, message: str) -> None:
        """
        Log a message to the console.

        Args:
            message (str): The message to be logged.
        """
        self.live.console.print(message)

    def attach(  # type: ignore[override]
        self,
        engine: Engine,
        tag: Literal["train", "dev", "test"],
        metric_names: str | list[str] | None = None,
        output_transform: Callable | None = None,
        state_attributes: list[str] | None = None,
        logging_strategy: Literal["epoch", "steps"] = "steps",
        logging_interval: int = 1,
    ) -> None:
        """
        Attach the progress bar handler to the engine.

        Args:
            engine: The engine instance to attach the handler to.
            tag: The tag representing the phase (`"train"`, `"dev"`, or `"test"`).
            metric_names: The names of metrics to log. Use `"all"` to log all metrics.
            output_transform: A function to transform the output.
            state_attributes: `state` attributes to log.
            logging_strategy: The strategy for logging progress, either `"epoch"` or `"steps"`.
            logging_interval: The number of steps or epochs between logging events.
        """
        event = self._event_for(logging_strategy)
        handler = ProgressBarOutputHandler(event, self, tag, metric_names, output_transform, state_attributes)

        engine.add_event_handler(Events.STARTED, handler.on_started)
        engine.add_event_handler(Events.EPOCH_STARTED, handler.on_epoch_started)
        engine.add_event_handler(event, handler.on_step)
        super().attach(engine, handler, event(every=logging_interval))
        engine.add_event_handler(Events.EPOCH_COMPLETED, handler.on_epoch_completed)
        engine.add_event_handler(Events.COMPLETED, handler.on_completed)

    def _event_for(self, logging_strategy: Literal["epoch", "steps"] = "steps") -> Events:
        if logging_strategy == "epoch":
            return Events.EPOCH_COMPLETED
        elif logging_strategy == "steps":
            return Events.ITERATION_COMPLETED
        else:
            raise ValueError(f"Invalid logging strategy: {logging_strategy}")

    def _create_output_handler(self, *args: Any, **kwargs: Any) -> ProgressBarOutputHandler:
        return ProgressBarOutputHandler(*args, **kwargs)

    def _create_opt_params_handler(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        raise NotImplementedError()

    def close(self) -> None:
        self.live.refresh()
        self.live.stop()

    def __rich__(self) -> Group:
        return Group(self.progress, self.metrics_by_tag)


class ProgressBarOutputHandler(BaseOutputHandler):
    def __init__(
        self,
        event: Events | CallableEventWithFilter,
        logger: ProgressBar,
        tag: Literal["train", "dev", "test"],
        metric_names: str | list[str] | None = None,
        output_transform: Callable | None = None,
        state_attributes: list[str] | None = None,
    ) -> None:
        if (metric_names is None) and (output_transform is None):
            metric_names = []

        super().__init__(
            tag,
            metric_names,
            output_transform,
            global_step_transform=None,
            state_attributes=state_attributes,
        )

        self.event = event
        self.logger = logger

        self.task_id: TaskID | None = None

    def on_started(self, engine: Engine) -> None:
        self.logger.live.start()

    def on_epoch_started(self, engine: Engine) -> None:
        if self._is_trainer and (self.task_id is not None):
            self._remove_task()

        self.task_id = self._add_task(engine)

    def on_step(self, engine: Engine) -> None:
        assert self.task_id is not None
        self.logger.progress.advance(self.task_id)

    def __call__(
        self,
        engine: Engine,
        logger: ProgressBar,
        event: str | Events,
    ) -> None:
        self._log_metrics(engine)

    def on_epoch_completed(self, engine: Engine) -> None:
        self._log_metrics(engine)

        if (not self._is_trainer) and (self.task_id is not None):
            self._remove_task()

    def on_completed(self, engine: Engine) -> None:
        should_close = False

        if self._is_trainer:
            if not self.logger.persist:
                self._remove_task()

            should_close = self.logger.progress.task_ids == [self.task_id]

        should_close = should_close or (not self.logger.progress.task_ids)

        if should_close:
            self.logger.close()

    @property
    def _is_trainer(self) -> bool:
        return self.tag == "train"

    def _add_task(self, engine: Engine) -> TaskID:
        if self._is_trainer:
            return self.logger.progress.add_task(
                f"[bold blue]Epoch[/bold blue] {engine.state.epoch}/{engine.state.max_epochs}",
                total=_num_steps(self.event, engine),
            )
        else:
            return self.logger.progress.add_task(
                "[bold blue]Prediction[/bold blue]", total=_num_steps(self.event, engine)
            )

    def _remove_task(self) -> None:
        if self.task_id is not None:
            self.logger.progress.remove_task(self.task_id)
            self.task_id = None

    def _log_metrics(self, engine: Engine) -> None:
        metrics = self._setup_output_metrics_state_attrs(engine)
        for (tag, name), value in metrics.items():
            self.logger.metrics_by_tag[tag][name] = value
