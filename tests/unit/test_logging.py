import csv
import os

import torch

from utils.logging import MetricsRecorder, format_duration, setup_run_logger


def test_format_duration_seconds_only():
    assert format_duration(45) == "45s"


def test_format_duration_minutes_and_seconds():
    assert format_duration(65) == "1m 5s"


def test_format_duration_hours_minutes_seconds():
    assert format_duration(3661) == "1h 1m 1s"


def test_setup_run_logger_creates_dir_and_log_file(tmp_path, cleanup_logging_handlers):
    logger, run_dir = setup_run_logger(str(tmp_path), name="testrun")
    cleanup_logging_handlers(logger)

    assert os.path.isdir(run_dir)
    assert str(tmp_path) in run_dir

    logger.info("hello from test")
    for handler in logger.handlers:
        handler.flush()

    log_path = os.path.join(run_dir, "train.log")
    assert os.path.exists(log_path)
    with open(log_path) as f:
        contents = f.read()
    assert "hello from test" in contents


def test_metrics_recorder_writes_header_and_rows(tmp_path):
    class_names = ["a", "b", "c"]
    recorder = MetricsRecorder(str(tmp_path), class_names)
    assert os.path.exists(recorder.path)

    recorder.log_epoch(
        epoch=1, train_loss=0.5, val_balanced_accuracy=0.6, lr=1e-4,
        epoch_time_sec=1.23, dice=torch.tensor([0.1, 0.2, 0.3]), gpu_mem_mb=None,
    )

    with open(recorder.path, newline="") as f:
        rows = list(csv.reader(f))

    header, data_row = rows[0], rows[1]
    assert header == ["epoch", "train_loss", "val_balanced_accuracy", "lr",
                       "epoch_time_sec", "gpu_mem_mb", "dice_a", "dice_b", "dice_c"]
    assert len(data_row) == len(header)
    assert data_row[0] == "1"


def test_metrics_recorder_appends_multiple_epochs(tmp_path):
    recorder = MetricsRecorder(str(tmp_path), ["a"])
    for epoch in range(3):
        recorder.log_epoch(
            epoch=epoch, train_loss=0.1, val_balanced_accuracy=0.2, lr=1e-3,
            epoch_time_sec=0.5, dice=torch.tensor([0.9]), gpu_mem_mb=123.4,
        )
    with open(recorder.path, newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 1 + 3  # header + 3 epochs
