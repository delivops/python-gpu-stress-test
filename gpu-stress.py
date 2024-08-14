import argparse
import logging
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim


def setup_logger():
    """Set up a logger with a specific format."""
    logger = logging.getLogger("GPUStressTest")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="GPU Stress Test Script")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Preferred device (default: cuda)",
    )
    return parser.parse_args()


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def gpu_stress_test(preferred_device: str = "cuda", logger=None):
    """Perform a GPU stress test using a simple model."""
    cuda_available = torch.cuda.is_available()

    if logger is None:
        print("Logger is not provided. Exiting...")
        sys.exit(1)

    if preferred_device == "cpu" and cuda_available:
        logger.warning("CUDA is available, but CPU is requested. Using CPU.")

    if preferred_device == "cuda" and not cuda_available:
        logger.error("CUDA is requested but not available. Exiting...")
        sys.exit(1)

    device = torch.device(
        preferred_device if cuda_available or preferred_device == "cpu" else "cpu"
    )

    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    model = SimpleModel().to(device)
    input_data = torch.randn(64, 3, 128, 128).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    logger.info("Starting stress test...")

    loop_counter = 0
    log_interval = 5  # Log every 5 seconds
    start_time = time.time()
    last_log_time = start_time

    with open("perf.log", "w") as perf_log:
        try:
            while True:
                optimizer.zero_grad()
                output = model(input_data)
                target = torch.randint(0, 10, (64,)).to(device)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                logger.info(f"Loss: {loss.item():.4f}")

                loop_counter += 1

                current_time = time.time()
                if current_time - last_log_time >= log_interval:
                    elapsed_time = current_time - start_time
                    perf_log.write(
                        f"Loop count: {loop_counter}, Total elapsed time: {elapsed_time:.2f} seconds\n"
                    )
                    perf_log.flush()
                    last_log_time = current_time

        except KeyboardInterrupt:
            logger.info("Stress test interrupted by user.")


if __name__ == "__main__":
    logger = setup_logger()
    args = parse_arguments()
    gpu_stress_test(preferred_device=args.device, logger=logger)
