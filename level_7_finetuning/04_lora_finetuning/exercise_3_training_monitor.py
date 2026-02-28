"""
Exercise 3: Training Monitor
=============================
Build a training monitor: track loss curve, learning rate schedule,
evaluation metrics, save checkpoints. Visualize with text-based charts.

Learning Objectives:
- Track training metrics over time
- Implement learning rate schedulers
- Build text-based visualizations for terminal
- Implement checkpoint saving logic

Run:
    python exercise_3_training_monitor.py
"""

import json
import math
import os
import random
import time
from collections import defaultdict


class TrainingMonitor:
    """Monitor and visualize training metrics in the terminal."""

    def __init__(self, total_steps: int, eval_every: int = 50,
                 checkpoint_every: int = 100, output_dir: str = "./checkpoints"):
        self.total_steps = total_steps
        self.eval_every = eval_every
        self.checkpoint_every = checkpoint_every
        self.output_dir = output_dir

        # Metric storage
        self.train_losses = []
        self.eval_losses = []
        self.eval_accuracies = []
        self.learning_rates = []
        self.step_times = []
        self.checkpoints = []

        # Best metrics
        self.best_eval_loss = float("inf")
        self.best_step = 0

    def log_step(self, step: int, loss: float, lr: float, duration: float):
        """Log metrics for a training step."""
        self.train_losses.append((step, loss))
        self.learning_rates.append((step, lr))
        self.step_times.append(duration)

    def log_eval(self, step: int, eval_loss: float, accuracy: float):
        """Log evaluation metrics."""
        self.eval_losses.append((step, eval_loss))
        self.eval_accuracies.append((step, accuracy))

        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.best_step = step
            return True  # new best
        return False

    def log_checkpoint(self, step: int, path: str):
        """Log a saved checkpoint."""
        self.checkpoints.append({"step": step, "path": path})

    def print_step(self, step: int, loss: float, lr: float, duration: float):
        """Print a training step summary line."""
        progress = step / self.total_steps * 100
        bar_len = 30
        filled = int(bar_len * step / self.total_steps)
        bar = "█" * filled + "░" * (bar_len - filled)

        # Estimate remaining time
        if self.step_times:
            avg_time = sum(self.step_times[-50:]) / len(self.step_times[-50:])
            remaining = avg_time * (self.total_steps - step)
            eta = f"{remaining:.0f}s"
        else:
            eta = "?"

        print(f"  Step {step:5d}/{self.total_steps} [{bar}] {progress:5.1f}%  "
              f"loss={loss:.4f}  lr={lr:.2e}  {duration:.3f}s/step  ETA: {eta}")

    def print_eval(self, step: int, eval_loss: float, accuracy: float, is_best: bool):
        """Print evaluation results."""
        marker = " ★ NEW BEST" if is_best else ""
        print(f"\n  ┌─ Eval at step {step}")
        print(f"  │  eval_loss:  {eval_loss:.4f}")
        print(f"  │  accuracy:   {accuracy:.2%}")
        print(f"  │  best_loss:  {self.best_eval_loss:.4f} (step {self.best_step}){marker}")
        print(f"  └─")

    def plot_loss_curve(self, width: int = 60, height: int = 15):
        """Render a text-based loss curve in the terminal."""

        print(f"\n{'=' * 60}")
        print("Loss Curve")
        print(f"{'=' * 60}\n")

        if not self.train_losses:
            print("  (no data)")
            return

        steps = [s for s, _ in self.train_losses]
        losses = [l for _, l in self.train_losses]

        # Downsample for display
        if len(losses) > width:
            bucket_size = len(losses) // width
            display_losses = []
            display_steps = []
            for i in range(0, len(losses), bucket_size):
                chunk = losses[i:i + bucket_size]
                display_losses.append(sum(chunk) / len(chunk))
                display_steps.append(steps[i])
        else:
            display_losses = losses
            display_steps = steps

        min_loss = min(display_losses) * 0.9
        max_loss = max(display_losses) * 1.1
        loss_range = max_loss - min_loss

        if loss_range == 0:
            loss_range = 1

        # Render
        for row in range(height):
            threshold = max_loss - (row / height) * loss_range
            label = f"  {threshold:6.3f} │"
            line = ""
            for val in display_losses:
                if val >= threshold:
                    line += "█"
                else:
                    line += " "
            print(f"{label}{line}")

        # X-axis
        print(f"  {'':>7s}└{'─' * len(display_losses)}")
        first_step = display_steps[0] if display_steps else 0
        last_step = display_steps[-1] if display_steps else 0
        print(f"  {'':>8s}{first_step:<{len(display_losses) // 2}}{last_step:>{len(display_losses) - len(display_losses) // 2}}")
        print(f"  {'':>8s}{'Steps':^{len(display_losses)}}")

    def plot_lr_schedule(self, width: int = 60, height: int = 10):
        """Render learning rate schedule."""

        print(f"\n{'=' * 60}")
        print("Learning Rate Schedule")
        print(f"{'=' * 60}\n")

        if not self.learning_rates:
            print("  (no data)")
            return

        lrs = [lr for _, lr in self.learning_rates]

        # Downsample
        if len(lrs) > width:
            bucket_size = len(lrs) // width
            display = [sum(lrs[i:i + bucket_size]) / bucket_size
                       for i in range(0, len(lrs), bucket_size)]
        else:
            display = lrs

        max_lr = max(display) * 1.1
        min_lr = min(display) * 0.9
        lr_range = max_lr - min_lr or 1e-8

        for row in range(height):
            threshold = max_lr - (row / height) * lr_range
            label = f"  {threshold:.1e} │"
            line = ""
            for val in display:
                if val >= threshold:
                    line += "▓"
                else:
                    line += " "
            print(f"{label}{line}")

        print(f"  {'':>11s}└{'─' * len(display)}")

    def plot_eval_metrics(self):
        """Show evaluation metrics over time."""

        print(f"\n{'=' * 60}")
        print("Evaluation Metrics")
        print(f"{'=' * 60}\n")

        if not self.eval_losses:
            print("  (no evaluations yet)")
            return

        print(f"  {'Step':>6s}  {'Eval Loss':>10s}  {'Accuracy':>9s}  {'Best?'}")
        print("  " + "-" * 42)

        for i, ((step, loss), (_, acc)) in enumerate(
            zip(self.eval_losses, self.eval_accuracies)
        ):
            is_best = "★" if step == self.best_step else " "
            print(f"  {step:6d}  {loss:10.4f}  {acc:9.2%}  {is_best}")

    def summary(self):
        """Print training summary."""

        print(f"\n{'=' * 60}")
        print("Training Summary")
        print(f"{'=' * 60}")

        if self.train_losses:
            print(f"\n  Steps completed:   {len(self.train_losses)}")
            print(f"  Initial loss:      {self.train_losses[0][1]:.4f}")
            print(f"  Final loss:        {self.train_losses[-1][1]:.4f}")

            # Loss reduction
            reduction = (1 - self.train_losses[-1][1] / self.train_losses[0][1]) * 100
            print(f"  Loss reduction:    {reduction:.1f}%")

        if self.eval_losses:
            print(f"  Best eval loss:    {self.best_eval_loss:.4f} (step {self.best_step})")
            print(f"  Final accuracy:    {self.eval_accuracies[-1][1]:.2%}")

        if self.step_times:
            avg_time = sum(self.step_times) / len(self.step_times)
            total_time = sum(self.step_times)
            print(f"  Avg step time:     {avg_time:.3f}s")
            print(f"  Total time:        {total_time:.1f}s")

        if self.checkpoints:
            print(f"  Checkpoints saved: {len(self.checkpoints)}")

    def save_log(self, filepath: str):
        """Save training log to JSON."""
        log = {
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
            "eval_accuracies": self.eval_accuracies,
            "learning_rates": self.learning_rates,
            "checkpoints": self.checkpoints,
            "best_eval_loss": self.best_eval_loss,
            "best_step": self.best_step,
        }
        with open(filepath, "w") as f:
            json.dump(log, f, indent=2)
        print(f"\n  Training log saved to: {filepath}")


def cosine_lr(step: int, total_steps: int, max_lr: float = 2e-4,
              min_lr: float = 1e-6, warmup_steps: int = 50) -> float:
    """Cosine learning rate with warmup."""
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def simulate_training_run(num_steps: int = 500, eval_every: int = 50,
                          checkpoint_every: int = 100):
    """Simulate a complete training run with monitoring."""

    print("\n--- Simulating Training Run ---")
    print(f"  Steps: {num_steps}, Eval every: {eval_every}, Checkpoint every: {checkpoint_every}\n")

    random.seed(42)

    monitor = TrainingMonitor(
        total_steps=num_steps,
        eval_every=eval_every,
        checkpoint_every=checkpoint_every,
    )

    # Training simulation
    for step in range(num_steps):
        # Simulate training step
        start_time = time.time()

        # Learning rate
        lr = cosine_lr(step, num_steps)

        # Simulated loss (decreasing with noise)
        progress = step / num_steps
        base_loss = 3.0 * (1 - progress) ** 1.5 + 0.35
        noise = random.gauss(0, 0.08 * (1 - progress))
        loss = max(0.1, base_loss + noise)

        # Simulate step duration (faster on GPU, slower on CPU)
        duration = random.uniform(0.05, 0.15)
        time.sleep(0.001)  # tiny sleep for realism

        # Log
        monitor.log_step(step, loss, lr, duration)

        # Print progress at intervals
        if step % 50 == 0 or step == num_steps - 1:
            monitor.print_step(step, loss, lr, duration)

        # Evaluation
        if step > 0 and step % eval_every == 0:
            eval_loss = loss + random.uniform(0.05, 0.15)
            accuracy = max(0, min(1, 1 - eval_loss / 3.5 + random.uniform(-0.02, 0.02)))
            is_best = monitor.log_eval(step, eval_loss, accuracy)
            monitor.print_eval(step, eval_loss, accuracy, is_best)

        # Checkpoint
        if step > 0 and step % checkpoint_every == 0:
            ckpt_path = f"./checkpoint-{step}"
            monitor.log_checkpoint(step, ckpt_path)
            print(f"  💾 Checkpoint saved: {ckpt_path}")

    return monitor


def main():
    """Build and use a training monitor."""

    print("=" * 60)
    print("Exercise 3: Training Monitor")
    print("=" * 60)

    # Run simulation
    monitor = simulate_training_run(
        num_steps=500,
        eval_every=50,
        checkpoint_every=100,
    )

    # Visualizations
    monitor.plot_loss_curve()
    monitor.plot_lr_schedule()
    monitor.plot_eval_metrics()
    monitor.summary()

    # Save log
    log_path = os.path.join(os.path.dirname(__file__) or ".", "training_log.json")
    monitor.save_log(log_path)

    print(f"\n{'=' * 60}")
    print("✓ Training monitor exercise complete!")
    print("  In real training, replace simulated values with actual")
    print("  metrics from Hugging Face Trainer callbacks.")


if __name__ == "__main__":
    main()
