#!/usr/bin/env python3
"""Entrypoint for the isolated PostgreSQL-backed training worker."""

import signal

from training_worker_service import TrainingWorkerService


service = TrainingWorkerService()


def signal_handler(signum, _frame):
    print(f"收到信号 {signum}，训练 Worker 正在退出...")
    service.stop()


def main():
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    print("启动独立训练 Worker...")
    service.run()
    print("训练 Worker 已退出")


if __name__ == "__main__":
    main()
