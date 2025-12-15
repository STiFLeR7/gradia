<div align="center">

# G R A D I A

**Next-Generation Local-First MLOps Platform**

[![PyPI Version](https://img.shields.io/pypi/v/gradia?style=for-the-badge&color=blue)](https://pypi.org/project/gradia/)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/STiFLeR7/gradia/ci-update.yaml?style=for-the-badge)](https://github.com/STiFLeR7/gradia/actions)

<p align="center">
  <img src="https://github.com/STiFLeR7/gradia/blob/master/docs/dashboard.png" alt="Gradia Dashboard" width="100%" />
</p>

</div>

---

## 🚀 Overview

**Gradia** is a high-performance, asynchronous monitoring solution designed for local machine learning workflows. Unlike cloud-native behemoths, Gradia focuses on **zero-latency**, **privacy-first** tracking that runs directly alongside your training loop.

Built on **FastAPI**, **WebSockets** (simulated via high-frequency polling), and **Reactive UI**, Gradia provides granular visibility into your model's training dynamics, system resource consumption, and feature importance without the overhead of external servers.

## ⚡ Key Capabilities

| Feature | Description |
| :--- | :--- |
| **Real-Time Telemetry** | Nanosecond-precision tracking of Loss, Accuracy, and custom metrics via async event dispatching. |
| **Intelligent Auto-Discovery** | Heuristic analysis of tabular datasets to automatically infer task types (Regression vs Classification) and suggest optimal architectures (CNNs, RFCs). |
| **System Profiling** | Integrated `psutil` hooks for monitoring CPU/GPU* and RAM saturation during training epochs. |
| **Artifact Management** | Automated checkpointing (`best-ckpt`) and structured logging (`events.jsonl`) with thread-safe IO. |
| **Comprehensive Reporting** | One-click generation of audit-ready PDF/JSON reports containing full training history and confusion matrices. |
| **Interactive Evaluation** | Post-training validation suite featuring dynamic Heatmap visualization for Confusion Matrices. |

## 🛠️ Architecture

Gradia employs a **Producer-Consumer** architecture:

1.  **Trainer Thread (Producer)**: Executes the Scikit-Learn training loop, emitting atomic events to a thread-locked `EventLogger`.
2.  **System Thread**: Asynchronously samples hardware metrics.
3.  **Visualization Server (Consumer)**: A lightweight FastAPI instance that aggregates logs and serves a reactive Single Page Application (SPA).

This decoupling ensures that monitoring never bottlenecks your training throughput.

## 📦 Installation

```bash
pip install gradia --upgrade
```

## 💻 Usage

### Quick Start
Initialize the environment and start the observer in one command. Gradia will auto-detect any CSV files in the directory.

```bash
gradia run .
```

### Advanced CLI
Override default heuristics and bind to specific interfaces.

```bash
gradia run . \
  --target "churn_label" \
  --port 8080 \
  --workers 4
```

## 📊 Dashboard

Access the dashboard at `http://localhost:8000`.

-   **Configure**: Select your model (Random Forest, MLP, etc.) and hyperparameters.
-   **Observe**: Watch metrics stream in real-time.
-   **Analyze**: Use the built-in Feature Importance charts to debug model bias.

## 🤝 Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for details on submitting logical PRs.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">
  <sub>Built with ❤️ by STiFLeR for the ML Community.</sub>
</div>
