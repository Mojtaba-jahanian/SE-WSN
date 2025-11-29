# SE-WSN: A Self-Evolving Wireless Sensor Network Architecture
### Integrating Temporal Graph Neural Networks and Multi-Agent Deep Reinforcement Learning

This repository provides a clean, executable, and minimal implementation of **SE-WSN**,  
the model proposed in the paper:

> **A Self-Evolving Wireless Sensor Network Architecture Integrating Temporal Graph Neural Networks and Multi-Agent Deep Reinforcement Learning**  
> Mojtaba Jahanian, Mehdi HosseinZadeh, Hossein Yarahmadi, Maryam Hajiei  
> (2025)

---

## 📌 Overview

SE-WSN is a **self-evolving WSN architecture** that integrates:

- Temporal Graph Representation Learning  
- Temporal Graph Neural Networks (T-GNN)  
- Multi-Agent Deep Reinforcement Learning (MADDPG)  
- Castalia-like WSN simulation environment  

This repository contains a **minimal reproducible template** of the architecture.

---

## 🚀 How to Run

### 1. Create virtual environment and install dependencies

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Place dataset

```
data/raw/Xdados.txt
```

### 3. Build temporal graphs

```bash
cd data/scripts
python preprocess_xdados.py
```

### 4. Train MADDPG controller

```bash
cd ../../src/training
python train.py
```

Trained models will be saved in:

```
results/trained_models/
```

---

## 🗂 Project Structure

```
SE-WSN/
  data/
    raw/
    processed/graphs/
    scripts/
  src/
    models/
    maddpg/
    simulator/
    training/
  results/
    trained_models/
    logs/
    figures/
  README.md
```

---

## 📝 Citation

```bibtex
@article{SEWSN2025,
  title={A Self-Evolving Wireless Sensor Network Architecture Integrating Temporal Graph Neural Networks and Multi-Agent Deep Reinforcement Learning},
  author={Jahanian, Mojtaba and HosseinZadeh, Mehdi and Yarahmadi, Hossein and Hajiei, Maryam},
  year={2025},
}
```

---

## 👤 Authors

- Mojtaba Jahanian  
- Mehdi HosseinZadeh  
- Hossein Yarahmadi  
- Maryam Hajiei  
