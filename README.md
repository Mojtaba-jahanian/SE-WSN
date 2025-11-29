SE-WSN — Minimal Executable Template
====================================

This is a clean, minimal, and executable template for your SE-WSN project
(temporal graphs + MADDPG + WSN environment), ready for GitHub.

Main steps to run:

1. Create venv and install requirements:

   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt

2. Put your Intel dataset file as:

   data/raw/Xdados.txt

3. Build temporal graphs:

   cd data/scripts
   python preprocess_xdados.py

4. Run training:

   cd ../../src/training
   python train.py

Trained models will be saved in results/trained_models/.
