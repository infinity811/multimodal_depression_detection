Automatic Depression Detection Using an Interpretable Audio-Textual Multi-modal Transformer-based Model

This repository contains the code and resources for our multi-modal Transformer-based framework that detects depression from audio and text modalities. Our approach leverages the self-attention mechanism to improve diagnostic accuracy and provide interpretability by identifying the features (tokens in text or acoustic cues in audio) that contribute most strongly to the model’s predictions.

--------------------------------------------------------------------------------
TABLE OF CONTENTS

1. Overview
2. Key Features
3. Model Architecture
4. Installation
5. Usage
6. Datasets
7. Results
8. Interpretability
9. Contributors
10. Citation
--------------------------------------------------------------------------------

1. OVERVIEW
Depression is a common mental disorder characterized by persistent low mood, loss of interest, and fatigue. Automated systems for early detection can encourage timely clinical intervention. In this project, we propose:
- A multi-modal Transformer model that fuses text and audio features.
- An interpretable architecture that provides attention-based insights into what drives classification decisions.

--------------------------------------------------------------------------------

2. KEY FEATURES
- Multi-modal Fusion: Combines audio (mel-spectrogram + NetVLAD) and text (BERT embeddings) signals.
- Transformer-based: Leverages self-attention to capture long-range dependencies in both modalities.
- High Accuracy: Evaluated on benchmark datasets, achieving state-of-the-art or near state-of-the-art performance.
- Explainable: Visualizes attention weights in the text modality to highlight which words or phrases are most influential for the final prediction.

--------------------------------------------------------------------------------

3. MODEL ARCHITECTURE
1) Text Encoding
   - Uses a BERT tokenizer and embedding layer to generate contextual embeddings of the text transcript.
2) Audio Encoding
   - Converts audio signals into mel-spectrograms, then encodes them via NetVLAD to produce 128-dimensional feature vectors.
3) Transformer Encoder
   - Processes each modality (text/audio) through a Transformer encoder, capturing contextual relationships and long-range dependencies.
4) Fusion
   - Concatenates the encoded audio and text embeddings into a joint representation.
5) Classification
   - Passes the fused representation through a feed-forward network to predict depressed or not depressed.

A schematic of the approach might look like this:

Audio ---> NetVLAD --> Transformer Encoder ----\
                                                --> Concatenate --> Feed-Forward --> Output
Text --->  BERT Embeddings --> Transformer Encoder ----/

--------------------------------------------------------------------------------

4. INSTALLATION
1) Clone this repository:
   git clone https://github.com/your-username/Depression-Detection.git
   cd Depression-Detection

2) Create a virtual environment (recommended) and install dependencies:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt

3) Download any required checkpoints (e.g., pre-trained BERT weights) as prompted by the code or from links provided in the repository.

--------------------------------------------------------------------------------

5. USAGE

1) Prepare Data
   - Organize your dataset (audio + text files) in the structure expected by the scripts in data_preprocessing/.
   - Ensure the audio and text transcripts are correctly aligned.

2) Run Training
   python train.py --config configs/config.json
   - Edit hyperparameters (e.g., learning rate, batch size) in configs/config.json.

3) Run Evaluation
   python evaluate.py --config configs/config.json --checkpoint path_to_checkpoint
   - Reports accuracy, F1-score, and other metrics on the test set.

4) Visualize Attention
   python visualize_attention.py --checkpoint path_to_checkpoint --sample_text "Your sample text here"
   - Saves attention heatmaps to the visualizations/ directory.

--------------------------------------------------------------------------------

6. DATASETS
We tested our framework on two datasets:

1) DAIC-WOZ
   - Clinical interviews for diagnosing psychological distress, including depression, anxiety, and PTSD.
   - Contains both audio recordings and transcripts, with depression labels (PHQ-8).

2) EATD Corpus
   - Emotional audio-textual dataset for automatic depression detection (based on SDS).
   - Contains shorter recordings (Chinese) and corresponding transcriptions.

Preprocessing scripts for both datasets can be found in the data_preprocessing/ folder. These scripts handle tasks such as audio segmentation, text tokenization, and alignment of modalities.

--------------------------------------------------------------------------------

7. RESULTS

EATD:
 Model                                  | Modality    | F1 Score
 ---------------------------------------|------------|---------
 Bi-LSTM & GRU + Attention (Shen 2022) | Text+Audio | 0.71
 Proposed Transformer                   | Text+Audio | 0.82

DAIC-WOZ:
 Model                                      | Modality    | Accuracy (%)
 -------------------------------------------|------------|-------------
 Topic-Attentive Transformer (Guo 2022)     | Text+Audio | 73.9
 Proposed Transformer                       | Text+Audio | 75.8

Our multi-modal Transformer outperforms single-modal baselines and other state-of-the-art methods on both DAIC-WOZ and EATD corpora.

--------------------------------------------------------------------------------

8. INTERPRETABILITY
To interpret the model’s decisions, we visualize the self-attention weights from the Transformer’s text encoder.

- Attention Heatmaps: Provide insights into which tokens (e.g., “smart”, “very”, “uh”) the model focuses on.
- Multi-head Attention: Each head may capture different semantics. Some heads may focus on emotion-related words, others on linguistic structure.
- Future Work: Extending attention-based interpretability to audio features (currently encoded as a single vector) would provide more granular insights into acoustic cues.

Example self-attention heatmap (text modality):
Token Index:  1    2    3   4  ...  N
Token:       [I] [feel] [very] ... [PAD]
Head0_Attn   0.2  0.8    0.7   ...  0.0
Head1_Attn   0.0  0.1    0.1   ...  0.8

--------------------------------------------------------------------------------

9. CONTRIBUTORS
- Om Jodhpurkar
  jodhpurk@usc.edu

- Sneh Thorat
  snehpram@usc.edu

- Mehrshad Saadatinia
  saadatin@usc.edu

- Pin-Tzu Lee
  pintzule@usc.edu

- Sreya Reddy Chinthala
  chinthal@usc.edu


--------------------------------------------------------------------------------

10. CITATION
If you find this repository helpful in your research or projects, please cite our work:


--------------------------------------------------------------------------------

NOTE: This project is intended for research purposes only. It is not a substitute for professional mental health diagnosis or treatment. Always seek advice from qualified healthcare professionals.
