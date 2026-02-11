# CSC5991 — Introduction to Large Language Models

**Student:** Anik Tahabilder
**Email:** tahabilderanik@gmail.com

---

## About

These are my personal study notes for the **CSC5991: Introduction to Large Language Models** course. I created interactive Jupyter notebooks as companions to each lecture, converting the slide content into hands-on, runnable code to deepen my understanding of the material.

Each notebook takes the concepts from the lecture slides and turns them into:
- Detailed explanations in my own words with line-by-line code comments
- Markdown tables summarizing key concepts
- ASCII architecture diagrams
- Matplotlib visualizations (heatmaps, plots, diagrams)
- From-scratch implementations using only **NumPy** and **Matplotlib** (Lectures 1-3) or **PyTorch** (Lecture 4)

---

## Repository Structure

```
CSC5991/
├── README.md
├── lectures/                          # Source lecture slides
│   ├── Lecture1.pptx
│   ├── Lecture1.pdf
│   ├── Lecture2.pdf
│   ├── Lecture3.pdf
│   └── Lecture4.pdf
│
├── lecture_1/                         # Lecture 1: Introduction to LLMs
│   ├── build_notebook.py              # Script to generate the notebook
│   └── 01_Introduction_to_LLMs.ipynb  # Companion notebook (51 cells)
│
├── lecture_2/                         # Lecture 2: RNN, LSTM & GRU
│   ├── build_notebook.py              # Script to generate the notebook
│   ├── 01_Supervised_Learning_Basics.ipynb
│   └── 04_RNN_LSTM_GRU.ipynb          # Companion notebook (31 cells)
│
├── lecture_3/                         # Lecture 3: Attention in Machine Learning
│   ├── build_notebook.py              # Script to generate the notebook
│   └── 01_Attention_in_Machine_Learning.ipynb  # Companion notebook (50 cells)
│
├── lecture_4/                         # Lecture 4: Image Captioning with Visual Attention
│   ├── build_notebook.py              # Script to generate the notebook
│   └── 01_Image_Captioning_with_Visual_Attention.ipynb  # Companion notebook (40 cells)
│
└── venv/                              # Python virtual environment (not tracked)
```

---

## Notebooks

### Lecture 1: Introduction to Large Language Models
**Notebook:** [`lecture_1/01_Introduction_to_LLMs.ipynb`](lecture_1/01_Introduction_to_LLMs.ipynb)
**Slides:** 1–30 | **Cells:** 51 (31 markdown, 20 code)
**Dependencies:** NumPy, Matplotlib

| Topic | Description |
|-------|-------------|
| ML vs Deep Learning | Venn diagram of AI/ML/DL/LLM relationships |
| History of Deep Learning | Timeline from 1958 to present, three eras |
| Transfer Learning | Pre-training and fine-tuning pipeline |
| Key Models | BERT, GPT, CLIP, LLaMA, PaLM, Gemini |
| Transformer Architecture | Encoder/decoder blocks, self-attention mechanism |
| BERT vs GPT | Side-by-side architecture comparison |
| Self-Supervised Learning | Masked word prediction demo |
| Prompt Engineering | Bad vs good prompts, in-context learning |
| Few-Shot Classification | Similarity-based classification from scratch |
| Bias & Safety | Embedding bias detection and visualization |
| Future Trends | Model size growth, open research questions |

**Code Demos:** Neural network forward pass, tokenization, word embeddings, attention scores, next-word prediction, self-supervised learning, few-shot classification, bias detection — all from scratch with NumPy.

---

### Lecture 2: Recurrent Neural Networks (RNN), LSTM & GRU
**Notebook:** [`lecture_2/04_RNN_LSTM_GRU.ipynb`](lecture_2/04_RNN_LSTM_GRU.ipynb)
**Slides:** 13–19 | **Cells:** 31 (14 markdown, 17 code)
**Dependencies:** NumPy, Matplotlib

| Topic | Description |
|-------|-------------|
| Sequential Tasks | Why order matters in language processing |
| History of Language Models | From n-grams to Transformers |
| Vanilla RNN | Complete from-scratch implementation |
| BPTT | Backpropagation Through Time with gradient tracking |
| Vanishing/Exploding Gradients | Empirical demonstration and visualization |
| LSTM | From-scratch implementation with gate analysis |
| GRU | From-scratch implementation |
| Character-Level Language Model | Training a text generator on "hello world" |
| Gradient Flow Comparison | RNN vs LSTM over 50 time steps |

**Code Demos:** Vanilla RNN, LSTM, GRU — all implemented from scratch with NumPy. Includes a working character-level language model with text generation.

---

### Lecture 3: Attention in Machine Learning
**Notebook:** [`lecture_3/01_Attention_in_Machine_Learning.ipynb`](lecture_3/01_Attention_in_Machine_Learning.ipynb)
**Slides:** 1–59 | **Cells:** 50 (24 markdown, 26 code)
**Dependencies:** NumPy, Matplotlib

| Topic | Description |
|-------|-------------|
| What is Attention? | Context in RNNs vs CNNs vs Attention |
| Seq2Seq with Attention | Machine translation pipeline |
| Bahdanau Attention | Additive attention from scratch |
| CNN Architecture | Convolution, pooling, feature maps from scratch |
| Spatial Attention | Channel and spatial attention in CNNs |
| Squeeze-and-Excitation | Channel recalibration mechanism |
| CBAM | Convolutional Block Attention Module |

**Code Demos:** Attention score computation, Seq2Seq encoder-decoder, CNN forward pass, spatial/channel attention, SE blocks, CBAM — all from scratch with NumPy.

---

### Lecture 4: Image Captioning with Visual Attention
**Notebook:** [`lecture_4/01_Image_Captioning_with_Visual_Attention.ipynb`](lecture_4/01_Image_Captioning_with_Visual_Attention.ipynb)
**Slides:** 1–8 | **Cells:** 40 (21 markdown, 19 code)
**Dependencies:** PyTorch, TorchVision, Pillow, NumPy, Matplotlib

| Topic | Description |
|-------|-------------|
| Attention Recap | Scaled dot-product and multi-head attention |
| CNN Encoder | ResNet-based feature extraction |
| Bahdanau Attention | Additive attention for image captioning |
| Decoder with Attention | LSTM decoder that attends to image regions |
| Attention Visualization | Heatmaps showing where the model "looks" |

**Code Demos:** Full image captioning pipeline with CNN encoder, Bahdanau attention, LSTM decoder, and attention visualization — implemented with PyTorch.

---

## Setup

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/CSC5991.git
cd CSC5991

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Register the kernel for Jupyter
python -m ipykernel install --user --name=venv --display-name="Python (CSC5991)"
```

### Running the Notebooks

```bash
# Option 1: Open in Jupyter Notebook
jupyter notebook

# Option 2: Open in VS Code
# Just open any .ipynb file and select the "Python (CSC5991)" kernel
```

### Regenerating Notebooks from Build Scripts

Each notebook is generated by a `build_notebook.py` script. To regenerate:

```bash
cd lecture_1 && python build_notebook.py
cd lecture_2 && python build_notebook.py
cd lecture_3 && python build_notebook.py
cd lecture_4 && python build_notebook.py
```

---

## Summary Table

| Lecture | Topic | Cells | Tables | Diagrams | Plots | Dependencies |
|---------|-------|-------|--------|----------|-------|-------------|
| 1 | Introduction to LLMs | 51 | 42 | 15 | 12 | NumPy, Matplotlib |
| 2 | RNN, LSTM & GRU | 31 | 17 | 4 | 5 | NumPy, Matplotlib |
| 3 | Attention in ML | 50 | 13 | 13 | 12 | NumPy, Matplotlib |
| 4 | Image Captioning | 40 | 14 | 4 | 6 | PyTorch, NumPy, Matplotlib |

---

## Disclaimer

These notebooks are my personal study notes and not official course materials. The lecture slides belong to the course instructor. The notebook implementations and explanations are my own work created for learning purposes.
