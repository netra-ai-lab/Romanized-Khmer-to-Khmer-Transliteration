<div align="center">
  <img src="assets/netra-logo-transparent-new.png" width="60%" alt="Netra Lab" />
</div>
<hr>

<p align="center">
    <a href="https://github.com/Chhunneng/khmer-text-transliteration"><b>Dataset</b></a> |
    <a href="khmer_transliterator/weights/"><b>Model Weights</b></a>
</p>


<h2>
<p align="center">
  <a href="">AkaraAlpha: An Efficient Romanized Khmer-Khmer Script Transliteration Model</a>
</p>
</h2>

<p align="center">
<img src="assets/benchmark.png" style="width: 1000px" align=center>
</p>

<p align="center">
<a href="">Performance benchmark and system optimization. (a) Comparison of standalone model architectures based on Character Error Rate (CER). (b) Trade-off analysis between Top-1 accuracy and inference latency across varying beam widths (k). The red dotted line denotes the 100ms real-time latency threshold, identifying k=5 as the optimal configuration for practical deployment.</a>       
</p>



## Overview
An English–Khmer transliteration system built on an Attention-Based Bidirectional GRU architecture. The model automatically converts romanized Khmer text written in the Latin alphabet into its corresponding Khmer script form. To enhance accuracy and ensure linguistic validity, the system incorporates a Khmer dictionary-based post-processing step for proof checking and correction.

## Dataset  
The dataset used in this project was sourced from the [Khmer Text Transliteration Dataset by Chhunneng (2023)](https://github.com/Chhunneng/khmer-text-transliteration), which provides parallel pairs of English–Khmer transliterations for machine learning research.  

There are 77 unique Khmer characters and 26 unique English characters in the dataset. The maximum sequence length for English (romanized Khmer) inputs is 25 characters, while the maximum Khmer output length is 24 characters.


<figure>
    <img src="./assets/word_length.png" alt="Word Length Distribution">
    <figcaption>
        <strong>Figure 1 | Word Length Distribution of Romanized Khmer, and Khmer Script</strong>
    </figcaption>
</figure>


- **Total Samples:** 28,569  
- **Train Set:** 22,855 (80%)  
- **Validation Set:** 5,714 (20%)  
- **Format:** Parallel text pairs (`brodae: ប្រដែ`)

## Model Architecture

The model is based on an Attention-Based Bidirectional GRU architecture designed for sequence-to-sequence transliteration. It follows an encoder–decoder structure, where the encoder processes the input Latin Script (English) sequence, and the decoder generates the corresponding Khmer script sequence character by character.

<figure>
 <img src="./assets/model_architecture.png" alt="Model Architecture">
<figcaption>
    <strong>Figure 2 | The proposed Attention-based Bidirectional Gated Recurrent Unit architecture.</strong>
  </figcaption>
</figure>

### Encoder  

The encoder uses a Bidirectional GRU layer to process text from both start and end directions within the input sequence. This allows the model to better understand dependencies across the entire input text, which is particularly useful for transliteration tasks where phonetic relationships depend on both preceding and succeeding characters.  

Each input token is first mapped into a continuous vector space through an embedding layer, which converts discrete character indices into dense embeddings of dimension 32. The bidirectional GRU then encodes these embeddings into a hidden state representation that encapsulates forward and backward context information.  

### Decoder  

The decoder consists of a GRU layer that processes the output sequence one Khmer character at a time. At each decoding step, it receives the previously predicted token and the projected hidden state from the encoder. The attention mechanism then combines the encoder’s output representations with the decoder’s current hidden state to generate a context vector, which helps the model focus on the most relevant parts of the input sequence. This context vector is concatenated with the decoder’s GRU output and passed through a dense softmax layer to produce the final character prediction.

- **Embedding Dimension:** 32  
- **GRU Units:** 64  
- **Attention Mechanism:** Additive Attention  

## Training Configuration
- **Batch Size:** 64  
- **Epochs:** 50  
- **Validation Split:** 20%  
- **Optimizer:** Adam  
- **Loss Function:** Sparse Categorical Crossentropy  
- **Learning Rate Scheduler:** ReduceLROnPlateau (factor=0.5, patience=3)

## Post-Processing Technique
To maximize accuracy without adding model bloat, our post-processing pipeline ensures all outputs are valid Khmer words. The model first generates three candidate transliterations using Beam Search (k=3). These are cross-referenced with a standard dictionary; if no exact match exists, a Levenshtein distance fallback corrects the prediction to the closest valid word (up to a 2-character edit limit).

<figure>
 <img src="./assets/full_pipeline.png" alt="Post-processing pipeline">
    <figcaption>
        <strong>Figure 3 | End-to-End inference pipeline with post-processing technique.</strong> The input flow through the encoder-decoder model to generate 3 prediction using beam search (k=3). Following the beam search output, a lexical validation layer cross-referencing the outputs with Khmer Dictionary. The pipeline uses Levenshtein distance to recover the closest orthographically valid entries, resulting in the finals words being all valid.
    </figcaption>
</figure>

## Results and Analysis

### 1. Evaluation Metrics - CER
To evaluate the performance of the model, Character Error Rate (CER) was used. CER quantifies the number of errors at the character level, providing a direct measure of the model's accuracy in converting individual Romanized graphemes to their corresponding Khmer script. It is calculated as:

$$CER = \frac{S + D + I}{N}$$

where $S$ represents the number of substitutions, $D$ is the number of deletions, $I$ is the number of insertions, and $N$ is the total number of characters in the ground truth sequence. A lower CER indicates higher accuracy in the transliteration output.

### 2. Evaluation Metrics – Top-1 and Top-k Hit Accuracy  
Additionally, to evaluate the post-processing technique, we also assess word-level accuracy by measuring the Top-1 and Top-k hit accuracy. We define Top-1 accuracy as the percentages of test samples where the highest-ranked candidates produced by the system exactly matches the ground truth:

$$Top - 1 = \frac{1}{M} \sum_{i=1}^{M} (\hat{y}_{i,1} = y_i)$$

where $M$ is the total number of test samples, $y_i$ is the ground truth, and $\hat{y}_{i,1}$ is the number 1 ranked candidate produced by the system.

Top-k Hit accuracy measures the frequency with which the correct Khmer word appears within the list of $k$ candidates generated by the system:

$$Top - k = \frac{1}{M} \sum_{i=1}^{M} (y_i \in \{\hat{y}_{i,1}, \hat{y}_{i,2}, \hat{y}_{i,3}, \dots, \hat{y}_{i,k}\})$$

### 3. Quantitative Analysis

<figure>
 <img src="./assets/top_k.png" alt="Trade-off analysis chart">
    <figcaption>
        <strong>Figure 4 | Analytical trade-off between transliteration accuracy and computational overhead.</strong> (a) Search space benefit analysis illustrating the gap between primary prediction (Top-1) and system recall (Top-K). The shaded region represents the additive benefit of beam search; at the recommended k=5 setting, the system provides a 9.81 percentage-point (pp) recall gain over the top-1 prediction. (b) The computational efficiency frontier, highlighting the relationship between accuracy gains and inference latency. The gray bars represent the absolute magnitude of computational overhead, while the red dotted line tracks the linear latency trajectory across different beam widths. The optimal operating point is k=5, which achieves a 29.16% absolute accuracy increase over greedy decoding while remaining under the 100ms real-time latency threshold; beyond this point, the system exhibits diminishing returns as latency exceeds the limits of user interaction.
    </figcaption>
</figure>  

<br>
</br>

Table 1 | Model comparison on the validation set using Character Error Rates (CER %). Lower CER indicates better performance. The Lowest CER is **bold** and the second lowest is *italicize*.

| Model | CER (%) | Parameters |
| :--- | :--- | :--- |
| RNN | 102.41 | 42,609 |
| LSTM | 31.78 | 58,417 |
| GRU | 51.64 | 46,385 |
| Transformer | *17.78* | 248,337 |
| Attention BiLSTM | 18.26 | 96,753 |
| **AkaraAlpha** | **15.07** | 78,705 |

Table 2 | Performance comparison of system configuration. A comparative analysis of Top-1 accuracy, Top-K hit rate, and average inference latency across greedy decoding and varying beam search widths (*k*)

| Configuration | Top-1 | Top-K Hit | Latency (ms/word) |
| :--- | :--- | :--- | :--- |
| Greedy Decoding | 44.44% | 44.44% | 21.92 |
| K=3 | 70.06% | 78.72% | 66.03 |
| K=5 | 73.60% | 83.41% | 94.15 |
| K=7 | 74.87% | 85.60% | 122.10 |
| K=10 | 75.85% | 87.37% | 167.21 |

### 4. Qualitative Analysis
Table 3 | Qualitative comparison across all model showing model performance across various instances of Romanized Khmer Script ranging from short to long words.
![Qualitative Analysis](./assets/qualitative_analysis.png)

## Installation

**From PyPI (recommended):**
```bash
pip install netra-transliterate
```

**From source:**
```bash
git clone https://github.com/NDarayut/english-khmer-transliteration.git
cd english-khmer-transliteration
pip install -e .
```

## Usage

### Python API

#### 1. Single best transliteration
```python
from khmer_transliterator import transliterate

print(transliterate("brodae"))
# 'ប្រដែ'
```

#### 2. Top-N candidates (raw model output)
```python
from khmer_transliterator import transliterate_top_n

print(transliterate_top_n("brodae", n=3))
# ['ប្រដែ', 'បរដែ', 'ប្រតែ']
```

#### 3. Top-N candidates with dictionary validation
```python
from khmer_transliterator import transliterate_with_dict

print(transliterate_with_dict("brodae", n=3))
# ['ប្រដែ', 'រដែ', 'ប្រែ']
```

The `Transliterator` class is also available for explicit instantiation:
```python
from khmer_transliterator import Transliterator

t = Transliterator()
print(t.transliterate("brodae"))
```

> **Note:** The Keras model loads lazily on the first call (~1–2 s one-time cost).

### Command Line

```
usage: khmer-transliterator [-h] [-n N] [--no-dict] [--shell] [--serve] [--port PORT] [WORD ...]
```

**Transliterate words directly:**
```bash
khmer-transliterator brodae
# brodae    ប្រដែ

khmer-transliterator brodae sokha -n 3
# brodae:
#   1. ប្រដែ
#   2. រដែ
#   3. ប្រែ
# sokha:
#   1. សុខា
#   ...
```

**Interactive shell:**
```bash
khmer-transliterator --shell
# or just:
khmer-transliterator
```

**Skip dictionary post-processing:**
```bash
khmer-transliterator brodae --no-dict
```

**Web server:**
```bash
khmer-transliterator --serve
# Starting web server at http://localhost:5000

khmer-transliterator --serve --port 8080
```

## Web Application

A browser-based UI is bundled with the package. Start it with:

```bash
khmer-transliterator --serve
```

Then open http://localhost:5000 in your browser. Type a romanized Khmer word to see live suggestions; use **Tab** to cycle through candidates and **Space** to accept one.

## Demo
![Web Application](./assets/video.gif)

## Citation
> Chhunneng. (2023). *Khmer Text Transliteration Dataset*. GitHub repository.  
> Available at: [https://github.com/Chhunneng/khmer-text-transliteration](https://github.com/Chhunneng/khmer-text-transliteration)
