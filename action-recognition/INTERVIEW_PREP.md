# 🎤 Interview Preparation - Gesture/Expression Recognition Thesis

Comprehensive Q&A guide for the AASIS project Master's thesis interview at Aalto/UH.

---

## 📋 Table of Contents
1. [Project Overview](#1-project-overview)
2. [Model Architecture](#2-model-architecture)
3. [Data Processing](#3-data-processing)
4. [Training Techniques](#4-training-techniques)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [Transfer Learning](#6-transfer-learning)
7. [Advanced Topics](#7-advanced-topics)
8. [Thesis-Related Questions](#8-thesis-related-questions)
9. [Questions to Ask](#9-questions-to-ask)

---

## 1. Project Overview

### Q: Can you introduce your project?

**A:** "I developed a video action recognition system based on **R3D-18** (3D ResNet-18). On the **UCF101 dataset** (101 action classes, 13,000+ videos), my model achieved **87.84% Top-1** and **97.03% Top-5** validation accuracy.

Key technical highlights:
- **3D convolutions** to capture spatio-temporal features
- **Kinetics-400 pre-trained weights** for transfer learning
- **Mixed precision training** (AMP) for 4-hour training (30 epochs)
- Data augmentation: uniform frame sampling, random flip, brightness jitter"

### Q: What is the project structure?

**A:**
```
action-recognition/
├── src/
│   ├── dataset.py    # Data loading & preprocessing
│   ├── model.py      # R3D-18 model definition
│   ├── train.py      # Training loop with AMP
│   └── evaluate.py   # Evaluation & visualization
├── configs/config.yaml
└── checkpoints/best_model.pth
```

---

## 2. Model Architecture

### Q: Why did you choose R3D-18?

**A:** "R3D-18 is a lightweight version of 3D ResNet with only 33M parameters. Compared to deeper networks like R3D-50:
- Easier to train on medium-scale datasets like UCF101
- Less prone to overfitting
- Good balance between performance and computation"

### Q: What is the difference between 3D CNN and 2D CNN + LSTM?

**A:**
| Method | How it works | Pros | Cons |
|--------|--------------|------|------|
| **3D CNN** | 3D kernels slide over time+space | End-to-end learning of spatio-temporal features | Higher computation |
| 2D + LSTM | 2D CNN per frame → LSTM for temporal | Lower computation | Decoupled spatial/temporal learning |

"3D CNN kernels slide simultaneously in time and space dimensions, directly learning **joint spatio-temporal features** like 'hand moving left to right'. 2D+LSTM learns spatial features first, then models temporal relationships separately."

### Q: What is pooling and what is FC layer?

**A:** 
- **Pooling**: Downsampling operation that reduces feature map size. Global Average Pooling compresses spatial dimensions to a single point, retaining channel-wise statistics.
- **FC (Fully Connected)**: Every input connects to every output. The final FC layer maps 512-dim features to 101 class scores.

```
Feature map (512, 1, 4, 4)
       ↓ Global Avg Pool
Feature vector (512,)
       ↓ FC Layer
Class scores (101,)
       ↓ Softmax
Probabilities (101,)
```

---

## 3. Data Processing

### Q: How do you handle videos of different lengths?

**A:** "I use **uniform sampling** strategy. Regardless of video length, I use `np.linspace` to uniformly select 16 frames:
- 100-frame video → sample every ~6 frames
- 30-frame video → sample all + repeat some

This ensures coverage of the entire video timeline without missing key actions."

### Q: What data augmentation did you use?

**A:**
| Augmentation | Probability | Purpose |
|--------------|-------------|---------|
| Horizontal Flip | 50% | Direction invariance |
| Random Crop (90%) | 50% | Position invariance |
| Brightness Jitter (±20%) | 50% | Lighting invariance |
| Normalization | 100% | Match pretrained distribution |

"These augmentations increase training data diversity, prevent overfitting, and improve generalization."

### Q: Why use Kinetics normalization values?

**A:** "Our model uses Kinetics-400 pretrained weights. Using the same normalization (mean/std) ensures the input data distribution matches what the pretrained model expects."

---

## 4. Training Techniques

### Q: What is Mixed Precision Training (AMP)? Why use GradScaler?

**A:** "AMP uses FP16 for forward pass (faster, less memory) and FP32 for gradients (accuracy).

**Problem**: FP16 has limited range. Gradients smaller than 6×10⁻⁵ become zero (underflow).

**Solution - GradScaler**:
1. Scale up loss by 1024×
2. Gradients are also scaled up → no underflow
3. Before weight update, scale back down

Analogy: Like using a microscope to see tiny ants, then recording at normal scale."

### Q: Why use Cosine Annealing instead of Step decay?

**A:**
- **Step decay**: Sudden LR drops at fixed epochs → can cause training instability
- **Cosine Annealing**: Smooth LR decay following cosine curve → stable convergence

"Cosine provides **smooth learning rate decay**. Early training uses high LR for fast progress; late training uses low LR for fine-tuning."

### Q: What is learning rate warmup?

**A:** "During the first 3 epochs, we gradually increase LR from 0 to the target value. This prevents gradient explosion when the model weights are not yet stable."

---

## 5. Evaluation Metrics

### Q: What is the difference between Top-1 and Top-5 accuracy?

**A:**
| Metric | Definition | Our Result |
|--------|------------|------------|
| Top-1 | Highest prediction is correct | 87.84% |
| Top-5 | Correct class in top 5 predictions | 97.03% |

**Example**: For a basketball dunk video:
```
Predictions:
1. Basketball: 40%
2. BasketballDunk: 35%  ← Correct answer
...
```
- Top-1: ❌ Wrong (predicted Basketball)
- Top-5: ✅ Correct (correct answer in top 5)

"Top-5 is higher because similar classes (Basketball vs BasketballDunk) may confuse the model, but it still ranks the correct answer highly."

### Q: What other evaluation methods did you use?

**A:**
- **Confusion Matrix**: Visualize which classes are often confused
- **Per-class Accuracy**: Identify well/poorly performing classes
- **Inference Speed**: Measure FPS for real-time feasibility

---

## 6. Transfer Learning

### Q: What is transfer learning?

**A:** "Transfer learning = using knowledge learned from Task A to improve performance on Task B.

In our project:
- **Task A**: Kinetics-400 action recognition (400K videos, 400 classes)
- **Task B**: UCF101 action recognition (13K videos, 101 classes)

The pretrained model already learned universal features (motion, poses). We only need to **fine-tune** for the new 101 classes."

### Q: Did you update the pretrained weights?

**A:** "Yes! We used **fine-tuning**, where all 33M parameters are updated during training. This is different from **feature extraction** where only the final layer is trained.

Fine-tuning gives better accuracy because:
- Features adapt to the target dataset
- But requires smaller learning rate to avoid destroying pretrained knowledge"

### Q: Why not train from scratch?

**A:**
- UCF101 has only 13K videos → easy to overfit
- Kinetics has 400K videos → pretrained model learned robust features
- Transfer learning: faster convergence + higher accuracy

---

## 7. Advanced Topics

### Q: How to handle multi-person scenarios?

**A:** "Two-stage pipeline:

**Stage 1: Person Detection & Tracking**
```
Video → Object Detector (YOLOv8) → Bounding boxes
     → Multi-object Tracker (DeepSORT) → Assign IDs
```

**Stage 2: Per-person Action Recognition**
```
Crop each person → Action Recognition Model → Action label
```

**Challenges**:
- Occlusion: Use tracking to maintain identity
- Person-person interaction: Use Graph Neural Networks to model relationships"

### Q: How to optimize for real-time inference?

**A:** Three main techniques:

| Method | Speedup | Accuracy Loss | Difficulty |
|--------|---------|---------------|------------|
| **Quantization** | 2-4x | ~1% | Easy |
| **Pruning** | 1.5-2x | 1-2% | Medium |
| **Knowledge Distillation** | 2-5x | 2-5% | Hard |

"**Quantization** converts FP32 to INT8 → 4x smaller, 2-4x faster.
**Pruning** removes unimportant neurons/channels.
**Distillation** trains a small model to mimic a large model's outputs."

### Q: How to ensure cross-domain generalization?

**A:**
1. **Data Augmentation**: Simulate various conditions (lighting, angles, noise)
2. **Domain Adaptation**: Align feature distributions between source and target domains
3. **Self-supervised Pretraining**: Learn from large unlabeled video data (VideoMAE)

"Start with strong augmentation; if insufficient, consider domain adaptation or self-supervised methods."

### Q: What are the latest advances in video understanding?

**A:**
- **Video Transformers**: TimeSformer, Video Swin Transformer
- **Self-supervised Learning**: VideoMAE (masked autoencoding)
- **Multi-modal**: Video + Audio + Text understanding
- **Video LLMs**: Combining video understanding with large language models

---

## 8. Thesis-Related Questions

### Q: How does your project relate to gesture/expression recognition?

**A:**
| Task | Granularity | Difference |
|------|-------------|------------|
| Action Recognition | Full body | Large-scale motion |
| Gesture Recognition | Hands | Need ROI cropping |
| Expression Recognition | Face | Need face detection |

"The core technology is the same - video understanding with 3D CNNs. For gestures, I would add **hand detection** preprocessing. For expressions, I would add **face detection**."

### Q: How would you process ELAN annotation data?

**A:** "ELAN produces timestamped annotations (start_time, end_time, label). I would:
1. Segment videos based on timestamps
2. Each segment becomes a training sample
3. Use the same frame sampling and training pipeline from my project"

### Q: What challenges do you expect in the thesis project?

**A:**
- **Fine-grained recognition**: Gestures are subtle compared to whole-body actions
- **Multi-person interaction**: Pair discussions involve two people
- **Multi-modal fusion**: Combining video and audio information
- **Limited data**: 100 students × 4 tasks may require careful data augmentation

---

## 9. Questions to Ask

**Suggested questions for the interviewer:**

1. "What is the scale and annotation granularity of the dataset?"
2. "Is there a preferred technology stack, or is there flexibility to explore?"
3. "Is the thesis focused on method innovation or application validation?"
4. "How is the collaboration structured between UH and Aalto?"
5. "What is the current state of the annotation quality?"
6. "Are there opportunities to work with multi-modal (audio+video) analysis?"

---

## 🏆 Key Takeaways

**Your Three Strengths:**
1. ✅ **Complete Project Experience** - From data processing to model training to evaluation
2. ✅ **87.84% Accuracy** - Proof that you can build effective systems
3. ✅ **HPC Experience** - Familiar with Triton cluster and SLURM

**Communication Tips:**
- Be **concise and accurate** on technical questions
- Show **deep understanding**, not just implementation
- Express **genuine interest** in the research topic

---

**Good luck with your interview! 🚀**
