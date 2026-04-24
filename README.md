# ALFRED: Emotion-Aware AI Assistant
### 🚀 Featuring YOLO26n & Llama 3

ALFRED (Autonomous Logical Facial Recognition & Emotional Delegate) is a real-time multimodal AI assistant. By leveraging **YOLO26n**, the state-of-the-art in edge-optimized vision, ALFRED perceives user sentiment and dynamically adapts its persona, speech cadence, and response empathy.

## 🌟 Why YOLO26n?
In this project, I transitioned from older architectures to **YOLO26n** (released January 2026) to achieve:
* **Zero-Post-Processing (NMS-Free):** Native end-to-end inference eliminates the Non-Maximum Suppression bottleneck, reducing total system latency.
* **43% Faster CPU Performance:** Specifically optimized for real-time interaction on devices without dedicated GPUs, leaving more headroom for the Llama 3 LLM.
* **STAL Integration:** Small-Target-Aware Labeling ensures high-accuracy detection of facial micro-expressions even from a distance.

## 🛠️ Technical Architecture

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Vision Core** | **Ultralytics YOLO26n** | Real-time sentiment & expression analysis |
| **Language Brain**| **Llama 3 (Ollama)** | Contextual reasoning and dialogue generation |
| **Vocal Core** | **Pyttsx3 (SAPI5)** | Emotion-congruent speech synthesis |
| **Logic Layer** | **LangChain** | Multi-threaded orchestration & memory |

## 📁 Project Structure
```text
/ALFRED-Project
│
├── final.py           # Main logic (Vision + Voice + LLM Integration)
├── main.py            # Dedicated YOLO26n Emotion Detector Module
├── last.pt            # Custom-trained YOLO26n weights for emotions
├── requirements.txt   # Project dependencies
└── README.md          # Documentation
```

## 🚀 Getting Started

1.  **Clone & Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Pull the LLM:**
    Make sure [Ollama](https://ollama.com/) is running and pull the model:
    ```bash
    ollama pull llama3
    ```
3.  **Run the Assistant:**
    Launch the full multimodal experience:
    ```bash
    python final.py
    ```
