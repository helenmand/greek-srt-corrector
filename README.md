# 📝 Greek SRT Corrector
## ✨ Automate Greek Subtitle Correction with AI!

A Streamlit-based web application for fast and accurate Greek subtitle correction using a powerful, locally hosted LLM from Ollama.

<div style="display: flex; justify-content: space-between;">
  <img src="media/srt_demo.gif" alt="app-demo" style="width: 100%;">
</div>

---

### 🚀 Key Features

* **Greek-focused Text Processing**
    -   ✔️ **Punctuation Removal:** Automatically cleans up subtitle text.
    -   ✔️ **Smart Lowercasing:** Corrects the first letter of each subtitle line.
    -   ✔️ **Greek Stop Word Handling:** Skips common words for efficiency.

* **Local AI Integration**
    -   🧠 **Offline Spell Checking:** Integrates seamlessly with a local **Ollama** server, ensuring your data remains private.
    -   🧠 **Context-Aware Corrections:** Uses a Greek-language model (`ilsp/meltemi-instruct-v1.5`) for better corrections. 

* **Comprehensive SRT Workflow**
    -   ▶️ **Effortless Parsing:** Automatically reads and structures `.srt` files.
    -   ✏️ **Interactive Editing:** Provides an intuitive UI to manually edit corrected text.
    -   💾 **Standard Export:** Downloads the final, corrected subtitles in valid `.srt` format.

* **User-Friendly Interface**
    -   ⚙️ **Customizable Options:** Easily toggle spell checking, punctuation removal, and lowercasing.
    -   ⏱️ **Real-time Metrics:** Tracks processing progress and estimates time remaining.

---

### ⚙️ Installation

1️⃣ **Clone the repository**
```bash
git clone https://github.com/helenmand/greek-srt-corrector.git
cd greek-srt-corrector
```
2️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

3️⃣ **Install & Run Ollama**

Follow the official Ollama installation guide. After installation, pull the required Greek model:

```bash
ollama pull ilsp/meltemi-instruct-v1.5
```

4️⃣ **Run the app**

```bash
streamlit run srt_corrector.py
```

---

### ⚙️ Why Meltemi?

The application suggests using `ilsp/meltemi-instruct-v1.5` model because it offers a balance of performance and accuracy for Greek-specific tasks.

While larger, more general-purpose models like `gemma3:12b` may have a higher overall quality rating, the Meltemi model is:

* **Lightweight:** At 7 billion parameters, it requires significantly less VRAM, making the app more accessible.
* **Highly Effective:** Its training data is tailored for the Greek language, ensuring high-quality, task-specific corrections. **Additionally, the application's use of a user-provided context significantly improves the model's performance by tailoring its corrections to the specific subject matter of the text.**

You are free to change the model from the app's sidebar to explore other options that might better suit your needs.

---

### 📌 Example

The app can automatically correct common spelling mistakes while applying your chosen formatting options.

Input:

```bash
1
00:00:02,000 --> 00:00:04,000
<b>Γειατί ήρθες εδώ;</b>
```

Output:
(with spell check, punctuation removal and lowercase options enabled)

```bash
1
00:00:02,000 --> 00:00:04,000
<b>γιατί ήρθες εδώ</b>
```
Correction: The word "Γειατί" was changed to "γιατί", the punctuation mark was removed and the first letter became lowercase
