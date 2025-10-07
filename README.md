# 🚀 Next Word Predictor using LSTM
🔗 [Colab Notebook](https://colab.research.google.com/drive/1PVXHaBerGX5W608ps-LoyUN1DL_-5epj?usp=sharing)

---

## 📌 Project Overview
This mini project demonstrates the implementation of a **Next Word Prediction** model using **Long Short-Term Memory (LSTM)** networks.  
The model learns from a text corpus and predicts the most likely word that follows a given sequence.  

This project was built to **understand and practice LSTM implementation in NLP** and showcases how deep learning can be applied in predictive text applications like autocomplete systems.

---

## 🎯 Objectives
- Learn the basics of text preprocessing and tokenization.
- Implement a sequential model using **LSTMs** for text prediction.
- Train and evaluate the model using a custom dataset.
- Demonstrate the use of embeddings and recurrent neural networks in NLP tasks.

---

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Libraries & Frameworks:** TensorFlow, Keras, NumPy, Pandas, NLTK, Matplotlib  

---

## 📚 Dataset
The dataset used in this project consists of **Frequently Asked Questions (FAQ)** from the *Cisco Networking Bootcamp*.  
It contains a collection of short technical Q&A texts used to train the LSTM model on real networking phrases.

📄 **Download / View Dataset:**  
[Cisco Networking Bootcamp FAQ Dataset (PDF)](https://github.com/user-attachments/files/22745489/New.Microsoft.Word.Document.4.pdf)

<details>
<summary>📄 Sample Dataset</summary>

**Q1:** What is a router?  
**A1:** A router is a device that forwards data packets between computer networks.  

**Q2:** What is an IP address?  
**A2:** An IP address is a unique identifier assigned to each device on a network.  

**Q3:** What is a switch?  
**A3:** A switch connects multiple devices within a LAN and forwards data only to the intended device.  

</details>

> **Preprocessing:** The text data was cleaned, tokenized, and converted into sequences to train the LSTM-based next-word prediction model.  

---

## 🔑 Features
- Text preprocessing: lowercasing, punctuation removal, tokenization.
- Sequence generation for model training.
- LSTM-based architecture with embedding layer.
- Predicts the **next word** given an input phrase.
- Generates **Top-N probable predictions** using softmax probabilities.
- Easily extendable for autocomplete or chatbot applications.

---

## ⚙️ How It Works
1. **Prepare the dataset** → Clean and tokenize the text corpus.  
2. **Generate sequences** → Convert sentences into (X, y) pairs for training.  
3. **Build the model** → Embedding layer → LSTM layers → Dense + Softmax.  
4. **Train the model** → Optimize using categorical cross-entropy and Adam optimizer.  
5. **Make predictions** → Input a phrase, and the model predicts the next word.  

---

## 📊 Results & Applications
- Successfully predicts context-aware next words.  
- Applications:  
  - Autocomplete in search engines & messaging apps  
  - Smart assistants (Google Assistant, Alexa, Siri)  
  - Code autocompletion systems  

---

## 🧠 Model Limitations & Ongoing Work
I have not deployed this model yet because I am currently working on improving and generalizing it further.  

The current model performs well on small text corpora but tends to **overfit** on larger datasets.  
This happens because:
- A **fixed number of hidden layers and units** cannot handle both small and large text equally well.
- Increasing layers helps learn complex dependencies but may cause overfitting.
- With fewer layers, the model underfits and struggles to capture longer context.

I am currently working on:
- Generalizing the model to adapt its depth dynamically based on dataset size.
- Applying regularization (dropout, early stopping).
- Testing on more diverse datasets for better generalization.

Once stabilized, I plan to deploy this model using **Streamlit** for real-time predictions.

---

## 🚀 Future Improvements
- Use **pre-trained embeddings** (Word2Vec, GloVe, FastText).  
- Experiment with **Bidirectional LSTMs** and **GRUs**.  
- Extend to **Transformer-based models** (GPT, BERT).  
- Deploy as a simple web app using Flask/Django.  

---

## 🧪 Example Predictions
| Input | Predicted Next Word |
|--------|-------------------|
| The sun rises in the | morning |
| Artificial intelligence is changing | everything |
| People wake up in the | morning |

---

## 🖼️ Screenshots
<img width="747" height="765" alt="Training" src="https://github.com/user-attachments/assets/88e6095c-700d-44a6-9f87-5d0b46a12f1e" />
<img width="835" height="801" alt="Loss Curve" src="https://github.com/user-attachments/assets/1452c78e-003d-42db-a10d-dbc5a21138d5" />
<img width="756" height="698" alt="Prediction" src="https://github.com/user-attachments/assets/eed9d893-dfef-49a6-b3d5-f4821cef7740" />
<img width="876" height="789" alt="Training Output" src="https://github.com/user-attachments/assets/44fafc35-0e51-4c20-8b4e-9a9d05bb9431" />
<img width="798" height="692" alt="Sample Run" src="https://github.com/user-attachments/assets/c0c63c3d-ac1e-47cd-912f-f70ce9661add" />

---

## 📅 Project Timeline
- **Duration:** May 2025 – June 2025  
- **Type:** Mini Project (LSTM implementation in NLP)

---

## 👨‍💻 Author
**Naman Agrawal**  
IIT Bhubaneswar  
📧 [LinkedIn](https://linkedin.com/in/yourusername) | 🌐 [GitHub](https://github.com/yourusername)
