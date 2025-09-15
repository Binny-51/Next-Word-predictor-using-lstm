# 🚀 Next Word Predictor using LSTM

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



## 🔑 Features
- Text preprocessing: lowercasing, punctuation removal, tokenization.
- Sequence generation for model training.
- LSTM-based architecture with embedding layer.
- Predicts the **next word** given an input phrase.
- Generates **Top-N probable predictions** using softmax probabilities.

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

## 🚀 Future Improvements
- Use **pre-trained embeddings** (Word2Vec, GloVe, FastText).  
- Experiment with **Bidirectional LSTMs** and **GRUs**.  
- Extend to **Transformer-based models** (GPT, BERT).  
- Deploy as a simple web app using Flask/Django.  

---

## 📅 Project Timeline
- **Duration:** May 2025 – June 2025  
- **Type:** Mini Project (for learning LSTM implementation in NLP)

---
