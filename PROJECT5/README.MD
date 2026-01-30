# Spam Detection Using Naive Bayes + TF-IDF

An end-to-end text classification project demonstrating how to build an automated spam detection system using Natural Language Processing (NLP) techniques, TF-IDF vectorization, and Multinomial Naive Bayes classifier.

## Project Overview

This project implements a machine learning pipeline to classify messages as either "ham" (legitimate) or "spam" (unwanted). The system mimics real-world email and SMS filtering applications by processing text data, extracting meaningful features, and using probabilistic classification to make predictions.

## Dataset

The project uses a curated spam dataset containing text messages labeled as either ham or spam:

- **Total Messages**: 40 samples
- **Features**: Text messages (variable length)
- **Classes**: 2 (Ham and Spam)
- **Distribution**: 
  - Ham (legitimate messages): ~60%
  - Spam (unwanted messages): ~40%

### Sample Messages

**Ham Examples:**
- "Hey! Are we still meeting for lunch today?"
- "I'll call you later."
- "Don't forget to bring the documents tomorrow."

**Spam Examples:**
- "Congratulations! You've won a $500 gift card. Click here to claim."
- "URGENT! Your account has been suspended. Click now."
- "FREE entry to the contest. Click here to participate."

### Dataset Source
The dataset is hosted on GitHub and loaded directly into the notebook:
```
https://raw.githubusercontent.com/programmer-sahil/Ardent_ML_Training/main/Project%205/spam_dataset.csv
```

## Prerequisites

Install the required Python libraries:

```bash
pip install pandas scikit-learn matplotlib
```

### Required Libraries:
- **pandas**: Data manipulation and CSV loading
- **scikit-learn**: Machine learning algorithms and tools
- **matplotlib**: Data visualization

## Project Workflow

The spam detection system follows an 8-step pipeline:

### Step 1: Import Libraries
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

### Step 2: Load Dataset
Load the spam dataset directly from GitHub using pandas:
```python
url = "https://raw.githubusercontent.com/programmer-sahil/Ardent_ML_Training/main/Project%205/spam_dataset.csv"
df = pd.read_csv(url)
```

### Step 3: Label Encoding
Convert text labels to numerical format (required for ML algorithms):
```python
df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})
```
- **Ham → 0** (legitimate message)
- **Spam → 1** (spam message)

### Step 4: Train-Test Split
Split data into training (80%) and testing (20%) sets:
```python
X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label_encoded'],
    test_size=0.2,
    random_state=42
)
```

**Why split?**
- Training set: Model learns patterns
- Testing set: Evaluate real-world performance
- 80-20 split: Standard practice for balanced evaluation

### Step 5: TF-IDF Vectorization
Convert text messages into numerical feature vectors:
```python
tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
```

**What is TF-IDF?**
- **TF (Term Frequency)**: How often a word appears in a document
- **IDF (Inverse Document Frequency)**: How unique/important a word is across all documents
- **Result**: Words like "FREE", "WIN", "URGENT" get higher weights in spam messages

### Step 6: Train Naive Bayes Model
Train a Multinomial Naive Bayes classifier:
```python
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
```

**Why Naive Bayes?**
- Very fast training and prediction
- Works exceptionally well for text classification
- Low memory usage
- Based on probability theory (Bayes' theorem)

### Step 7: Make Predictions
Generate predictions on test data:
```python
y_pred = model.predict(X_test_tfidf)
```

### Step 8: Evaluate Model
Assess model performance using multiple metrics:
```python
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
print(confusion_matrix(y_test, y_pred))
```

## Results

### Model Performance

**Accuracy**: 87.5%

### Classification Report
```
              precision    recall  f1-score   support
         ham       1.00      0.83      0.91         6
        spam       0.67      1.00      0.80         2

    accuracy                           0.88         8
   macro avg       0.83      0.92      0.85         8
weighted avg       0.92      0.88      0.88         8
```

### Confusion Matrix
```
[[5 1]   ← 5 ham correctly classified, 1 misclassified as spam
 [0 2]]  ← 0 spam missed, 2 correctly identified as spam
```

**Interpretation:**
- **True Negatives (Ham correctly identified)**: 5
- **False Positives (Ham incorrectly flagged as spam)**: 1
- **False Negatives (Spam missed)**: 0
- **True Positives (Spam correctly caught)**: 2

### Key Insights
1. **Perfect Spam Recall (100%)**: Model caught ALL spam messages
2. **Ham Precision (100%)**: When model says "ham", it's always correct
3. **Trade-off**: 1 legitimate message was incorrectly flagged (acceptable in most cases)

## Performance Metrics Explained

### Accuracy
Overall correctness of predictions: **(TP + TN) / Total**

### Precision
Quality of spam predictions: **TP / (TP + FP)**
- High precision = Few false alarms
- Spam precision (67%): When model predicts spam, it's right 67% of the time

### Recall (Sensitivity)
How many spam messages were caught: **TP / (TP + FN)**
- High recall = Few spam messages slip through
- Spam recall (100%): ALL spam was detected

### F1-Score
Harmonic mean of precision and recall
- Balances both metrics
- Useful when classes are imbalanced

## Real-World Testing

### Custom Message Testing
Test the model with new, unseen messages:

```python
sample_msgs = [
    "Congratulations! You have won a FREE prize. Call now!",
    "Hey bro, are we still meeting tomorrow?"
]

sample_tfidf = tfidf.transform(sample_msgs)
predictions = model.predict(sample_tfidf)
print(predictions)  # Output: [1 0] → [spam, ham]
```

**Results:**
- Message 1 → **Spam** (correctly identified keywords: "Congratulations", "FREE", "Call now")
- Message 2 → **Ham** (conversational tone, no spam indicators)

## Visualization

### Distribution Chart
The project includes a bar chart showing the count of ham vs spam messages:

```python
import matplotlib.pyplot as plt

df['label'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Count of Ham vs Spam Messages')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()
```

**Visual Insights:**
- Green bar: Ham messages (legitimate)
- Red bar: Spam messages (unwanted)
- Shows class distribution in dataset

## How TF-IDF Works

### Example Breakdown

Consider the message: **"Win FREE tickets now!"**

1. **Tokenization**: ["Win", "FREE", "tickets", "now"]
2. **Stop Words Removal**: (Common words like "the", "is" are removed)
3. **TF Calculation**: Frequency of each word in this message
4. **IDF Calculation**: How rare/important each word is across all messages
5. **TF-IDF Score**: TF × IDF for each word

**Result**: Words like "FREE" and "Win" get HIGH scores in spam messages but LOW scores in ham messages.

## Why Naive Bayes for Spam Detection?

### Advantages:
1. **Speed**: Extremely fast training and prediction
2. **Text-Friendly**: Natural fit for text classification
3. **Probabilistic**: Provides confidence scores
4. **Low Data Requirements**: Works well even with small datasets
5. **Interpretable**: Easy to understand how it makes decisions

### How It Works:
Naive Bayes uses **Bayes' Theorem** to calculate:

```
P(Spam | Message) = P(Message | Spam) × P(Spam) / P(Message)
```

**In simple terms:**
- What's the probability this message is spam, given these words?
- Combines probabilities of individual words appearing in spam

## Real-World Applications

This spam detection technique is used in:

1. **Email Filters**: Gmail, Outlook, Yahoo Mail
2. **SMS Filtering**: Mobile carrier spam protection
3. **Social Media**: Filtering spam comments and messages
4. **Customer Support**: Auto-routing messages
5. **Fraud Detection**: Identifying phishing attempts
6. **Content Moderation**: Detecting unwanted content

## Project Extensions

### Potential Improvements:

1. **Larger Dataset**
   - Current: 40 messages
   - Improvement: Use 5,000+ messages for better generalization
   - Sources: Kaggle, UCI ML Repository

2. **Advanced Feature Engineering**
   - Add: Message length, punctuation count, capital letters ratio
   - Extract: URLs, phone numbers, email addresses
   - Create: Custom spam indicators

3. **Model Comparison**
   - Try: Logistic Regression, SVM, Random Forest
   - Compare: Accuracy, speed, interpretability
   - Ensemble: Combine multiple models

4. **Hyperparameter Tuning**
   - TF-IDF: Adjust max_features, ngram_range
   - Naive Bayes: Tune alpha (smoothing parameter)
   - Grid Search: Find optimal combinations

5. **Advanced NLP Techniques**
   - Stemming/Lemmatization: Reduce words to root forms
   - N-grams: Capture word sequences (bigrams, trigrams)
   - Word Embeddings: Word2Vec, GloVe for semantic understanding

6. **Deployment**
   - Build: Web API using Flask/FastAPI
   - Deploy: Cloud service (AWS, Google Cloud)
   - Interface: Simple web form for testing

7. **Real-Time Classification**
   - Stream: Process incoming messages instantly
   - Update: Retrain model with new spam patterns
   - Monitor: Track performance over time

## Technical Implementation Details

### TF-IDF Configuration
```python
TfidfVectorizer(stop_words='english')
```
- **stop_words='english'**: Removes common words (the, is, and, etc.)
- **Default max_features**: Uses all unique words
- **Default ngram_range**: (1,1) → single words only

### Naive Bayes Configuration
```python
MultinomialNB()
```
- **Default alpha**: 1.0 (Laplace smoothing)
- **fit_prior**: True (learns class probabilities)
- **class_prior**: None (calculates from data)

## Common Spam Indicators

The model learns to recognize these patterns:

### High Spam Probability:
- All caps words: "FREE", "URGENT", "WINNER"
- Money-related: "$$$", "cash", "prize", "reward"
- Action words: "Click", "Call now", "Act fast"
- Urgency: "Limited time", "Expires today"
- Suspicious links: "Click here", "Claim now"

### High Ham Probability:
- Personal language: "Hey", "Thanks", "See you"
- Specific references: Names, dates, locations
- Conversational tone: Questions, casual language
- Context-specific: Work, family, friends

## Limitations and Considerations

### Current Limitations:
1. **Small Dataset**: Only 40 messages (needs more for production)
2. **Language**: English only (no multilingual support)
3. **Context Blind**: Doesn't understand sarcasm or context
4. **Evolving Spam**: Spammers constantly change tactics
5. **False Positives**: May flag legitimate promotional emails

### Mitigation Strategies:
- **Continuous Learning**: Retrain with new data regularly
- **User Feedback**: Learn from corrections
- **Whitelist/Blacklist**: Trusted senders and known spammers
- **Confidence Threshold**: Adjust sensitivity based on needs
- **Human Review**: Flag uncertain cases for manual check

## Best Practices

1. **Data Quality**
   - Clean, labeled dataset
   - Balanced classes (or handle imbalance)
   - Representative of real-world messages

2. **Preprocessing**
   - Lowercase conversion
   - Remove special characters (optional)
   - Handle URLs, emails, numbers

3. **Evaluation**
   - Use multiple metrics (not just accuracy)
   - Test on unseen data
   - Monitor false positive rate (important for ham messages)

4. **Production Considerations**
   - Save trained model: `pickle` or `joblib`
   - Version control: Track model versions
   - A/B testing: Compare new models with production
   - Monitoring: Track performance in real-time

## Usage

### Running the Notebook:
1. Open in Google Colab or Jupyter Notebook
2. Run cells sequentially from top to bottom
3. Dataset loads automatically from GitHub
4. View results after Step 8

### Testing Custom Messages:
```python
# Your own messages
my_messages = [
    "Your message here",
    "Another message to test"
]

# Transform and predict
transformed = tfidf.transform(my_messages)
predictions = model.predict(transformed)

# Interpret results
for msg, pred in zip(my_messages, predictions):
    label = "SPAM" if pred == 1 else "HAM"
    print(f"{label}: {msg}")
```

## Key Learning Outcomes

After completing this project, you will understand:

1. **Text Preprocessing**: Converting text to numerical features
2. **TF-IDF**: How to weight word importance
3. **Naive Bayes**: Probabilistic classification
4. **Model Evaluation**: Precision, recall, F1-score
5. **Real-World NLP**: Practical text classification pipeline
6. **Production Thinking**: Considerations for deployment

## Resources for Further Learning

### Algorithms:
- Naive Bayes: Andrew Ng's ML course
- TF-IDF: Information Retrieval basics
- Text Classification: Natural Language Processing courses

### Tools:
- Scikit-learn documentation: Text feature extraction
- NLTK/spaCy: Advanced NLP preprocessing
- Kaggle: Spam datasets and competitions

### Advanced Topics:
- Deep Learning for NLP: RNN, LSTM, BERT
- Ensemble Methods: Voting, stacking classifiers
- Active Learning: Improving with user feedback

## Contributing

This is an educational project. Suggestions for improvement:
- Larger, more diverse datasets
- Multi-language support
- Deep learning approaches
- Real-time deployment examples

## License

Educational project for learning purposes. Dataset credits to the original source.

## Acknowledgments

- Dataset: Custom spam dataset (GitHub)
- Libraries: Scikit-learn, Pandas, Matplotlib
- Algorithm: Multinomial Naive Bayes
- Technique: TF-IDF vectorization

## Important Notes

### For Production Use:
- This is a demonstration project with a small dataset
- Real-world spam filters require:
  - Thousands of training examples
  - Regular retraining
  - Multiple models and features
  - User feedback loops
  - Sophisticated preprocessing

### Ethical Considerations:
- Privacy: Handle user messages securely
- Transparency: Users should know about automated filtering
- Fairness: Avoid bias in spam classification
- Control: Allow users to override decisions

---

**Remember**: Spam detection is an ongoing challenge. Spammers constantly evolve their tactics, so models need continuous updates and monitoring. This project demonstrates the core concepts that power real-world spam filters, scaled down for educational purposes.

## Project Statistics

- **Lines of Code**: ~80
- **Accuracy Achieved**: 87.5%
- **Training Time**: < 1 second
- **Prediction Time**: < 0.1 seconds
- **Model Type**: Supervised Learning (Classification)
- **Application Domain**: Natural Language Processing (NLP)

**Built with ❤️ for learning and understanding spam detection fundamentals.**
