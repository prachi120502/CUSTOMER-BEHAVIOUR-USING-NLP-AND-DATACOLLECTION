import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from flask import Flask, request, jsonify
import os

# Generate synthetic e-commerce data
def generate_data():
    np.random.seed(42)
    data = {
        'review_text': [
            'Great product, very satisfied!',
            'Terrible experience, will not buy again.',
            'Okay, but could be better.',
            'Loved it! Highly recommend.',
            'Not worth the price.',
            'Fantastic quality, will purchase again.',
            'Disappointed, it broke after a week.',
            'Excellent service and product.',
            'Mediocre, nothing special.',
            'I am very happy with my purchase.'
        ],
        'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative', 
                      'positive', 'negative', 'positive', 'neutral', 'positive']
    }
    return pd.DataFrame(data)

# Load and preprocess data
df = generate_data()

# Visualize sentiment distribution
sns.countplot(x='sentiment', data=df)
plt.title('Sentiment Distribution')
plt.savefig('sentiment_distribution.png')
plt.show()

# Prepare data for modeling
X = df['review_text']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Evaluate the model
y_pred = model.predict(X_test_vectorized)
print(classification_report(y_test, y_pred))

# Create Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['review_text']
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)
    return jsonify({'sentiment': prediction[0]})

if __name__ == '__main__':
    # Save the model and vectorizer for later use
    import joblib
    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    # Run the Flask app
    app.run(debug=True)