# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

# Load dataset
df = pd.read_csv('/content/drive/MyDrive/lp5 dl/imdb_master.csv', encoding='latin-1')
df = df[['review', 'label']]
df = df[df['label'].isin(['pos', 'neg'])]

# Encode labels: pos = 1, neg = 0
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Class meaning
# Positive (1): User enjoyed the movie
# Negative (0): User disliked the movie

# Class balance visualization
sns.countplot(x='label', data=df)
plt.xticks([0, 1], ['Negative', 'Positive'])
plt.title('Class Distribution')
plt.show()

# 🌥 WordCloud for each sentiment
for label in [0, 1]:
    text = " ".join(df[df['label'] == label]['review'].values)
    wordcloud = WordCloud(max_words=200, background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Positive Reviews" if label == 1 else "Negative Reviews")
    plt.show()

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['label'], test_size=0.2, random_state=42)

# Tokenization & padding
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=200, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=200, padding='post', truncating='post')

# Build Deep Neural Network
model = Sequential([
    Embedding(input_dim=10000, output_dim=16, input_length=200),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_pad, y_train, epochs=10, validation_data=(X_test_pad, y_test), batch_size=32)

# Accuracy/Loss plots
def plot_history(history):
    plt.figure(figsize=(12, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.show()

plot_history(history)

# Evaluate
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f"\nFinal Test Accuracy: {accuracy:.4f}")

# 🔍 Confusion matrix & classification report
y_pred = (model.predict(X_test_pad) > 0.5).astype("int32")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
