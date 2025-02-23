# NLP-ML
sentiment analysis of a text with 7 emotions using the rubert-base-cased model

```python
import pandas as pd
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.model_selection import train_test_split
```

    D:\Anaconda\envs\ML\lib\site-packages\tqdm\auto.py:22: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    


```python
# Загрузка и подготовка данных
def prepare_data(file_path):
    # Загрузка данных
    df = pd.read_csv(file_path, sep=';', header=None, names=['label', 'text'], encoding='windows-1251')
    # Разделение на тексты и метки
    X = df['text'].values
    y = df['label'].values
    
    return X, y
```


```python
# Путь к файлу
file_path = "dataset(text)/train.csv"  # Укажите путь к вашему файлу

# Подготовка данных
X, y = prepare_data(file_path)
```


```python
print(X[0], y[0])
```

    Сегодня на улице ясная погода. 0
    


```python
# Загрузка токенизатора и модели
model_name = "DeepPavlov/rubert-base-cased"  # или другая версия RuBERT
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=7, from_pt=True)
```

    Some weights of the PyTorch model were not used when initializing the TF 2.0 model TFBertForSequenceClassification: ['bert.embeddings.position_ids']
    - This IS expected if you are initializing TFBertForSequenceClassification from a PyTorch model trained on another task or with another architecture (e.g. initializing a TFBertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing TFBertForSequenceClassification from a PyTorch model that you expect to be exactly identical (e.g. initializing a TFBertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights or buffers of the TF 2.0 model TFBertForSequenceClassification were not initialized from the PyTorch model and are newly initialized: ['classifier.weight', 'classifier.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    


```python
# Токенизация данных
def tokenize_data(texts, labels):
    tokenized = tokenizer(
        texts.tolist(),  # Преобразуем в список
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="tf",
    )
    return tokenized, labels
```


```python
# Разделение данных на обучающую и валидационную выборки
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Токенизация обучающих и валидационных данных
train_encodings, train_labels = tokenize_data(X_train, y_train)
val_encodings, val_labels = tokenize_data(X_val, y_val)

# Преобразование в TensorFlow Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels,
)).shuffle(len(train_labels)).batch(16)

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_labels,
)).batch(16)
```


```python
# Компиляция модели
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Для целочисленных меток
    metrics=["accuracy"],
)
```


```python
# Обучение модели
epochs = 4
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
)
```

    Epoch 1/4
    421/421 [==============================] - 80s 164ms/step - loss: 0.4412 - accuracy: 0.8577 - val_loss: 0.2985 - val_accuracy: 0.9198
    Epoch 2/4
    421/421 [==============================] - 67s 160ms/step - loss: 0.1248 - accuracy: 0.9656 - val_loss: 0.2118 - val_accuracy: 0.9545
    Epoch 3/4
    421/421 [==============================] - 67s 159ms/step - loss: 0.0587 - accuracy: 0.9836 - val_loss: 0.1514 - val_accuracy: 0.9652
    Epoch 4/4
    421/421 [==============================] - 68s 160ms/step - loss: 0.0476 - accuracy: 0.9884 - val_loss: 0.1508 - val_accuracy: 0.9626
    


```python
# Сохранение модели
model.save_pretrained("dataset(text)/models")
tokenizer.save_pretrained("dataset(text)/models")
```




    ('dataset(text)/models\\tokenizer_config.json',
     'dataset(text)/models\\special_tokens_map.json',
     'dataset(text)/models\\vocab.txt',
     'dataset(text)/models\\added_tokens.json')




```python
import matplotlib.pyplot as plt
import numpy as np

tr_acc = history.history['accuracy']
tr_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]
index_acc = np.argmax(val_acc)
acc_highest = val_acc[index_acc]

loss_label = f'best epoch= {str(index_loss + 1)}'
acc_label = f'best epoch= {str(index_acc + 1)}'

Epochs = [i+1 for i in range(len(tr_acc))]


plt.figure(figsize= (20, 8))
plt.style.use('fivethirtyeight')

plt.subplot(1, 2, 1)
plt.plot(Epochs, tr_loss, 'orange', label= 'Training loss')
plt.plot(Epochs, val_loss, label= 'Validation loss')
plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'red', label= loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(Epochs, tr_acc, 'orange', label= 'Training Accuracy')
plt.plot(Epochs, val_acc, label= 'Validation Accuracy')
plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'red', label= acc_label)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
```


    
![png](output_10_0.png)
    



```python
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report

# Путь к файлу
file_path = "dataset(text)/test.csv"  # Укажите путь к вашему файлу

# Подготовка данных
X_test, y_test = prepare_data(file_path)
```


```python
def tokenize_data(texts):
    return tokenizer(
        texts.tolist(),  # Преобразуем в список
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="tf",
    )
```


```python
test_encodings = tokenize_data(X_test)
```


```python
test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
)).batch(16)

# Предсказание на тестовых данных
predictions = model.predict(test_dataset)
predicted_labels = tf.argmax(predictions.logits, axis=-1).numpy()

# Если есть истинные метки, оцениваем модель
if y_test is not None:
    accuracy = accuracy_score(y_test, predicted_labels)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Подробный отчет по классификации
    print(classification_report(y_test, predicted_labels))
else:
    # Если меток нет, просто выводим предсказания
    print("Predicted labels:", predicted_labels)
```

    Accuracy: 0.9786
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        20
               1       1.00      1.00      1.00        20
               2       0.91      1.00      0.95        20
               3       1.00      1.00      1.00        20
               4       1.00      1.00      1.00        20
               5       0.95      1.00      0.98        20
               6       1.00      0.85      0.92        20
    
        accuracy                           0.98       140
       macro avg       0.98      0.98      0.98       140
    weighted avg       0.98      0.98      0.98       140
    
    


```python
import tensorflow as tf

# Функция для получения вектора предсказаний
def get_prediction_vector(text):
    # Токенизация текста
    inputs = tokenizer(
        text,
        return_tensors="tf",
        padding=True,
        truncation=True,
        max_length=128,       
    )
    outputs = model(inputs)
    logits = outputs.logits
    predicted_labels = tf.argmax(logits, axis=-1).numpy()[0]
    probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
    
    return logits.numpy()[0], probabilities, predicted_labels
```


```python
emotion_labels = {
    0: "нейтральный",
    1: "счастье",
    2: "печаль",
    3: "энтузиазм",
    4: "страх",
    5: "отвращение",
    6: "гнев",
}
# Пример использования
text = "Мича пупсик, юна пупсик"
logits, probabilities, label = get_prediction_vector(text)

print("Логиты:", logits)
print("Вероятности:", probabilities)
print("Метка:", label, f"- {emotion_labels[label]}")
```

    Логиты: [-1.4823898   6.5051713  -0.19415474 -0.7500228  -1.0005175  -0.6821778
     -2.33525   ]
    Вероятности: [3.3839967e-04 9.9628520e-01 1.2271660e-03 7.0387073e-04 5.4790406e-04
     7.5328193e-04 1.4422393e-04]
    Метка: 1 - счастье
    


```python
from transformers import TFBertForSequenceClassification, BertTokenizer
model = TFBertForSequenceClassification.from_pretrained("dataset(text)/models")
tokenizer = BertTokenizer.from_pretrained("dataset(text)/models")

```

    Some layers from the model checkpoint at dataset(text)/models were not used when initializing TFBertForSequenceClassification: ['dropout_37']
    - This IS expected if you are initializing TFBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing TFBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    All the layers of TFBertForSequenceClassification were initialized from the model checkpoint at dataset(text)/models.
    If your task is similar to the task the model of the checkpoint was trained on, you can already use TFBertForSequenceClassification for predictions without further training.
    


```python

```
