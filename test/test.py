import tensorflow as tf
import numpy as np
from transformers import TFBertForSequenceClassification, BertTokenizer


model = TFBertForSequenceClassification.from_pretrained("model/tf_model")
tokenizer = BertTokenizer.from_pretrained("model/tokenizer")


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
text = "Невозможно описать запах, но он был ужасным"
logits, probabilities, label = get_prediction_vector(text)

print("Логиты:", logits)
print("Вероятности:", probabilities)
print("Метка:", label, f"- {emotion_labels[label]}")
