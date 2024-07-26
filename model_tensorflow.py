import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import seaborn as sns
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import prepare_stopwords
import standard_data


def process(input, language):
    
    data = input

    # sns.countplot(x='spam', data=data)
    # plt.show()

    # Downsampling để cân bằng
    print("Downsampling")
    ham_msg = data[data.spam == 0]
    spam_msg = data[data.spam == 1]
    ham_msg = ham_msg.sample(n = len(spam_msg), random_state = 42)
    

    balanced_data = pd.concat([ham_msg, spam_msg], ignore_index = True)

    # sns.countplot(x='spam', data=balanced_data)
    # plt.show()

    # 2. Tiền xử lí dữ liệu
    print("Tiền xử lí dữ liệu")
    if language == "vi":
        balanced_data['text'] = balanced_data['text'].apply(lambda text: standard_data.standard_vi(text))
    else:
        balanced_data['text'] = balanced_data['text'].apply(lambda text: standard_data.standard_en(text))

    # 3. Chia dữ liệu
    print("Chia dữ liệu")
    train_X, test_X, train_Y, test_Y = train_test_split(balanced_data['text'],
                                                        balanced_data['spam'],
                                                        test_size = 0.2,
                                                        random_state = 42)


    # 4. Tokenize dữ liệu
    print("Tokenize dữ liệu")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_X)


    train_sequences = tokenizer.texts_to_sequences(train_X)
    test_sequences = tokenizer.texts_to_sequences(test_X)

    # Đệm chuỗi -> cùng độ dài
    max_len = 100 # độ dài tối đa
    train_sequences = pad_sequences(train_sequences, maxlen = max_len, padding = 'post', truncating = 'post')
    test_sequences = pad_sequences(test_sequences, maxlen = max_len, padding = 'post', truncating = 'post')


    # Mô hình
    # Điều chỉnh LSTM và thêm dropout để hiệu quả
    print("Mô hình")
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim = len(tokenizer.word_index) + 1,
                                        output_dim = 32,
                                        input_length = max_len))
    model.add(tf.keras.layers.LSTM(16))
    model.add(tf.keras.layers.Dense(32, activation = 'relu'))
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))


    model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
                metrics = ['accuracy'],
                optimizer = 'adam')


    es = EarlyStopping(patience=3,
                    monitor = 'val_accuracy',
                    restore_best_weights = True)

    lr = ReduceLROnPlateau(patience = 2,
                        monitor = 'val_loss',
                        factor = 0.5,
                        verbose = 0)


    # Train
    print("Train")
    history = model.fit(train_sequences, train_Y,
                        validation_data=(test_sequences, test_Y),
                        epochs=20,
                        batch_size=32,
                        callbacks = [lr, es]
                    )


    # Đánh giá
    print("Đánh giá")
    test_loss, test_accuracy = model.evaluate(test_sequences, test_Y)
    print('Test Loss :',test_loss)
    print('Test Accuracy :',test_accuracy)

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    return model, tokenizer