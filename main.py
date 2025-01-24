from keras.api.datasets import fashion_mnist
from keras.api.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
from keras_tuner import BayesianOptimization


def build_model(hp):
    model = Sequential()
    activation_choice = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid', 'elu', 'selu'])

    # Входной слой
    model.add(Dense(units=hp.Int('units_input', min_value=512, max_value=1024, step=32),
                    input_dim=784, activation=activation_choice))

    # Скрытые слои
    for i in range(hp.Int('num_layers', 2, 5)):
        model.add(Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
                        activation=activation_choice))

    # Выходной слой
    model.add(Dense(10, activation='softmax'))  # 10 классов для Fashion MNIST

    # Компиляция модели
    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop', 'SGD']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# Загрузка данных
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

# Настройка и запуск тюнера
tuner = BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=20,
    directory='my_dir',
    project_name='fashion_mnist_tuning'
)

tuner.search_space_summary()

# Поиск гиперпараметров
tuner.search(x_train, y_train, batch_size=256, epochs=5, validation_split=0.2, verbose=1)

# Результаты
print(tuner.results_summary())
best_models = tuner.get_best_models(num_models=2)
print(best_models)