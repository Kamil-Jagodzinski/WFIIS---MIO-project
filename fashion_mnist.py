import numpy as np
from matplotlib import pyplot as pl
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import distutils
from datetime import date, datetime
from tensorflow.keras import regularizers
import visualkeras
from PIL import ImageFont





if distutils.version.LooseVersion(tf.__version__) <= '2.0':
    raise Exception(
        'This notebook is compatible with TensorFlow 1.14 or higher, for TensorFlow 1.13 or lower please use the previous version at https://github.com/tensorflow/tpu/blob/r1.13/tools/colab/fashion_mnist.ipynb')



x_size, y_size = 28, 28


def basic_model():
    model = Sequential([    Flatten(input_shape=(x_size, y_size)),
                            Dense(256, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(128, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(64, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(10, activation='softmax')
    ])
    return model


def L1_model(L1_val=0.01):
    return  Sequential([    Flatten(input_shape=(x_size, y_size)),
                            Dense(256, activation='relu', use_bias=True, bias_initializer='ones'  ),
                            Dense(128, activation='relu', use_bias=True, bias_initializer='ones'  ),
                            Dense(64, activation='relu', use_bias=True, bias_initializer='ones' ),
                            Dense(10, kernel_regularizer=regularizers.l1(L1_val), activation='softmax')
                        ])

def L2_model(L2_val=0.01):
    return  Sequential([    Flatten(input_shape=(x_size, y_size)),
                            Dense(256, activation='relu', use_bias=True, bias_initializer='ones' ),
                            Dense(128, activation='relu', use_bias=True, bias_initializer='ones' ),
                            Dense(64, activation='relu', use_bias=True, bias_initializer='ones' ),
                            Dense(10, kernel_regularizer=regularizers.l2(L2_val), activation='softmax')
                        ])  

def dropout_model(d = 0.3):
    return Sequential([     Flatten(input_shape=(x_size, y_size)),
                            Dense(256, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(128, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dropout(d),
                            Dense(64, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(10, activation='softmax')
                        ])
def simplify_model():
    return Sequential([     Flatten(input_shape=(x_size, y_size)),
                            Dense(128, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(10, activation='softmax')
                        ])



# --------------------------------------main------------------------------------

class_names = ['T-shirt/top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
               'Ankle boot']
(X_train, Y_train), (X_test, Y_test) = keras.datasets.fashion_mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0


# model = basic_model()
# history = model.fit(X_train, Y_train,validation_split=0.30, epochs=6)
# result = model.evaluate(X_test, Y_test, verbose=0)

# pl.plot(history.history['accuracy'])
# pl.plot(history.history['val_accuracy'])
# pl.title('model accuracy')
# pl.ylabel('accuracy')
# pl.xlabel('epoch')
# pl.legend(['train', 'validation'])
# pl.show()
# # summarize history for loss
# pl.plot(history.history['loss'])
# pl.plot(history.history['val_loss'])
# pl.title('model loss')
# pl.ylabel('loss')
# pl.xlabel('epoch')
# pl.legend(['train', 'validation'])
# pl.show()

# print('test loss, test acc: ', result)




def compare_mlp(comp_with: int):
    ''' 1-L1, 2-L2, 3-Dropout,
        4-EarlyStop, 5-Simplify, 6-DataAugmentation'''

    # parametry naucznia - epoki i %danych do walidacji
    val_split=0.55
    epo=30
    # kolory na wykresy
    basic_color_train, basic_color_valid = 'blue', 'cyan'
    comp_color_train, comp_color_valid = 'red', 'orange'

    # zawartość do tworzenia wyjściowego wykresu
    basic_nets_history = []
    comp_nets_history = []
    plot_title, type_of_plot = "", ""
    
    for i in range(3):
        mlp_basic = basic_model()

        # nadanie tytułu wykresu i nazwy grafiki oraz stowrznie określonej sieci mlp
        if comp_with == 1:
            plot_title = "Basic MLP vs MLP with regularization L1"
            type_of_plot = "basic_vs_L1" 
            comp_mlp = L1_model()
        if comp_with == 2:
            plot_title = "Basic MLP vs MLP with regularization L2" 
            type_of_plot = "basic_vs_L2" 
            comp_mlp = L2_model()
        if comp_with == 3:
            plot_title = "Basic MLP vs MLP with Dropout"
            type_of_plot = "basic_vs_dropout" 
            comp_mlp = dropout_model()
        if comp_with == 4:
            plot_title = "Basic MLP vs MLP with EarlyStop"
            type_of_plot = "basic_vs_earlystop" 
            comp_mlp = basic_model()
        if comp_with == 5:
            plot_title = "Basic MLP vs Simplifiled MLP"
            type_of_plot = "basic_vs_simplfy" 
            comp_mlp = simplify_model()
        if comp_with == 6: 
            plot_title = "Basic MLP vs Basic MLP with Data Augmentation"
            type_of_plot = "basic_vs_dataAug" 
            comp_mlp = basic_model()

        #szkolenie modelu podstawowego
        mlp_basic.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        basic_nets_history.append( mlp_basic.fit(X_train, Y_train, verbose=1, validation_split=val_split, epochs=epo) )

        #szkolenie porównywanego modelu
        if comp_with == 4: #szkolenie z early stop
            comp_mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            comp_nets_history.append( comp_mlp.fit(X_train, Y_train, verbose=1, validation_split=val_split, epochs=epo, callbacks=[EarlyStopping(patience=2)] ) )
        elif comp_with == 6: #szkolenie na rozszerzonym zbiorze danych

            #dodanie zaszumionych danych
            temp_x = X_train.copy()
            for mtx in temp_x:
                noise = np.random.normal(0, 0.2, mtx.shape)
                mtx = mtx + noise
            X_train_extend = np.concatenate( (X_train, temp_x[: len(temp_x)//4] ), axis=-3)
            Y_train_extend = np.concatenate( (Y_train, Y_train[: len(Y_train)//4] ), axis=0)
            comp_mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            comp_nets_history.append( comp_mlp.fit(X_train_extend, Y_train_extend, verbose=1, validation_split=val_split, epochs=epo) )

        else: #szkolenie sieci o zmodyfikowanej architekturze
            comp_mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            comp_nets_history.append( comp_mlp.fit(X_train, Y_train, verbose=1, validation_split=val_split, epochs=epo) )


    legend_acc, legend_loss = [], []
    pl.figure(figsize=(16, 6))
    for i in range(3):
        # wykres porownania model accuracy
        pl.subplot(1,2,1)
        pl.plot(basic_nets_history[i].history['accuracy'], linewidth=1, color=basic_color_train)
        pl.plot(basic_nets_history[i].history['val_accuracy'], linewidth=1, color=basic_color_valid)
        pl.plot(comp_nets_history[i].history['accuracy'], linewidth=1, color=comp_color_train)
        pl.plot(comp_nets_history[i].history['val_accuracy'], linewidth=1, color=comp_color_valid)

        # wykres porownania model loss
        pl.subplot(1,2,2)
        pl.plot(basic_nets_history[i].history['loss'], linewidth=1, color=basic_color_train)
        pl.plot(basic_nets_history[i].history['val_loss'], linewidth=1, color=basic_color_valid)
        pl.plot(comp_nets_history[i].history['loss'], linewidth=1, color=comp_color_train)
        pl.plot(comp_nets_history[i].history['val_loss'], linewidth=1, color=comp_color_valid)

    # tworzenie legendy do obu wykresow
    pl.subplot(1,2,1)
    legend_acc.append(f"training - basic mlp model")
    legend_acc.append(f"validation - basic mlp model")
    legend_acc.append(f"training - optimized mlp model")
    legend_acc.append(f"validation - optimized mlp model")

    legend_loss.append(f"training - basic mlp model")
    legend_loss.append(f"validation - basic mlp model")
    legend_loss.append(f"training - optimized mlp model")
    legend_loss.append(f"validation - optimized mlp model")

    pl.title(f"Model accuracy\n {plot_title}:")
    pl.ylabel('accuracy')
    pl.xlabel('epoch')
    pl.legend(legend_acc, fontsize=10)

    pl.subplot(1,2,2)
    pl.title(f"Model loss\n {plot_title}:")
    pl.ylabel('loss')
    pl.xlabel('epoch')
    pl.legend(legend_loss, fontsize=10)


    today = date.today()
    cur_time = datetime.now().strftime("%H_%M_%S")
    time_stamp = str(f"{today}_{cur_time}")

    #zapis do folderu z time stampem
    pl.savefig(f"fashion_mnist_plots/{time_stamp}_{type_of_plot}.png")
    return comp_mlp



compare_mlp(1)    #L1
compare_mlp(2)    #L2
compare_mlp(3)    #Dro/pout
compare_mlp(4)    #EarlyStop 
compare_mlp(5)    #Simplify
compare_mlp(6)    #DataAugmentation



#Rysowanie modelu sieci

# model = simplify_model()
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(X_train, Y_train, verbose=1, validation_split=0.3, epochs=5)
# # model = compare_mlp(3)

# font = ImageFont.truetype("arial.ttf", 32)  # using comic sans is strictly prohibited!
# visualkeras.layered_view(model, legend=True, font=font).show()  # font is optional!


#sprawdzenie dzialania sieci na kilku obrazach ze zbioru testowego
# predictions = model.predict(np.array(X_test))

# def show_images(data,number_of_images):
#     n = number_of_images
#     plt.figure(figsize=(20, 4))
#     for i in range(n):
#         # display original
#         ax = plt.subplot(2, n, i + 1)
#         plt.imshow(data[i].reshape(28, 28))
#         plt.title(f"Prediction = {class_names[np.argmax(predictions[i], axis=0)]}\nActual = {class_names[Y_test[i]]}", fontsize=10)
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)

#     plt.show()

# show_images(X_test,10)






