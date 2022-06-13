from re import L
import numpy as np
from matplotlib import pyplot as pl
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from datetime import date, datetime

################################################################################################################################
# wczytanie danych 

data_path = "datasets/semeion.data"
x_size, y_size = 16, 16
ovr_size = x_size * y_size

with open(data_path, 'r') as dataset:
    data = [ line.split() for line in dataset ]

X_train, Y_train = [], []

for d in data:
    X_train.append( np.reshape( list( np.float_(d[:ovr_size]) ) , (x_size, y_size)) )
    Y_train.append( d[-10:].index('1') )

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.02)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

################################################################################################################################
# funkcje tworzące modele sieci

def basic_model():
    return Sequential([     Flatten(input_shape=(x_size, y_size)),
                            Dense(1000, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(500, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(250, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(100, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(10, activation='softmax')
                        ])

def L1_model(L1_val=0.05):
    return  Sequential([    Flatten(input_shape=(x_size, y_size)),
                            Dense(1000, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(500, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(250, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(100, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(10, activation='softmax', kernel_regularizer=regularizers.l1(L1_val))
                        ])

def L2_model(L2_val=0.15):
    return  Sequential([    Flatten(input_shape=(x_size, y_size)),
                            Dense(1000, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(500, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(250, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(100, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(L2_val))
                        ])

def L1L2_model(L1_val=0.1, L2_val=0.3):
    return  Sequential([    Flatten(input_shape=(x_size, y_size)),
                            Dense(1000, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(500, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(250, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(100, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(10, activation='softmax', kernel_regularizer=regularizers.L1(L1_val), activity_regularizer=regularizers.l2(L2_val) )
                        ])

def dropout_model(d = 0.2):
    return Sequential([     Flatten(input_shape=(x_size, y_size)),
                            Dropout(d),
                            Dense(1000, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(500, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(250, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dropout(d),
                            Dense(100, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(10, activation='softmax')
                        ])
def simplify_model():
    return Sequential([     Flatten(input_shape=(x_size, y_size)),
                            Dense(250, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(10, activation='softmax')
                        ])

################################################################################################################################
# porownanie modeli

def compare_mlp(comp_with: int, single_model: bool):
    ''' 1-L1, 2-L2, 3-Dropout,
        4-EarlyStop, 5-Simplify, 6-DataAugmentation'''

    # parametry naucznia - epoki i %danych do walidacji
    val_split=0.3
    epo=150
    models_comparisons = 7
    if single_model is True:
        models_comparisons = 1
    # kolory na wykresy
    basic_color_train, basic_color_valid = 'blue', 'cyan'
    comp_color_train, comp_color_valid = 'red', 'orange'

    # zawartość do tworzenia wyjściowego wykresu
    basic_nets_history = []
    comp_nets_history = []
    plot_title, type_of_plot = "", ""
    
    for i in range(models_comparisons):
        mlp_basic = basic_model()

        # nadanie tytułu wykresu i nazwy grafiki oraz stowrznie określonej sieci mlp
        if comp_with == 1:
            plot_title = "MLP with regularization L1"
            type_of_plot = "basic_vs_L1" 
            comp_mlp = L1_model()
        if comp_with == 2:
            plot_title = "MLP with regularization L2" 
            type_of_plot = "basic_vs_L2" 
            comp_mlp = L2_model()
        if comp_with == 3:
            plot_title = "MLP with Dropout"
            type_of_plot = "basic_vs_dropout" 
            comp_mlp = dropout_model()
        if comp_with == 4:
            plot_title = "MLP with EarlyStop"
            type_of_plot = "basic_vs_earlystop" 
            comp_mlp = basic_model()
        if comp_with == 5:
            plot_title = "Simplifiled MLP"
            type_of_plot = "basic_vs_simplfy" 
            comp_mlp = simplify_model()
        if comp_with == 6: 
            plot_title = "Basic MLP with Data Augmentation"
            type_of_plot = "basic_vs_dataAug" 
            comp_mlp = basic_model()

        #szkolenie modelu podstawowego
        if single_model is False:
            mlp_basic.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            basic_nets_history.append( mlp_basic.fit(X_train, Y_train, verbose=1, validation_split=val_split, epochs=epo) )

        #szkolenie porównywanego modelu
        if comp_with == 4: #szkolenie z early stop
            comp_mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            comp_nets_history.append( comp_mlp.fit(X_train, Y_train, verbose=1, validation_split=val_split, epochs=epo, callbacks=[EarlyStopping(patience=5)] ) )
        elif comp_with == 6: #szkolenie na rozszerzonym zbiorze danych
            
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
    for i in range(models_comparisons):
        # wykres porownania model accuracy
        pl.subplot(1,2,1)
        if single_model is False:
            pl.plot(basic_nets_history[i].history['accuracy'], linewidth=1, color=basic_color_train)
            pl.plot(basic_nets_history[i].history['val_accuracy'], linewidth=1, color=basic_color_valid)
        pl.plot(comp_nets_history[i].history['accuracy'], linewidth=1, color=comp_color_train)
        pl.plot(comp_nets_history[i].history['val_accuracy'], linewidth=1, color=comp_color_valid)

        # wykres porownania model loss
        pl.subplot(1,2,2)
        if single_model is False:
            pl.plot(basic_nets_history[i].history['loss'], linewidth=1, color=basic_color_train)
            pl.plot(basic_nets_history[i].history['val_loss'], linewidth=1, color=basic_color_valid)
        pl.plot(comp_nets_history[i].history['loss'], linewidth=1, color=comp_color_train)
        pl.plot(comp_nets_history[i].history['val_loss'], linewidth=1, color=comp_color_valid)

    # tworzenie legendy do obu wykresow
    pl.subplot(1,2,1)
    if single_model is False:
        legend_acc.append(f"training - basic mlp model")
        legend_acc.append(f"validation - basic mlp model")
    legend_acc.append(f"training - optimized mlp model")
    legend_acc.append(f"validation - optimized mlp model")

    if single_model is False:
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
    pl.ylim([0, 2.5])
    pl.legend(legend_loss, fontsize=10)


    today = date.today()
    cur_time = datetime.now().strftime("%H_%M_%S")
    time_stamp = str(f"{today}_{cur_time}")

    #zapis do folderu z time stampem
    pl.savefig(f"semeion_plots/{time_stamp}_{type_of_plot}.png")
    pl.clf()
    return comp_mlp

print('Hand Written Digit Recognition')
print("1) Test model with L1")
print("2) Test model with L2")
print("3) Test model with Dropout")
print("4) Test model with EarlyStop")
print("5) Test model with Simplify")
print("6) Test model with DataAugmentation")
print("other) Train all models and save results")

model_no = int(input('Select model for test: '))
if model_no == 1:
    print("1) Test model with L1")
    model = compare_mlp(1, True)

elif model_no == 2:
    print("2) Test model with L2")
    model = compare_mlp(2, True)

elif model_no == 3:
    print("3) Test model with Dropout")
    model = compare_mlp(3, True)

elif model_no == 4:
    print("4) Test model with EarlyStop")
    model = compare_mlp(4, True)

elif model_no == 5:
    print("5) Test model with Simplify")
    model = compare_mlp(5, True)

elif model_no == 6:
    print("6) Test model with DataAugmentation")
    model = compare_mlp(6, True)

else:
    print("Train all models and save results")
    compare_mlp(1, False)   #L1
    compare_mlp(2, False)   #L2
    compare_mlp(3, False)   #Dropout
    compare_mlp(4, False)   #EarlyStop 
    compare_mlp(5, False)   #Simplify
    compare_mlp(6, False)   #DataAugmentation

if model_no in [1,2,3,4,5,6]:
    predictions = model.predict(np.array(X_test))
    for i in range(len(X_test)):
        pl.subplot(1 + len(X_test)//12, 12, i+1)
        pl.imshow(np.array(X_test[i]), cmap=pl.get_cmap('YlGnBu'))
        pl.title(f"P={np.argmax(predictions[i], axis=0)}\nA={Y_test[i]}", fontsize=10)
        pl.axis('off')
    pl.show()





