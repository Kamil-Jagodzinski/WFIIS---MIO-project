import numpy as np
from matplotlib import pyplot as pl
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
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

X_train = np.array(X_train)
Y_train = np.array(Y_train)

################################################################################################################################
# funkcje tworzące modele sieci

def basic_model():
    return Sequential([     Flatten(input_shape=(x_size, y_size)),
                            Dense(1000, activation='sigmoid', use_bias=True, bias_initializer='ones'),
                            Dense(500, activation='sigmoid', use_bias=True, bias_initializer='ones'),
                            Dense(250, activation='sigmoid', use_bias=True, bias_initializer='ones'),
                            Dense(100, activation='sigmoid', use_bias=True, bias_initializer='ones'),
                            Dense(10, activation='softmax')
                        ])

def L1_model(L1_val=0.1):
    return  Sequential([    Flatten(input_shape=(x_size, y_size)),
                            Dense(1000, activation='sigmoid', use_bias=True,  kernel_regularizer=regularizers.L1(L1_val) ),
                            Dense(500, activation='sigmoid', use_bias=True, kernel_regularizer=regularizers.L1(L1_val) ),
                            Dense(250, activation='sigmoid', use_bias=True, kernel_regularizer=regularizers.L1(L1_val) ),
                            Dense(100, activation='sigmoid', use_bias=True, kernel_regularizer=regularizers.L1(L1_val) ),
                            Dense(10, activation='softmax')
                        ])

def L2_model(L2_val=0.05):
    return  Sequential([    Flatten(input_shape=(x_size, y_size)),
                            Dense(1000, activation='sigmoid', use_bias=True,  kernel_regularizer=regularizers.L2(L2_val) ),
                            Dense(500, activation='sigmoid', use_bias=True, kernel_regularizer=regularizers.L2(L2_val) ),
                            Dense(250, activation='sigmoid', use_bias=True, kernel_regularizer=regularizers.L2(L2_val) ),
                            Dense(100, activation='sigmoid', use_bias=True, kernel_regularizer=regularizers.L2(L2_val) ),
                            Dense(10, activation='softmax' )
                        ])

def dropout_model(d = 0.2):
    return Sequential([     Flatten(input_shape=(x_size, y_size)),
                            Dense(1000, activation='sigmoid', use_bias=True, bias_initializer='ones'),
                            Dense(500, activation='sigmoid', use_bias=True, bias_initializer='ones'),
                            Dense(250, activation='sigmoid', use_bias=True, bias_initializer='ones'),
                            Dropout(d),
                            Dense(100, activation='sigmoid', use_bias=True, bias_initializer='ones'),
                            Dense(10, activation='softmax')
                        ])
def simplify_model():
    return Sequential([     Flatten(input_shape=(x_size, y_size)),
                            Dense(100, activation='sigmoid', use_bias=True, bias_initializer='ones'),
                            Dense(10, activation='softmax')
                        ])

################################################################################################################################
# porownanie modeli

def compare_mlp(comp_with: int):
    ''' 1-L1, 2-L2, 3-Dropout,
        4-EarlyStop, 5-Simplify, 6-DataAugmentation'''

    # parametry naucznia - epoki i %danych do walidacji
    val_split=0.3
    epo=20
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
            plot_title = "Basic MLP vs MLP with egularization L1"
            type_of_plot = "basic_vs_L1" 
            comp_mlp = L1_model()
        if comp_with == 2:
            plot_title = "Basic MLP vs MLP with egularization L2" 
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
            comp_mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            comp_nets_history.append( comp_mlp.fit(X_train, Y_train, verbose=1, validation_split=val_split, epochs=epo) )
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
    pl.savefig(f"semeion_plots/{time_stamp}_{type_of_plot}.png")
    return comp_mlp


# compare_mlp(1)    #L1
compare_mlp(2)    #L2
# model3 = compare_mlp(3)    #Dro/pout
# compare_mlp(4)    #EarlyStop 
# model5 = compare_mlp(5)    #Simplify
# model5.save('models/semeion/simplyfy_model')
# compare_mlp(6)    #DataAugmentation