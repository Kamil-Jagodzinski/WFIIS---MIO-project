# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras import regularizers
from sklearn.model_selection import train_test_split
from datetime import date, datetime
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from scipy.ndimage import shift
from PIL import ImageFont
import visualkeras

# %% [markdown]
# ## Wczytywanie danych

# %%

arial = pd.read_csv('datasets/ARIAL.csv')
agency = pd.read_csv('datasets/AGENCY.csv')
baiti = pd.read_csv('datasets/BAITI.csv')

# %% [markdown]
# Filtracja danych (wybór tylko liter)

# %%
def isletter(letter):
    return chr(letter).isalpha()

letters_arial = arial[arial['m_label'].apply(isletter)]
letters_agency = agency[agency['m_label'].apply(isletter)]
letters_baiti = baiti[baiti['m_label'].apply(isletter)]

# %% [markdown]
# Konkatynacja obrazów w jedną macierz

# %%
X1 = letters_arial.loc[:, "r0c0":].to_numpy()
indices = np.arange(X1.shape[0])
indices = np.random.choice(indices, size=600)
X1 = X1[indices]

X = np.concatenate((X1))

X2 = letters_agency.loc[:, "r0c0":].to_numpy()

X3 = letters_baiti.loc[:, "r0c0":].to_numpy()
indices_x3 = np.arange(X3.shape[0])
indices_x3 = np.random.choice(indices_x3, size=600)
X3 = X3[indices_x3]

X = np.concatenate((X1, X2, X3))

# %%
fonts = ['arial', 'agency', 'baiti']

y = np.concatenate((
    np.full(X1.shape[0], 0),
    np.full(X2.shape[0], 1),
    np.full(X3.shape[0], 2)
))

# %% [markdown]
# Tasowanie zbioru danych

# %%
indices = np.arange(y.shape[0])
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

X = StandardScaler().fit_transform(X)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %% [markdown]
# Agmentacja danych

# %%
def roll(img, shifts, axis):
    img = np.copy(img)
    indexing = [slice(0, img.shape[i], 1) for i in range(img.ndim)]
    for i in range(img.shape[axis]):
        indexing[axis] = i
        img[tuple(indexing)] = np.roll(img[tuple(indexing)], shifts[i])
    return img

Aug_X, Aug_y = [], []

for img, font in zip(X, y):
    
    img = np.reshape(img, (20,20))

    for dx, dy in [(1, 0), (0,   1),   (-1, 0), (0, -1),
                (1, 1), (-1, -1), (-1, 1), (1, -1)]:
        Aug_X.append(shift(img, (dx, dy)))
        Aug_y.append(font)
        
    # shearing
    for _ in range(5):
        # along axis 0
        shifts0 = np.random.randint(0, 2, size=img.shape[0])
        rolled0 = roll(img, shifts0, 0)
        Aug_X.append(rolled0)
        Aug_y.append(font)
        
        # along axis 1
        shifts1 = np.random.randint(0, 2, size=img.shape[1])
        Aug_X.append(roll(img, shifts1, 1))
        Aug_y.append(font)
        
        # along both axes
        Aug_X.append(roll(rolled0, shifts1, 1))
        Aug_y.append(font)

Aug_X = np.reshape(Aug_X, (-1, 400))
Aug_y = np.array(Aug_y)
Aug_X_train, Aug_X_test, Aug_y_train, Aug_y_test = train_test_split(Aug_X, Aug_y, test_size=0.2)

# %% [markdown]
# ## Fabryka modeli

# %% [markdown]
# Normalizacja danych

# %%
def basic_model():
    return Sequential([     
                            Dense(100, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(50, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(3, activation='softmax')
                        ])

def L1_model(L1_val=0.1):
    return  Sequential([    
                            Dense(100, activation='relu'),
                            Dense(50, activation='relu'),
                            Dense(3, activation='softmax', use_bias=True, kernel_regularizer=regularizers.L1(L1_val))
                        ])

def L2_model(L2_val=0.1):
    return  Sequential([    
                            Dense(100, activation='relu'),
                            Dense(50, activation='relu'),
                            Dense(3, activation='softmax', use_bias=True, kernel_regularizer=regularizers.L2(L2_val))
                        ])

def dropout_model(d = 0.7):
    return Sequential([     
                            Dense(100, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dropout(d),
                            Dense(50, activation='relu', use_bias=True, bias_initializer='ones'),
                            Dense(3, activation='softmax')
                        ])
def simplify_model():
    return Sequential([     
                            Dense(3, activation='softmax')
                        ])




# %% [markdown]
# Porównywanie modeli

# %%
def compare_mlp(comp_with: int):
    ''' 1-L1, 2-L2, 3-Dropout,
        4-EarlyStop, 5-Simplify, 6-DataAugmentation'''

    # parametry naucznia - epoki i %danych do walidacji
    val_split=0.3
    epo=200
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
        basic_nets_history.append( mlp_basic.fit(X_train, y_train, verbose=1, validation_split=val_split, epochs=epo) )

        #szkolenie porównywanego modelu
        if comp_with == 4: #szkolenie z early stop
            comp_mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            comp_nets_history.append( comp_mlp.fit(X_train, y_train, verbose=1, 
                                        validation_split=val_split, epochs=epo, callbacks=[EarlyStopping(patience=4)] ) )
        elif comp_with == 6: #szkolenie na rozszerzonym zbiorze danych
            comp_mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            comp_nets_history.append( comp_mlp.fit(Aug_X_train, Aug_y_train, verbose=1, validation_split=val_split, epochs=epo) )
        else: #szkolenie sieci o zmodyfikowanej architekturze
            comp_mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            comp_nets_history.append( comp_mlp.fit(X_train, y_train, verbose=1, validation_split=val_split, epochs=epo) )

        
        # visualkeras.layered_view(comp_mlp, legend=True).show()  # font is optional!

    

    legend_acc, legend_loss = [], []
    plt.figure(figsize=(16, 6))
    for i in range(3):
        # wykres porownania model accuracy
        plt.subplot(1,2,1)
        plt.plot(basic_nets_history[i].history['accuracy'], linewidth=1, color=basic_color_train)
        plt.plot(basic_nets_history[i].history['val_accuracy'], linewidth=1, color=basic_color_valid)
        plt.plot(comp_nets_history[i].history['accuracy'], linewidth=1, color=comp_color_train)
        plt.plot(comp_nets_history[i].history['val_accuracy'], linewidth=1, color=comp_color_valid)

        # wykres porownania model loss
        plt.subplot(1,2,2)
        plt.plot(basic_nets_history[i].history['loss'], linewidth=1, color=basic_color_train)
        plt.plot(basic_nets_history[i].history['val_loss'], linewidth=1, color=basic_color_valid)
        plt.plot(comp_nets_history[i].history['loss'], linewidth=1, color=comp_color_train)
        plt.plot(comp_nets_history[i].history['val_loss'], linewidth=1, color=comp_color_valid)

    # tworzenie legendy do obu wykresow
    plt.subplot(1,2,1)
    legend_acc.append(f"training - basic mlp model")
    legend_acc.append(f"validation - basic mlp model")
    legend_acc.append(f"training - optimized mlp model")
    legend_acc.append(f"validation - optimized mlp model")

    legend_loss.append(f"training - basic mlp model")
    legend_loss.append(f"validation - basic mlp model")
    legend_loss.append(f"training - optimized mlp model")
    legend_loss.append(f"validation - optimized mlp model")

    plt.title(f"Model accuracy\n {plot_title}:")
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(legend_acc, fontsize=10)

    plt.subplot(1,2,2)
    plt.title(f"Model loss\n {plot_title}:")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(legend_loss, fontsize=10)


    today = date.today()
    cur_time = datetime.now().strftime("%H_%M_%S")
    time_stamp = str(f"{today}_{cur_time}")

    #zapis do folderu z time stampem
    plt.savefig(f"font-classification-plots/{time_stamp}_{type_of_plot}.png")
    return comp_mlp



# compare_mlp(1)    #L1 
# compare_mlp(2)    #L2
# compare_mlp(3)    #Dropout
# compare_mlp(4)    #EarlyStop 
# compare_mlp(5)    #Simplify
# compare_mlp(6)    #DataAugmentation

# %%



