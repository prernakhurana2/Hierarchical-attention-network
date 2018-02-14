from sklearn import metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, Lambda, Activation
from keras.layers import Dense, Input, Flatten, RepeatVector, Permute
from keras.layers import merge, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K

earlyStopping = EarlyStopping(monitor='val_loss', patience=1, verbose=0)
architecture = 'HAN'
weight_fname = architecture
save_best_model_file = weight_fname+'.h5'
saveBestModel = ModelCheckpoint(save_best_model_file, monitor = 'val_loss', verbose=0, save_best_only = True, save_weights_only = True)

embedding_layer = Embedding(len(w2indx) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_weights],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True)

sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
  
lstm_sentence = Bidirectional(GRU(nodes, return_sequences=True, recurrent_dropout=dropout, W_regularizer=l2(W_reg)))(embedded_sequences)
dense_sentence = Dense(nodes, activation='tanh')(lstm_sentence) 

#Attention Layer
attention = Dense(1, activation='tanh')(lstm_sentence) # try diff act
attention = Flatten()(attention)
attention = Activation('softmax')(attention) # try different activations
attention = RepeatVector(nodes)(attention)
attention = Permute([2, 1])(attention)
sent_representation = merge([dense_sentence, attention], mode='mul')
sentence_attention = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(nodes,))(sent_representation)

sentEncoder = Model(sentence_input, sentence_attention)

dialogue_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH,), dtype='int32')
dialogue_encoder = TimeDistributed(sentEncoder)(dialogue_input)
lstm_dialogue = Bidirectional(GRU(nodes, return_sequences=True, recurrent_dropout=dropout, W_regularizer=l2(W_reg)))(dialogue_encoder)
dense_dialogue = TimeDistributed(Dense(nodes, activation='tanh'))(lstm_dialogue) 

#Attention Layer
attention = Dense(1, activation='tanh')(dense_dialogue) # try diff act
attention = Flatten()(attention)
attention = Activation('softmax')(attention) # try different activations
attention = RepeatVector(nodes)(attention)
attention = Permute([2, 1])(attention)
dialogue_representation = merge([dense_dialogue, attention], mode='mul')
dialogue_attention = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(nodes,))(dialogue_representation)

preds = Dense(classes, activation='softmax')(dialogue_attention)
model = Model(dialogue_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'],callbacks=[earlyStopping, saveBestModel])

print("model fitting - Hierachical LSTM")
print model.summary()
       
# learn new model                  
model.fit(x_train, y_train, validation_data=(x_test, y_test_cat),
          nb_epoch=nb_epoch, batch_size=batch_size, verbose=2)

       
y_proba = model.predict(x_test, verbose=2)
y_pred = y_proba.argmax(axis = 1)

acc_classification = metrics.accuracy_score(y_test, y_pred)
print("Acc for classification model - ", acc_classification)

