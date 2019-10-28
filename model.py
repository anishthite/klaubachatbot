#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.cost import cross_entropy_seq, cross_entropy_seq_with_mask
from tqdm import tqdm
from sklearn.utils import shuffle
from data.twitter import data
from tensorlayer.models.seq2seq import Seq2seq
import os


class Chatbot:
    def __init__(self):
        self.decoder_seq_length = 20
        self.metadata, self.trainX, self.trainY, self.testX, self.testY, self.validX, self.validY = self.initial_setup(data_corpus)
        self.model = Seq2seq(
            decoder_seq_length = self.decoder_seq_length,
            cell_enc=tf.keras.layers.GRUCell,
            cell_dec=tf.keras.layers.GRUCell,
            n_layer=3,
            n_units=256,
            embedding_layer=tl.layers.Embedding(vocabulary_size=self.vocabulary_size, embedding_size=self.emb_dim),
            )
    

    def initial_setup(self, data_corpus):
        metadata, idx_q, idx_a = data.load_data(PATH='data/{}/'.format(data_corpus))
        (trainX, trainY), (testX, testY), (validX, validY) = data.split_dataset(idx_q, idx_a)
        trainX = tl.prepro.remove_pad_sequences(trainX.tolist())
        trainY = tl.prepro.remove_pad_sequences(trainY.tolist())
        testX = tl.prepro.remove_pad_sequences(testX.tolist())
        testY = tl.prepro.remove_pad_sequences(testY.tolist())
        validX = tl.prepro.remove_pad_sequences(validX.tolist())
        validY = tl.prepro.remove_pad_sequences(validY.tolist())


        src_len = len(trainX)
        tgt_len = len(trainY)

        assert src_len == tgt_len

        self.batch_size = 32
        self.n_step = src_len // self.batch_size
        src_vocab_size = len(metadata['idx2w']) # 8002 (0~8001)
        self.emb_dim = 1024

        self.word2idx = metadata['w2idx']   # dict  word 2 index
        self.idx2word = metadata['idx2w']   # list index 2 word

        self.unk_id = self.word2idx['unk']   # 1
        pad_id = self.word2idx['_']     # 0

        self.start_id = src_vocab_size  # 8002
        self.end_id = src_vocab_size + 1  # 8003

        self.word2idx.update({'start_id': self.start_id})
        self.word2idx.update({'end_id': self.end_id})
        self.idx2word = self.idx2word + ['start_id', 'end_id']

        src_vocab_size = tgt_vocab_size = src_vocab_size + 2

        self.num_epochs = 50
        self.vocabulary_size = src_vocab_size
        return metadata, trainX, trainY, testX, testY, validX, validY

    def inference(self, seed, top_n, load=False):
        if load:
            load_weights = tl.files.load_npz(name='model.npz')
            tl.files.assign_weights(load_weights, self.model)
        self.model.eval()
        seed_id = [self.word2idx.get(w, self.unk_id) for w in seed.split(" ")]
        sentence_id = self.model(inputs=[[seed_id]], seq_length=20, start_token=self.start_id, top_n = top_n)
        sentence = []
        for w_id in sentence_id[0]:
            w = self.idx2word[w_id]
            if w == 'end_id':
                break
            sentence = sentence + [w]
        return sentence

    def train(self):
        optimizer = tf.optimizers.Adam(learning_rate=0.001)
        
        for epoch in range(self.num_epochs):
            self.model.train()
            trainX, trainY = shuffle(trainX, trainY, random_state=0)
            total_loss, n_iter = 0, 0
            for X, Y in tqdm(tl.iterate.minibatches(inputs=trainX, targets=trainY, batch_size=self.batch_size, shuffle=False), 
                            total=self.n_step, desc='Epoch[{}/{}]'.format(epoch + 1, self.num_epochs), leave=False):

                X = tl.prepro.pad_sequences(X)
                _target_seqs = tl.prepro.sequences_add_end_id(Y, end_id=self.end_id)
                _target_seqs = tl.prepro.pad_sequences(_target_seqs, maxlen=self.decoder_seq_length)
                _decode_seqs = tl.prepro.sequences_add_start_id(Y, start_id=self.start_id, remove_last=False)
                _decode_seqs = tl.prepro.pad_sequences(_decode_seqs, maxlen=self.decoder_seq_length)
                _target_mask = tl.prepro.sequences_get_mask(_target_seqs)

                with tf.GradientTape() as tape:
                    ## compute outputs
                    output = self.model(inputs = [X, _decode_seqs])
                    
                    output = tf.reshape(output, [-1, self.vocabulary_size])
                    ## compute loss and update model
                    loss = cross_entropy_seq_with_mask(logits=output, target_seqs=_target_seqs, input_mask=_target_mask)

                    grad = tape.gradient(loss, self.model.all_weights)
                    optimizer.apply_gradients(zip(grad, self.model.all_weights))
                
                total_loss += loss
                n_iter += 1

            # printing average loss after every epoch
            print('Epoch [{}/{}]: loss {:.4f}'.format(epoch + 1, self.num_epochs, total_loss / n_iter))

            # for seed in seeds:
            #     print("Query >", seed)
            #     top_n = 3
            #     for i in range(top_n):
            #         sentence = self.inference(seed, top_n)
            #         print(" >", ' '.join(sentence))

            tl.files.save_npz(self.model.all_weights, name='model.npz')


if __name__ == "__main__":
    data_corpus = "twitter" 
    chatbot = Chatbot()
    method = input("Enter 1 for train, enter 2 for inference")
    if method == "1":
        chatbot.train()
    else:    
        while True:
            seed = input("Enter input:")
            print("Query >", seed)
            top_n = 3
            for i in range(top_n):
                sentence = chatbot.inference(seed, top_n)
                print(" >", ' '.join(sentence))



        
    
    
