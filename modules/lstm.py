#!/usr/bin/env python
# coding: utf-8
"""
Refer to the following list of references during any confusion:
https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/
"""

import torch
import torch.nn as nn
import pandas as pd
import log
import random
from tqdm import tqdm
import numpy as np
import multiprocessing


__author__ = 'Biribiri,BlackDChase'
__version__ = '1.3.1'


class LSTM(nn.Module):
    """
    LSTM that takes the custom processed Ontario dataset as input and trains
    itself and now can be used to output predicted next datapoints.
    """
    def __init__(self, output_size, input_dim, hidden_dim, layer_dim=1,debug=False):
        super(LSTM, self).__init__()
        self.output_size = output_size

        # number of LSTM layers stacked on top of each other
        self.layer_dim = layer_dim

        # size of input at each time step
        self.input_dim = input_dim

        # size of hidden state and cell state at each time step
        # output of the LSTM will be of the equal to hidden_dim, not input_dim
        self.hidden_dim = hidden_dim

        self.debug=debug

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True)

        # Fully connected linear layer which adjusts its weights 
        # based on correlation between inputs from preceding layer (LSTM layer) and outputs 
        # Since the output from LSTM will depends on the hidden dim hence we need to project them into a vector of size given by output_size (in our case output_size = 13)
        # which is done by this linear fc layer
        self.linear =  nn.Linear(hidden_dim, output_size)

        #TODO, Find a better way to do this
        """
        Final output will be clipped acc to the min-max linear values after passing through the hardTanh layer
        The output = { min_val if out < min_val
                       max_val if out > max_val
                       out      otherwise
        The output will range from [min_val,max_val]
        """
        self.norm = nn.Hardtanh(min_val=-0.01,max_val=2)

    def forward(self, input_batch, batch=True, numpy=False):
        """
        TODO Fix this if necessary, add sane comments

        If numpy=True, then assume input is 2D numpy array, and convert to
        appropriate tensor, do the forward, and then return a numpy array.
        """
        if numpy:
            # we have 2D numpy input
            # convert to tensor
            input_batch = torch.Tensor(input_batch)

            # convert to batch
            input_batch = self.convert_to_batch(input_batch)

        # Initialize initial hidden and cell state to zero
        # TODO Is this sensible?
        hidden_state = torch.zeros(self.layer_dim, input_batch.size(0), self.hidden_dim).requires_grad_()
        cell_state = torch.zeros(self.layer_dim, input_batch.size(0), self.hidden_dim).requires_grad_()

        curr = multiprocessing.current_process()
        if self.debug:
            log.debug(f"Hidden, cell state made {curr.name}")

        # Propagate input through LSTM
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        hidden_state.detach()
        cell_state.detach()

        if self.debug:
            log.debug(f"Input batch: {input_batch.shape}, {curr.name}")
            log.debug(f"Hidden shape: {hidden_state.shape}, {curr.name}")
            log.debug(f"Cell shape: {cell_state.shape}, {curr.name}")


        # With multi threading, mulitple states are not parsed through the
        # model, they get stuck.
        # We therefore parse them one state at a time, then make the stack
        # and return it.
        out, (hn, cn) = self.lstm(input_batch, (hidden_state.detach(), cell_state.detach()))
        if self.debug:
            log.debug(f"LSTM forwarded {curr.name}")
            log.debug(f"hn = {hn}")
            log.debug(f"cn = {cn}")
        # Index hidden state of last time step
        # out.size() --> 100, 28, 100 aka (batch_dim, seq_dim, feature_dim)
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! (batch_dim, feature_dim)
        """
        Norm is Hardtanh, will force thevalues to stay between [-0.5,10], while values should be between
        [1,1], as data is min normalized
        """
        out = self.norm(self.linear(out[:, -1, :]))
        #out.size() --> 100, 10 (batch_dim, output_size)

        # if numpy, then return numpy ndarray
        if numpy:
            out = out.detach().numpy()
            # reshape so that the output is (13), instead of (1, 13)
            out = out.squeeze()
        if self.debug:
            log.debug(f"Forward finished {curr.name}")
        return out

    def create_datasets(self, csv_path):
        """
        Do the actual work of:
        1. verifying the csv dataset
        2. splitting the dataset into training and testing

        @input         = csv_path (path of the csv dataset)
        @output        = train,test 
        """
        df = pd.read_csv(csv_path)
        if self.debug:
            log.debug(f"df.head(5) = {df.head(5)}")

        # Ontario price
        # df.iloc[row slice, column slice]
        y = df.iloc[:,0:1]
        if self.debug:
            log.debug(f"Any negative values in Ontario price: {(y < 0).any().any()}")

        size = len(df)
        if self.debug:
            log.debug(f"size = {size}")

        # split the dataset 9:1 into train and test
        # TODO Do I need to enable grad on them to make them "differentiable"?
        train = torch.Tensor(df.values)
        if self.debug:
            log.debug(f"train.shape = {train.shape}")
        x=int(np.random.rand() * size)
        test = torch.Tensor(df.iloc[x:x+size//10, :].values)
        if self.debug:
            log.debug(f"test.shape = {test.shape}")

        return train, test

    def train(self,
            train_batch,
            model_parameters,
            loss_fn=torch.nn.MSELoss(),
            num_epochs=10000,
            num_timesteps_per_batch=100,
            input_history_length=50,
            learning_rate=0.01):
        """
        Train the LSTM on the custom processed Ontario dataset
        There is no "target feature", our aim is to reproduce the entire set of features as part of the output.
        We do not do any scaling because scaling is a "nice to have", not a necessity.
        We /do/ separate the data into training and testing sets, mostly as a formality and a way to gauge the quality of the training.
        """
        optimizer=torch.optim.Adam(model_parameters, lr=learning_rate)
        train_batch_size = len(train_batch)
        for epoch in tqdm(range(num_epochs)):


            # create a batch input and batch label of 100 different possible inputs
            # create empty batch tensors
            input_batch = torch.Tensor()
            label_batch = torch.Tensor()

            for i in range(num_timesteps_per_batch):
                # input_history_length is fixed size of history for every input for the lstm
                # -1 because randint is inclusive of both args
                rand_start_index = random.randint(0, train_batch_size - input_history_length - 1)

                input_plus_label = train_batch[rand_start_index : rand_start_index + input_history_length]
                if self.debug:
                    log.debug(f"Input plus label shape {input_plus_label.shape}")
                inputData = input_plus_label[:-1]
                label = input_plus_label[-1]
                if self.debug:
                    log.debug(f"input data shape = {inputData}")
                label = label.reshape(-1, label.shape[0])
                if self.debug:
                    log.debug(f"label shape = {label.shape}")

                # Convert 2D row to batch
                inputData = self.convert_to_batch(inputData)

                # append input to input_batch
                input_batch = torch.cat((input_batch, inputData), 0)
                label_batch = torch.cat((label_batch, label), 0)


            # enable gradient accumulation
            # TODO Verify if this is the right spot to put this
            input_batch = input_batch.requires_grad_()

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # forward pass
            output = self.forward(input_batch)

            if self.debug:
                log.debug(f"Output shape = {output.shape}")
                log.debug(f"Label Batch shape = {label_batch.shape}")
                
            # get loss output
            loss = loss_fn(output, label_batch)
            if self.debug:
                log.debug(f"loss = {loss}")

            # get gradients
            loss.backward()

            # Update parameters
            optimizer.step()


    def test(self,
            test_batch,
            loss_fn=torch.nn.MSELoss(),
            num_epochs=10):
        test_batch_size = len(test_batch)
        total_loss = 0
        for epoch in range(num_epochs):
            rand_start_index = random.randint(0, test_batch_size)
            input_plus_label = test_batch[rand_start_index : min(rand_start_index + 50, test_batch_size)]
            input = input_plus_label[:-1]
            label = input_plus_label[-1]

            # Convert 2D row to batch
            input = self.convert_to_batch(input)

            # forward pass
            output = self.forward(input)

            # get loss output
            loss = loss_fn(output, label)
            if self.debug:
                log.debug(f"loss = {loss}")
            total_loss = total_loss + loss

        log.info(f"total_loss = {total_loss}")

    def convert_to_batch(self, non_batch):
        # Convert input to tensor of shape
        # (batch_dim, seq_dim, feature_dim)
        # seq_dim = length of sequence of timesteps of input
        batch = non_batch.reshape(-1, non_batch.shape[0], non_batch.shape[1])

        if self.debug:
            log.debug(f"batch.shape = {batch.shape}")

        return batch

    def saveM(self,name):
        torch.save(self.lstm.state_dict(),name)
        log.info(f"LSTM saved = {self.lstm}")
    
    def loadM(self,path):
        self.lstm.load_state_dict(torch.load(path))
        log.info(f"LSTM loaded = {self.lstm}")
