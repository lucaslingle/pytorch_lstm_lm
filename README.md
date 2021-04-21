# pytorch_lstm_lm

This is a pytorch implementation of an LSTM-based language model. The dependencies are pytorch 1.8.1 and torchtext 0.9.1.

You can train the model by running
```
python main.py --mode=train
```

After training, you can sample from the model by running 
```
python main.py --mode=generate
```

The samples will be written to a file ```samples.txt```. In this git repo, samples from a pretrained LSTM are provided in this file.
After you clone the repo, delete the file to start from scratch. 
