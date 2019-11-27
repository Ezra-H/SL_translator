# component

## feature extraction

- P3D19: 3D-ResNet19  8 length of clip with 4 stride.
 you can also choose P3D64, P3D131, P3D199

## encode

- BLSTM

## decode

- LSTM decoder with attention

- CTC decoder

# work
## unfinish
 - CTC decoder
 - CTC optimizer method

## finished
 - data_loader
 - encoder
 - batch is limited in 1
 - decoder
 - value
 - loss
 - the main model body
 - options
 
 
## problem
 - High memory usage of GPU.

 
 # usage

 python my_train.py   
 
 
