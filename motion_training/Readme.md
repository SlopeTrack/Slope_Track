# Training the Motion Model 

Install mamba-ssm. Check the official codebase for more details. 
~~~
pip install mamba-ssm[causal-conv1d]
~~~

1. Get pickle file of train and validation sets.

~~~
python create_pickle.py
~~~

Run
~~~
python train.py --option 4 --min-len 60 --max-len 60 --batch-size 128 --epochs 700 --target-len 60 --model mamba --train
~~~

We used 1 gpu to train the model.
