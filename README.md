# Monte Carlo

This is a repo for Monte Carlo method.
Some of guessing method has been rewritten and we can get more information
from the result.


## Update 2021/10/09 

> Subword level Backoff, using secondary training file

`backwords_secondary_trainer.py` could train a backword model based on a given backword model.
For example, given training file T and train a model $M_{T}$, 
then given training file P and train a model $M_{P,M_T}$.
Note that $M_{P,M_T}$ is equal to $M_{P \union T}$.
```shell
python backwords_secondary_trainer.py -t 'training file' \
    -s 'save the trained model here' \
    -m 'provide a trained model and use the training file as the secondary training file to further update the provided model'
```

`backwords_secondary_simulator.py` reads the model obtained by `backwords_secondary_trainer.py`,
then evaluates the passwords in the given testing dataset. 

```shell
python backwords_secondary_simulator.py -m 'trained model' \
    -t 'testing file' \
    -s 'save the results of the evaluation here'
```