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
