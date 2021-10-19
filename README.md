# Monte Carlo

This is a repo for Monte Carlo method. Some of guessing method has been rewritten and we can get more information from
the result.

## Update 2021/10/19

> Intermediate results could affect the final results

Now we can use `--based-on-prior-guesses` to calculate guess numbers of the passwords.
For example, if we have two rounds, and the first round generate 10,000 guesses, then 
the guesses generated during the second round will start from 10,001.

## Update 2021/10/09

> Subword level Backoff, using secondary training file

`backwords_secondary_main.py` will automatically crack the passwords in the given testing dataset and retrain the model
based on the cracked passwords (as a secondary training file). Users could specify the number of cracked passwords when
retraining the model according to the option `--secondary-sample`. Besides, by specifying the option of `-g`, users
could specify how many rounds the model will be retrained.
```shell
python backwords_secondary_main.py \
    -i 'training file' \
    -t 'testing file' \
    -s 'save intermediate files in this folder' \
    -g 'one or more guess numbers to crack passwords in the testing file' \
    --start4word 'the index of the first "chunk" or "character" for each password' \
    --skip4word 'start4word + skip4word = the next index of the "chunk" or "character"' \
    --splitter 'use "empty" for character level backoff, "space" for " ", "tab" for "\t"' \
    --secondary-sample 'sample some, instead of all, cracked passwords to retrain the model'
```
Note that you could also generate guesses based on Monte Carlo as follows:

```shell
python backwords_secondary_main.py \
    -i 'training file' \
    -t 'testing file' \
    -s 'save intermediate files in this folder' \
    --using-samples 'the number of iterations to sample passwords' \
    --start4word 'the index of the first "chunk" or "character" for each password' \
    --skip4word 'start4word + skip4word = the next index of the "chunk" or "character"' \
    --splitter 'use "empty" for character level backoff, "space" for " ", "tab" for "\t"' \
    --secondary-sample 'sample some, instead of all, cracked passwords to retrain the model'
```
In this case, the `-g` flag is ignored.

-------

`backwords_secondary_trainer.py` could train a backword model based on a given backword model. For example, given
training file T and train a model $M_{T}$, then given training file P and train a model $M_{P,M_T}$. Note that $M_
{P,M_T}$ is equal to $M_{P \union T}$.

```shell
python backwords_secondary_trainer.py -t 'training file' \
    -s 'save the trained model here' \
    -m 'provide a trained model and use the training file as the secondary training file to further update the provided model'
```

`backwords_secondary_simulator.py` reads the model obtained by `backwords_secondary_trainer.py`, then evaluates the
passwords in the given testing dataset.

```shell
python backwords_secondary_simulator.py -m 'trained model' \
    -t 'testing file' \
    -s 'save the results of the evaluation here'
```