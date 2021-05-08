import pickle

from bpeX.modelreader import read_bpe
from bpe_simulator import BpePcfgSim


def wrapper(model_path: str):
    bpePcfg = read_bpe(model_path)
    pickle.dump(bpePcfg, open("./test.pickle", 'wb'))
    slim = pickle.load(open('./test.pickle', 'rb'))


if __name__ == '__main__':
    wrapper("/home/cw/Documents/tmp/xmbpe/model_E4.5")
