import matplotlib.pyplot as plt
import mglearn


def execute():
    S = mglearn.datasets.make_signals()
    fig = plt.figure(figsize=(6, 1))
    plt.plot(S, '-')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    fig.savefig('signals/signals.png')
