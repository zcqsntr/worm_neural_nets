import matplotlib.pyplot as plt
import numpy as np

def int_wrap(string):

    try:
        return int(string)
    except:
        return 0



def load_data(file):

    with open(file) as f:


        f.readline()
        f.readline()

        no_cond_no_odour = []
        no_cond_odour = []
        aversive_odour = []
        sex_odour = []

        for line in f:
            line = line.split(',')

            if not all([x == 0 for x in list(map(int_wrap, line[1:7]))]): no_cond_no_odour.append(list(map(int_wrap, line[1:7])))
            if not all([x == 0 for x in list(map(int_wrap, line[8:14]))]): no_cond_odour.append(list(map(int_wrap, line[8:14])))
            if not all([x == 0 for x in list(map(int_wrap, line[15:21]))]): aversive_odour.append(list(map(int_wrap, line[15:21])))
            if not all([x == 0 for x in list(map(int_wrap, line[22:28]))]): sex_odour.append(list(map(int_wrap, line[22:28])))

    return (no_cond_no_odour, no_cond_odour, aversive_odour, sex_odour)



if __name__ == '__main__':


    no_cond_no_odour, no_cond_odour, aversive_odour, sex_odour = load_data('./data/behaviourdatabysector_NT.csv')
    for d in [no_cond_no_odour, no_cond_odour, aversive_odour, sex_odour]:
        print(len(d))
    for d in [no_cond_odour]:
        plt.figure(figsize = (2,1.5))
        plt.violinplot(list(map(sum, d)))
        plt.ylabel('Score')
        plt.gca().set_xticks([])
        plt.xticks([])
        plt.savefig('data_plot.pdf')


    plt.show()

