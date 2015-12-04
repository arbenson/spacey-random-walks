import seaborn as sns
sns.set(style='white')
import matplotlib.pyplot as plt

def GetSeqs(seq_file):
    seqs = []
    with open(seq_file) as f:
        for line in f:
            seqs.append([int(x) for x in line.strip().split(',')])
    return seqs

if __name__ == '__main__':
    seqs = GetSeqs('processed_data/manhattan-year-seqs.txt')
    lengths = [len(seq) for seq in seqs]
    sns.distplot(lengths, kde=False, color='b')
    fsz = 16
    plt.xlabel('Trajectory length', fontsize=fsz)
    plt.ylabel('Number of occurrences', fontsize=fsz)
    plt.title('Distribution of taxi trajectory lengths', fontsize=fsz)
    plt.tick_params(axis='both', which='major', labelsize=fsz - 2)
    plt.tick_params(axis='both', which='minor', labelsize=fsz - 2)
    sns.despine()
    plt.savefig('trajectory-dist.pdf', bbox_inches='tight')
    plt.show()
