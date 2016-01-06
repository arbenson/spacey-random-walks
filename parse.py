import numpy as np

def parse_file(filename):
    def last_float(st):
        return float(st.split()[-1])

    with open(filename) as f:
        for line in f:
            if line.find('Spacey (true):') != -1:
                true_rmse = last_float(line)
            elif line.find('Spacey (estimated):') != -1:
                srw_rmse = last_float(line)
            elif line.find('Second-order:') != -1:
                somc_rmse = last_float(line)
            elif line.find('First-order:') != -1:
                fomc_rmse = last_float(line)


        return true_rmse, srw_rmse, somc_rmse, fomc_rmse

def collect(nsim):
    all_true_rmse = []
    all_srw_rmse = []
    all_somc_rmse = []
    all_fomc_rmse = []
    for k in xrange(1, nsim + 1):
        file = 'processed_data/synthetic/uniform/P-4-100-200-data.%d.txt' % k
        true_rmse, srw_rmse, somc_rmse, fomc_rmse = parse_file(file)
        all_true_rmse.append(true_rmse)
        all_srw_rmse.append(srw_rmse)
        all_somc_rmse.append(somc_rmse)
        all_fomc_rmse.append(fomc_rmse)


    return all_true_rmse, all_srw_rmse, all_somc_rmse, all_fomc_rmse

if __name__ == '__main__':
    all_true_rmse, all_srw_rmse, all_somc_rmse, all_fomc_rmse = collect(20)
    print ' & %0.3f$\pm$%0.3f & %0.3f$\pm$%0.3f & %0.3f$\pm$%0.3f & %0.3f$\pm$%0.3f' % (
        np.mean(all_true_rmse), np.std(all_true_rmse),
        np.mean(all_srw_rmse), np.std(all_srw_rmse),
        np.mean(all_somc_rmse), np.std(all_somc_rmse),
        np.mean(all_fomc_rmse), np.std(all_fomc_rmse)
        )
