import os

import pandas as pd

if __name__ == '__main__':

    # read and process the data from the prior experiment
    res0 = pd.read_csv('../data/old_data.csv')
    res0_nobs = 3
    res0_obs_nums = range(1, res0_nobs + 1)
    res0 = res0[['patch_desc'] + ['obs_%d' % i for i in res0_obs_nums]]
    res0_cols = ['res0_%02d' % i for i in res0_obs_nums]
    res0.columns = ['patch'] + res0_cols
    res0.index = res0.patch
    res0 = res0[res0_cols]

    # the new experiment didn't follow our color naming guidelines exactly, so we have to fix it
    col_map = {c: c for c in 'BLUE,BROWN,GREEN,MAGENTA,NEUTRAL,ORANGE,PURPLE,RED,YELLOW'.split(',')}
    col_map['BLUE'] = 'BLUECYAN'
    col_map['PURPLE'] = 'PURPLEVIOLET'

    # read and process the data from the first batch in the latest experiment
    res1 = pd.read_csv('../data/results1.csv')
    res1_nobs = 17
    res1_obs_nums = range(1, res1_nobs + 1)
    res1 = res1[['color_patch'] + [str(i) for i in res1_obs_nums]]
    res1_cols = ['res1_%02d' % i for i in res1_obs_nums]
    res1.columns = ['patch'] + res1_cols
    res1.index = res1.patch
    res1 = res1[res1_cols]
    res1 = res1.apply(lambda x: x.str.upper())
    res1 = res1.apply(lambda x: x.map(col_map))

    # read and process the data from the second batch in the latest experiment
    res2 = pd.read_csv('../data/results2.csv')
    res2_nobs = 16
    res2_obs_nums = range(1, res2_nobs + 1)
    res2 = res2[['color_patch'] + [str(i) for i in range(1, 17)]]
    res2_cols = ['res2_%02d' % i for i in range(1, 17)]
    res2.columns = ['patch'] + res2_cols
    res2.index = res2.patch
    res2 = res2[['res2_%02d' % i for i in range(1, 17)]]
    res2 = res2.apply(lambda x: x.str.upper())
    res2 = res2.apply(lambda x: x.map(col_map))

    # join all the data together
    # we use res2 as the base because it has all patches from the latest pantone set
    # we allow it to remove rows from res0 that are not in the new set
    # res1 should be a strict subset of res2
    # the we put the columns back in order
    all = res2.join(res1)
    all = all.join(res0)
    all = all[res0_cols + res1_cols + res2_cols]

    all.to_csv('../data/all_data.csv')
    all.stack().to_csv('../data/all_stacked.csv')
