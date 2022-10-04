# as this is an multilabel classification Problem, so we need to predict all classes
# for MLC we have iterative_stratification library
# https://github.com/trent-b/iterative-stratification

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pandas as pd 

if __name__ == '__main__':
    df = pd.read_csv('input/train.csv')
    print(df.head())
    df.loc[:, 'kfold'] = -1 # create kfold col, and fill it with -1

    X = df.sample(frac = 1).reset_index(drop=True)

    # shuffling 
    df = df.sample(frac=1).reset_index(drop=True) # now it will keep index same

    # let's define X and y
    X = df.image_id.values
    y = df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values

    # let's initialize MSKF 
    mskf = MultilabelStratifiedKFold(n_splits=5)

    # let's get the train and validation Indices
    for fold, (trn_, val_) in enumerate(mskf.split(X, y)):
        print('Training Indices', trn_, 'Validation indices', val_)
        df.loc[val_, 'kfold'] = fold # will assign on of fold to each indices

    print(df.kfold.value_counts()) # will count no of indices 
    df.to_csv('input/train_folds.csv', index=False)



