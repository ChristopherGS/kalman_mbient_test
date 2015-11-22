import os
import pandas as pd
import numpy as np
import random
import math
import itertools

_basedir = os.path.abspath(os.path.dirname(__file__))
UPLOADS = 'data'
UPLOAD_FOLDER = os.path.join(_basedir, UPLOADS)


# NEW Training data from mbient sensor

filename = 'MBIENT_straightPunch_chris.csv'
filename2 = 'MBIENT_straightPunchtraining2_chris.csv'
filename3 = 'MBIENT_straightPunchtraining3_chris.csv'
filename4 = 'MBIENT_straightPunchtraining4_chris.csv'

non_punch_filename = 'MBIENT_trainingGeneralMovement.csv'
non_punch_filename2 = 'MBIENT_generalMotionTraining_chris.csv'
non_punch_filename3 = 'MBIENT_gmt3.csv'

hook_punch_filename = 'MBIENT_hookPunch_chris.csv'
hook_punch_filename2 = 'MBIENT_hookpunchTraining_chris.csv'
hook_punch_filename3 = 'MBIENT_hookpunchTraining2.csv'

TRAIN_DATA = UPLOAD_FOLDER + '/punch/' + filename
TRAIN_DATA4 = UPLOAD_FOLDER + '/punch/' + filename2
TRAIN_DATA5 = UPLOAD_FOLDER + '/punch/' + filename3
TRAIN_DATA10 = UPLOAD_FOLDER + '/punch/' + filename4
TRAIN_DATA6 = UPLOAD_FOLDER + '/punch/' + hook_punch_filename
TRAIN_DATA7 = UPLOAD_FOLDER + '/punch/' + hook_punch_filename2
TRAIN_DATA8 = UPLOAD_FOLDER + '/punch/' + hook_punch_filename3

TRAIN_DATA2 = UPLOAD_FOLDER + '/non_punch/' + non_punch_filename
TRAIN_DATA3 = UPLOAD_FOLDER + '/non_punch/' + non_punch_filename2
TRAIN_DATA9 = UPLOAD_FOLDER + '/non_punch/' + non_punch_filename3

TEST_DATA = UPLOAD_FOLDER + '/mix/4_punches_mix.csv'

columns = ['ACCELEROMETER_X',
            'ACCELEROMETER_Y',
            'ACCELEROMETER_Z',
            'timestamp',
            'state']


def clean_up(df):
    clean_df = pd.DataFrame(df.index.tolist(), columns=['ACCELEROMETER_X', 'ACCELEROMETER_Y', 'ACCELEROMETER_Z'])
    clean_df = clean_df.applymap(str)
    clean_df = clean_df.apply(lambda s: s.str.replace('(', ''))
    clean_df = clean_df.apply(lambda s: s.str.replace(')', ''))
    clean_df = clean_df.applymap(float)
    clean_df = clean_df.reindex(columns=columns)
    return clean_df

def set_straight_punch(the_df):
    the_df['state'] = 1
    punch_final_df = the_df
    return punch_final_df

def set_non_punch(my_df):
    my_df['state'] = 0
    non_punch_final_df = my_df
    return non_punch_final_df

def set_hook_punch(hook_df):
    hook_df['state'] = 2
    hook_punch_final_df = hook_df
    return hook_punch_final_df

df_punch = pd.read_csv(TRAIN_DATA, skiprows=[0], names=['initial'])
df_punch = clean_up(df_punch)
df_punch = set_straight_punch(df_punch)

df_punch2 = pd.read_csv(TRAIN_DATA4, skiprows=[0], names=['initial'])
df_punch2 = clean_up(df_punch2)
df_punch2 = set_straight_punch(df_punch2)

df_punch3 = pd.read_csv(TRAIN_DATA5, skiprows=[0], names=['initial'])
df_punch3 = clean_up(df_punch3)
df_punch3 = set_straight_punch(df_punch3)

df_punch4 = pd.read_csv(TRAIN_DATA10, skiprows=[0], names=['initial'])
df_punch4 = clean_up(df_punch4)
df_punch4 = set_straight_punch(df_punch4)

df_hook_punch = pd.read_csv(TRAIN_DATA6, skiprows=[0], names=['initial'])
df_hook_punch = clean_up(df_hook_punch)
df_hook_punch = set_hook_punch(df_hook_punch)

df_hook_punch2 = pd.read_csv(TRAIN_DATA7, skiprows=[0], names=['initial'])
df_hook_punch2 = clean_up(df_hook_punch2)
df_hook_punch2 = set_hook_punch(df_hook_punch2)

df_hook_punch3 = pd.read_csv(TRAIN_DATA8, skiprows=[0], names=['initial'])
df_hook_punch3 = clean_up(df_hook_punch3)
df_hook_punch3 = set_hook_punch(df_hook_punch3)

df_non_punch2 = pd.read_csv(TRAIN_DATA2, skiprows=[0], names=['initial'])
df_non_punch2 = clean_up(df_non_punch2)
df_non_punch2 = set_non_punch(df_non_punch2)

df_non_punch3 = pd.read_csv(TRAIN_DATA9, skiprows=[0], names=['initial'])
df_non_punch3 = clean_up(df_non_punch3)
df_non_punch3 = set_non_punch(df_non_punch3)

df_non_punch = pd.read_csv(TRAIN_DATA3, skiprows=[0], names=['initial'])
df_non_punch = clean_up(df_non_punch)
df_non_punch = set_non_punch(df_non_punch)

df_train = pd.concat([df_punch, df_punch2, df_punch3, df_punch4, df_hook_punch, df_hook_punch2, df_hook_punch3, df_non_punch2, df_non_punch3, df_non_punch], ignore_index=True)

def load_data():
	kalman_df = pd.read_csv(TEST_DATA, skiprows=[0], names=['initial'])
	kalman_df = clean_up(kalman_df)
	return kalman_df
