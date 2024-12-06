import os
import argparse
import shutil
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', default='tokyo3')
    args = parser.parse_args()
    os.makedirs(f'./data/{args.city}/', exist_ok=True)
    shutil.copy(f'../../DeepTrip/Trip/origin_data/poi-{args.city}.csv', f'./data/{args.city}/poi-{args.city}.csv')

    df_traj = pd.read_csv(f'../../DeepTrip/Trip/origin_data/traj-{args.city}.csv')
    df_traj['userID'] = df_traj['userID'].astype(str)
    df_traj['userID'] = 'user_' + df_traj['userID']
    df_traj.to_csv(f'./data/{args.city}/traj-{args.city}.csv')
