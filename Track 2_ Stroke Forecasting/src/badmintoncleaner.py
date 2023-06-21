from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None


PAD = 0


class BadmintonDataset(Dataset):
    def __init__(self, matches, config):
        super().__init__()
        self.max_ball_round = config['max_ball_round']
        group = matches[['rally_id', 'ball_round', 'type', 'landing_x', 'landing_y', 'player', 'set']].groupby('rally_id').apply(lambda r: (r['ball_round'].values, r['type'].values, r['landing_x'].values, r['landing_y'].values, r['player'].values, r['set'].values))

        self.sequences, self.rally_ids = {}, []
        for i, rally_id in enumerate(group.index):
            ball_round, shot_type, landing_x, landing_y, player, sets = group[rally_id]
            #filter out shot_type less than encoding length
            if len(shot_type) <= config['encode_length']:
                continue
            self.sequences[rally_id] = (ball_round, shot_type, landing_x, landing_y, player, sets)
            self.rally_ids.append(rally_id)

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        rally_id = self.rally_ids[index]
        ball_round, shot_type, landing_x, landing_y, player, sets = self.sequences[rally_id]

        pad_input_shot = np.full(self.max_ball_round, fill_value=PAD, dtype=int)
        pad_input_x = np.full(self.max_ball_round, fill_value=PAD, dtype=float)
        pad_input_y = np.full(self.max_ball_round, fill_value=PAD, dtype=float)
        pad_input_player = np.full(self.max_ball_round, fill_value=PAD, dtype=int)
        pad_output_shot = np.full(self.max_ball_round, fill_value=PAD, dtype=int)
        pad_output_x = np.full(self.max_ball_round, fill_value=PAD, dtype=float)
        pad_output_y = np.full(self.max_ball_round, fill_value=PAD, dtype=float)
        pad_output_player = np.full(self.max_ball_round, fill_value=PAD, dtype=int)

        # pad or trim based on the max ball round
        if len(ball_round) > self.max_ball_round:
            rally_len = self.max_ball_round

            pad_input_shot[:] = shot_type[0:-1:1][:rally_len]                                   # 0, 1, ..., max_ball_round-1
            pad_input_x[:] = landing_x[0:-1:1][:rally_len]
            pad_input_y[:] = landing_y[0:-1:1][:rally_len]
            pad_input_player[:] = player[0:-1:1][:rally_len]
            pad_output_shot[:] = shot_type[1::1][:rally_len]                                    # 1, 2, ..., max_ball_round
            pad_output_x[:] = landing_x[1::1][:rally_len]
            pad_output_y[:] = landing_y[1::1][:rally_len]
            pad_output_player[:] = player[1::1][:rally_len]
        else:
            rally_len = len(ball_round) - 1                                                     # 0 ~ (n-2)
            
            pad_input_shot[:rally_len] = shot_type[0:-1:1]                                      # 0, 1, ..., n-1
            pad_input_x[:rally_len] = landing_x[0:-1:1]
            pad_input_y[:rally_len] = landing_y[0:-1:1]
            pad_input_player[:rally_len] = player[0:-1:1]
            pad_output_shot[:rally_len] = shot_type[1::1]                                       # 1, 2, ..., n
            pad_output_x[:rally_len] = landing_x[1::1]
            pad_output_y[:rally_len] = landing_y[1::1]
            pad_output_player[:rally_len] = player[1::1]

        return (pad_input_shot, pad_input_x, pad_input_y, pad_input_player,
                pad_output_shot, pad_output_x, pad_output_y, pad_output_player,
                rally_len, sets[0])


def prepare_dataset(config):
    train_matches = pd.read_csv(f"{config['data_folder']}train.csv")
    val_matches = pd.read_csv(f"{config['data_folder']}val_given.csv")
    test_matches = pd.read_csv(f"{config['data_folder']}test_given.csv")

    # #--Additional steps to split for val--
    # num_val_rally = len(val_matches['rally_id'].unique())
    # train_rally_ids = train_matches['rally_id'].unique()
    # val_rally_ids = train_rally_ids[-num_val_rally:]   #take last num_val_rally
    # # train_rally_ids = np.setdiff1d(train_rally_ids, val_rally_ids)
    # # train_matches = train_matches[train_matches['rally_id'].isin(train_rally_ids)].reset_index(drop=True)
    # val_matches = train_matches[train_matches['rally_id'].isin(val_rally_ids)].reset_index(drop=True) #use train matches for val for now

    # #remove rallies with less than 10 shots
    # rally_counts = val_matches['rally_id'].value_counts()
    # rally_ids_to_delete = rally_counts[rally_counts < 10].index
    # val_matches = val_matches[~val_matches['rally_id'].isin(rally_ids_to_delete)].reset_index(drop=True)
    # #--End Additional steps to split for val--

    # encode shot type
    codes_type, uniques_type = pd.factorize(train_matches['type'])
    train_matches['type'] = codes_type + 1                                # Reserve code 0 for paddings
    val_matches['type'] = val_matches['type'].apply(lambda x: list(uniques_type).index(x)+1)
    test_matches['type'] = test_matches['type'].apply(lambda x: list(uniques_type).index(x)+1)
    config['uniques_type'] = uniques_type.to_list()
    config['shot_num'] = len(uniques_type) + 1                            # Add padding

    # encode player
    train_matches['player'] = train_matches['player'].apply(lambda x: x+1)
    val_matches['player'] = val_matches['player'].apply(lambda x: x+1)
    test_matches['player'] = test_matches['player'].apply(lambda x: x+1)
    config['player_num'] = 35 + 1     # Add padding

    train_dataset = BadmintonDataset(train_matches, config)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    val_dataset = BadmintonDataset(val_matches, config)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    test_dataset = BadmintonDataset(test_matches, config)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    return config, train_dataset, train_dataloader, val_dataloader, test_dataloader, train_matches, val_matches, test_matches