from pdb import set_trace as keyboard
import xgboost as xgb

import argparse, sys, os, io, contextlib
import cvd_mlp
import numpy as np
import torch
import torch.optim as optim
import train_util
import math
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from torch.utils.data import DataLoader, TensorDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))

def train_model(par):
    mdl = cvd_mlp.cvd_mlp(par['n_hidden_layers'], par['w_hidden_layers'], par['p_dropout'])
    mdl = mdl.to(device)
    opt = optim.Adam(mdl.parameters(), lr=par['lr'], weight_decay=par['weight_decay'])
    criterion = torch.nn.MSELoss()

    if 'load_model_from' in par and par['load_model_from'] is not None:
        model_path = par['load_model_from']
        model_ = torch.load(model_path)
        mdl.load_state_dict(model_['mdl__state_dict'])
        opt.load_state_dict(model_['mdl__state_dict'])
    
    # SET TRAIN AND TEST FILES AS YOU WISH
    X_data = np.load('player_avg_features.npy')
    y_data = np.load('player_game_scores.npy')
    baseline = np.load('player_profiles.npy')
    X_data = np.float32(X_data)
    y_data = np.float32(y_data)

    train_X, test_X, train_y, test_y = train_test_split(X_data, y_data, test_size = 0.2)
    train_X = train_X.reshape((train_X.shape[0], train_X.shape[1]*train_X.shape[2]))
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[1]*test_X.shape[2]))

    train_dataset = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
    test_dataset = TensorDataset(torch.from_numpy(test_X), torch.from_numpy(test_y))

    batch_sz = par['batch_sz']
    train_loader = DataLoader(train_dataset,
                             batch_size=batch_sz,
                             shuffle=True,
                             num_workers=4,
                             pin_memory=False)

    test_loader = DataLoader(test_dataset,
                            batch_size=batch_sz,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=False)

    par.update({'device':device})
    # train model
    train_util.train_cls_standard(mdl, opt, criterion, train_loader, test_loader, par)

    # Save model
    model_path = os.path.join(par['output_dir'], 'mdl.pt')
    torch.save({'net0__state_dict' : mdl.state_dict(),
                'opt0__state_dict' : opt.state_dict()},
            model_path)

    # Get baseline predictions
    totals = baseline * np.expand_dims(baseline[:,:,0], axis=2) # Multiply all stats by num mins played
    baseline_preds = np.zeros((baseline.shape[0], baseline.shape[1]))
    for i, game in enumerate(totals):
        for j, row in enumerate(game):
            baseline_preds[i][j]=row[15]+0.4*row[2]*row[1]/math.sqrt(row[0])-0.7*row[1]-0.4*(row[5]-row[6]*row[5]/math.sqrt(row[0]))+0.7*row[7]+0.3*row[8]+row[11]+0.7*row[10]+0.7*row[12]-0.4*row[14]-row[13]
    
    print(f"BASELINE MSE: {criterion(torch.from_numpy(baseline_preds), torch.from_numpy(y_data))}")

def train_xgb(par):
    X_data = np.load('player_avg_features.npy')
    y_data = np.load('player_game_scores.npy')
    # baseline = np.load('player_profiles.npy')
    X_data = np.float32(X_data)
    y_data = np.float32(y_data)
    criterion = torch.nn.MSELoss()

    train_X, test_X, train_y, test_y = train_test_split(X_data, y_data, test_size = 0.2)
    train_X = train_X.reshape((train_X.shape[0], train_X.shape[1]*train_X.shape[2]))
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[1]*test_X.shape[2]))

    par.update({'device':device})
    
    multioutputregressor = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror')).fit(train_X, train_y)
    preds = multioutputregressor.predict(test_X)
    print(f"MSE: {criterion(torch.from_numpy(preds), torch.from_numpy(test_y))}")


def main(par):
    if par['model_type'] == 'MLP':
        train_model(par)
    elif par['model_type'] == 'XGB':
        train_xgb(par) 
    else:
        print("UNSUPPORTED MODEL TYPE")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True, choices={'MLP', 'XGB'})
    parser.add_argument('--batch_sz', type=int, default=128)
    parser.add_argument('--n_epochs', type=int,  default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--p_dropout', type=float, default=0.1,
                        help='used for either train or eval or both')     
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--load_model_from', default=None, help='/PATH/TO/FILE.pt')
    parser.add_argument('--n_hidden_layers', type=int, default =5)
    parser.add_argument('--w_hidden_layers', type=int, default = 100)
    parser.add_argument('--weight_decay', type=float, default=0)
    
    par__main=vars(parser.parse_args(sys.argv[1:]))
    main(par__main)
