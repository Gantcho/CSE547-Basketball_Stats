import numpy as np

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse


x_data = np.load('player_profiles_with_synergies.npy')
y_data = np.load('player_labels_gamescores.npy')

x_data_expanded = x_data
y_data_expanded = y_data
for teamorder in [0,1]:

  indeces = np.arange(12)
  if teamorder == 1:
    indeces = indeces[[6,7,8,9,10,11,0,1,2,3,4,5]]
  addy = []
  addx = []
  for firstplayer in range(6):
    print(firstplayer)
    if not (teamorder==0 and firstplayer == 0):
      topmins = indeces[0] # swap top mins on a team with the chosen index player
      indeces[0] = indeces[firstplayer]
      indeces[firstplayer] = topmins
      x_data_expanded = np.concatenate((x_data_expanded,x_data[:,indeces,:]))
      y_data_expanded = np.concatenate((y_data_expanded,y_data[:,indeces]))
      #y_data_expanded = np.concatenate((y_data_expanded,y_data[:,indeces,:]))

y_firstplayerpoints = y_data_expanded[:,1] # now this is first player gamescore #np.rint(y_data_expanded[:,0,0]*y_data_expanded[:,0,15])



X_stacked = np.reshape(x_data_expanded,(x_data_expanded.shape[0],x_data_expanded.shape[1]*x_data_expanded.shape[2]))
X_stacked.shape

train_x, test_x, train_y, test_y = train_test_split(X_stacked,y_firstplayerpoints)

dtrain_reg = xgb.DMatrix(train_x, train_y, enable_categorical=True)

dtest_reg = xgb.DMatrix(test_x, test_y, enable_categorical=True)

evals = [(dtest_reg, "validation"), (dtrain_reg, "train")]

params = {"objective": "reg:squarederror", 
          "tree_method": "hist",
  'learning_rate': 1,
    'lambda': 0.8,
    'alpha': 0.4,
    'max_depth': 6}

n=600

model = xgb.train(

   params=params,

   dtrain=dtrain_reg,

   num_boost_round=n,

   evals=evals,

   verbose_eval=5

)

preds = model.predict(dtest_reg)

rmse = mse(test_y, preds, squared=False)

print(f"RMSE of the base model: {rmse:.3f}")

model.save_model('6_depth_expanded_data.model')

model.dump_model('dump.raw.txt')