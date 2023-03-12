import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse

model = xgb.Booster()
model.load_model("1000_epoch_6_depth_2_parralell_trees_xgboost.model")


xgb.plot_tree(model)
plt.show()

"""
#player_profile = =['MIN','MFGA','FGPCT','MFG3A','FG3PCT',
#                                    'MFTA','FTPCT','MOREB','MDREB','MREB',
#                                    'MAST','MSTL','MBLK','MTO','MPF','MPTS','MPLUSMIN','prediction']

#Synergies = [how much players game scoer improves when playing with cluster 1,
              how much players game scoer improves when playing with cluster 2, ... 
              how much players game scoer improves when playing with cluster 10,
              how much players game scoer improves when playing against cluster 1 ...]

player_profiles_with_sysnergy = player_profile + synergies concatenated horizontally

player_avg_features = ["<lambda>(MFGA)", "<lambda>(FGPCT)", 
                "<lambda>(MFG3A)", 
                 "<lambda>(FG3PCT)", 
                  "<lambda>(MFTA)", 
                 "<lambda>(FTPCT)", 
                 "<lambda>(MOREB)", 
                 "<lambda>(MDREB)", 
                 "<lambda>(MREB)", 
                 "<lambda>(MAST)",  
                  "<lambda>(MSTL)", 
                 "<lambda>(MBLK)", 
                 "<lambda>(MTO)", 
                 "<lambda>(MPF)", 
                 "<lambda>(MPTS)", 
                 "<lambda>(MPLUSMIN)" ]
                 Lmabda is the z score for eac hstat relative to the distribution for a particular season

player_labels = [label = ["MIN",'MFGA_IG','FG_PCT','MFG3A_IG','FG3_PCT', #the pct with the _ is in game, the one without is total season average!
#                                    'MFTA_IG','FT_PCT','MOREB_IG','MDREB_IG','MREB_IG',
#                                    'MAST_IG','MSTL_IG','MBLK_IG','MTO_IG','MPF_IG','MPTS_IG','MPLUSMIN_IG']

player_game_scores = just 1x12 vector of eachpl ayers gamescore (google the formula)

"""