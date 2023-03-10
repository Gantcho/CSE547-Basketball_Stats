from pdb import set_trace as keyboard

import os
import numpy as np
import torch
import torch.nn.functional as F

#########################################################################
#          train_cls_standard
#########################################################################
def train_cls_standard(mdl, opt, criterion,
                       train_loader, test_loader, par):
    n_epochs = par['n_epochs']
    batch_losses = []

    mdl.train()
    for epoch in np.arange(n_epochs):
        mdl.train() # mode may have been reset to .eval() when making predictions
                    # on the test set

        # capture true and estimated values for subsequent analysis
        y_gt_all = []
        y_est_all = []

        for batch_idx, batch_data in enumerate(train_loader):
            x, y = batch_data
            y_gt = y.to(par['device'])
            x_  = x.to(par['device'])

            y_est = mdl(x_)
            loss = criterion(y_est, y_gt)
            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_losses.append(loss.item())


        eval_on_test_loader_this_epoch = True
        if eval_on_test_loader_this_epoch:
            preds = predict_cls_standard(mdl, criterion, test_loader, par)
                        
        print('epoch {:04d} train loss (last batch): {} '.format(epoch, batch_losses[-1]))



#########################################################################
#          predict_cls_standard
#########################################################################
def predict_cls_standard(mdl, criterion, dloader, par):
    """
    Apply "mdl" to data provided by "dloader".  Typically, dloader will be a test_loader set up
    in the client code

    par['device'] specifies the device
    """

    """
    :NOTE: if we were doing MC ensembles with dropout, we'd set
    mdl.train() or define a command-line argument
    --dropout_at_test=True
    """

    mdl.eval()
    y_logits = []
    x_in = []
    y_gt = []    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dloader): 
            x_b, y_b = batch_data
            y_b = y_b.to(par['device'])
            x_b  = x_b.to(par['device'])
            y_logits_b = mdl(x_b)
            y_logits.append(y_logits_b)
            x_in.append(x_b)
            y_gt.append(y_b)
            
        y_logits = torch.cat(y_logits)
        x_in = torch.cat(x_in)
        y_gt = torch.cat(y_gt)        

    #x_in, y_gt, y_logits = map(lambda t_ : t_.cpu().detach().numpy(),
    #                           (x_in, y_gt, y_logits))

    test_loss = criterion(y_logits, y_gt)
    print(f'Test Loss: {test_loss}')

    return {'x_in':x_in,
            'y_gt':y_gt,
            'y_logits':y_logits}
