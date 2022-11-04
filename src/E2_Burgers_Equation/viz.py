'''
===
Distributed by: Computational Science Initiative, Brookhaven National Laboratory (MIT Liscense)
- Associated publication:
url: 
doi: 
github: 
=====
'''
import os
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colorbar as mcbar

from utils import toNumpy, toTuple


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def vizPred(args, model, test_loader, tstep):
    model.eval()
    mb_size = int(len(test_loader.dataset)/len(test_loader))
    u_out = torch.zeros(mb_size, tstep+1, 2, args.nel, args.nel).to(args.device)

    for bidx, (input0, uTarget0) in enumerate(test_loader):
        dims = torch.ones(len(input0.shape))
        dims[1] = (args.input_channels/2)
        input = input0.repeat(toTuple(toNumpy(dims).astype(int))).to(args.device)
        
        if(bidx == 0):
            u_out[:,0,:,:,:] = input0
            u_target = uTarget0

        for t_idx in range(tstep):
            uPred = model(input[:,-args.input_channels:,:,:])
            if(bidx == 0):
                u_out[:,t_idx+1,:,:,:] = uPred
            
            input = input[:,-int(args.input_channels-2):,:].detach()
            input0 = uPred.detach()
            input = torch.cat([input,  input0], dim=1)
        break
        
    return u_out, u_target


def vizAE(uPred, uTarget, ntrain, expName, taskName):
    idxs = [2*idx for idx in range(101)]
    uPred = uPred.to('cpu')
    uTarget = uTarget.to('cpu')
    uPred = uPred[:,idxs,:,:,:]

    uTarget = toNumpy(uTarget)
    uPred = toNumpy(uPred)

    err = np.abs(uTarget-uPred)
    err = np.sum(err, axis=(1,2,3,4))
    errLst = []
    for idx, item in enumerate(err):
        errLst.append((idx, item))
    errLst = sorted(errLst, key=lambda x:x[1])
    
    levels = 30
    cmap = plt.get_cmap('RdBu_r')
    mpl.rcParams.update({'font.family': 'serif', 'font.size': 37})

    num = 5
    idxPlot = []
    for idx in range(num):
        idxPlot.append(errLst[idx][0])
    
    for idx in range(1):
        dirSave = 'figures/{}/{}/ntrain_{}/'.format(taskName, expName, ntrain)
        mkdir(dirSave)

        x_component_Pred = uPred[idx,:,0,:,:]
        y_component_Pred = uPred[idx,:,1,:,:]
        x_component_Tar = uTarget[idx,:,0,:,:]
        y_component_Tar = uTarget[idx,:,1,:,:]
        
        x_Error = np.abs(x_component_Pred - x_component_Tar)
        y_Error = np.abs(y_component_Pred - y_component_Tar)

        x_component_Pred_save = x_component_Pred[[25, 75],:,:]
        y_component_Pred_save = y_component_Pred[[25, 75],:,:]
        x_component_Tar_save = x_component_Tar[[25, 75],:,:]
        y_component_Tar_save = y_component_Tar[[25, 75],:,:]

        np.savez(os.path.join(dirSave, 'inter_extra'), xPred=x_component_Pred_save, yPred=y_component_Pred_save, xTar=x_component_Tar_save, yTar=y_component_Tar_save)

    for idx in idxPlot:
        uSave = 'figures/{}/{}/ntrain_{}/{}/u'.format(taskName, expName, ntrain, idx)
        vSave = 'figures/{}/{}/ntrain_{}/{}/v'.format(taskName, expName, ntrain, idx)
        mkdir(uSave)
        mkdir(vSave)

        x_component_Pred = uPred[idx,:,0,:,:]
        y_component_Pred = uPred[idx,:,1,:,:]
        x_component_Tar = uTarget[idx,:,0,:,:]
        y_component_Tar = uTarget[idx,:,1,:,:]

        for idxTime in range(x_component_Pred.shape[0]):
            if not (idxTime%10):
                minVal = min(np.min(x_component_Pred[idxTime,:,:]), np.min(x_component_Tar[idxTime,:,:]))
                maxVal = max(np.max(x_component_Pred[idxTime,:,:]), np.max(x_component_Tar[idxTime,:,:]))
                norm = mpl.colors.Normalize(vmin=minVal, vmax=maxVal)
                
                plt.close("all")
                fig, axs = plt.subplots(1, 3, figsize=(36, 15))
                for col in range(3):
                    ax = axs[col]
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if col == 0:
                        pcm = ax.contourf(x_component_Tar[idxTime,:,:], levels=levels, cmap=cmap, norm=norm)
                        ax.set_title('Target at t = {:01.1f}'.format(idxTime*0.01))
                    elif col == 1:
                        pcm = ax.contourf(x_component_Pred[idxTime,:,:], levels=levels, cmap=cmap, norm=norm)
                        ax.set_title('Prediction at t = {:01.1f}'.format(idxTime*0.01))
                    elif col == 2:
                        pcm = ax.contourf(x_Error[idxTime,:,:], levels=levels, cmap=cmap)
                        ax.set_title('Error at t = {:01.1f}'.format(idxTime*0.01))

                    if col == 1:
                        cax, cbar_kwds = mcbar.make_axes(axs[:2], shrink=0.95, aspect=37, orientation='horizontal')
                        ticks = np.linspace(0, 1, 7)
                        tickLabels = np.linspace(minVal, maxVal, 7)
                        tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
                        cbar = mcbar.ColorbarBase(cax, cmap=plt.get_cmap(cmap), orientation='horizontal', ticks=ticks)
                        cbar.set_ticklabels(tickLabels)

                    if col == 2:
                        cax, cbar_kwds = mcbar.make_axes(axs[2], shrink=0.95, aspect=17, orientation='horizontal')
                        ticks = np.linspace(0, 1, 4)
                        tickLabels = np.linspace(np.min(x_Error[idxTime,:,:]), np.max(x_Error[idxTime,:,:]), 4)
                        tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
                        cbar = mcbar.ColorbarBase(cax, cmap=plt.get_cmap(cmap), orientation='horizontal', ticks=ticks)
                        cbar.set_ticklabels(tickLabels)

                plt.savefig(uSave+'/time_{}.png'.format(idxTime), bbox_inches='tight') 

                minVal = min(np.min(y_component_Pred[idxTime,:,:]), np.min(y_component_Tar[idxTime,:,:]))
                maxVal = max(np.max(y_component_Pred[idxTime,:,:]), np.max(y_component_Tar[idxTime,:,:]))
                norm = mpl.colors.Normalize(vmin=minVal, vmax=maxVal)

                plt.close("all")
                fig, axs = plt.subplots(1, 3, figsize=(36, 15))
                for col in range(3):
                    ax = axs[col]
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if col == 0:
                        pcm = ax.contourf(y_component_Tar[idxTime,:,:], levels=levels, cmap=cmap, norm=norm)
                        ax.set_title('Target at t = {:01.1f}'.format(idxTime*0.01))
                    elif col == 1:
                        pcm = ax.contourf(y_component_Pred[idxTime,:,:], levels=levels, cmap=cmap, norm=norm)
                        ax.set_title('Prediction at t = {:01.1f}'.format(idxTime*0.01))
                    elif col == 2:
                        pcm = ax.contourf(y_Error[idxTime,:,:], levels=levels, cmap=cmap)
                        ax.set_title('Error at t = {:01.1f}'.format(idxTime*0.01))

                    if col == 1:
                        cax, cbar_kwds = mcbar.make_axes(axs[:2], shrink=0.95, aspect=37, orientation='horizontal')
                        ticks = np.linspace(0, 1, 7)
                        tickLabels = np.linspace(minVal, maxVal, 7)
                        tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
                        cbar = mcbar.ColorbarBase(cax, cmap=plt.get_cmap(cmap), orientation='horizontal', ticks=ticks)
                        cbar.set_ticklabels(tickLabels)

                    if col == 2:
                        cax, cbar_kwds = mcbar.make_axes(axs[2], shrink=0.95, aspect=17, orientation='horizontal')
                        ticks = np.linspace(0, 1, 4)
                        tickLabels = np.linspace(np.min(y_Error[idxTime,:,:]), np.max(y_Error[idxTime,:,:]), 4)
                        tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
                        cbar = mcbar.ColorbarBase(cax, cmap=plt.get_cmap(cmap), orientation='horizontal', ticks=ticks)
                        cbar.set_ticklabels(tickLabels)
                plt.savefig(vSave+'/time_{}.png'.format(idxTime), bbox_inches='tight')
        
        print('Plotting of sample {} is completed'.format(idx))


def postMSE(uPred, uTarget):
    idxs = [2*idx for idx in range(101)]
    uPred = uPred.to('cpu')
    uTarget = uTarget.to('cpu')
    uPred = uPred[:,idxs,:,:,:]

    mseLst = []
    for idxBatch in range(uPred.shape[0]):
        y_pred = uPred[idxBatch]
        y_true = uTarget[idxBatch]
        temp = []
        for idxTime in range(y_pred.shape[0]):
            mse = torch.mean((y_pred[idxTime] - y_true[idxTime]) ** 2).to('cpu')
            temp.append(toNumpy(mse))
        mseLst.append(temp)
    
    mseLst = np.transpose(np.asarray(mseLst))
    return mseLst


def postTable(uPred, uTarget, ntrain, expName, taskName):
    idxs = [2*idx for idx in range(101)]
    uPred = uPred.to('cpu')
    uTarget = uTarget.to('cpu')
    uPred = uPred[:,idxs,:,:,:]
    uPred = toNumpy(uPred)
    uTarget = toNumpy(uTarget)
    
    errLst = []
    R2Lst = []
    for idxBatch in range(uPred.shape[0]):
        y_pred = uPred[idxBatch]
        y_true = uTarget[idxBatch]
        
        temp = []
        for idxTime in range(1, y_pred.shape[0]):
            errL2 = 0.
            for i in range(2):
                actual_value = y_true[idxTime, i]
                predicted_value = y_pred[idxTime, i]
                numerator = np.sqrt(np.sum(np.power((actual_value-predicted_value),2)))
                denominator = np.sqrt(np.sum(np.power(actual_value,2)))
                errL2 += (numerator/denominator)
            errL2 /= 2
            temp.append(errL2)
        errLst.append(temp)
        
        temp = []
        for idxTime in range(1, y_pred.shape[0]):
            r2_score = 0.
            for i in range(2):
                actual_value = y_true[idxTime, i]
                predicted_value = y_pred[idxTime, i]
                sse  = np.square( predicted_value - actual_value ).sum()
                sst  = np.square( actual_value - actual_value.mean() ).sum()
                r2_score += (1 - sse / (sst + 1e-6))
            r2_score /= 2
            temp.append(r2_score)
        R2Lst.append(temp)
 
    errLst = np.transpose(np.asarray(errLst))
    R2Lst = np.transpose(np.asarray(R2Lst))
    
    saveDir = 'figures/{}/{}'.format(taskName, expName)
    mkdir(saveDir)

    np.savetxt(saveDir+'/errL2_{}.txt'.format(ntrain), errLst)
    np.savetxt(saveDir+'/r2_{}.txt'.format(ntrain), R2Lst)
    
    
    