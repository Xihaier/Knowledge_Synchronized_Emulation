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

from utils.utils import toNumpy, toTuple


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def vizPred(args, model, test_loader, tstep):
    model.eval()
    mb_size = int(len(test_loader.dataset)/len(test_loader))
    u_out = torch.zeros(mb_size, tstep+1, 1, args.nel, args.nel).to(args.device)
    
    for bidx, (input0, uTarget0) in enumerate(test_loader):
        input0 = torch.unsqueeze(input0, 1)
        uTarget0 = torch.unsqueeze(uTarget0, 2)
        
        dims = torch.ones(len(input0.shape))
        dims[1] = args.nic        
        inputs = input0.repeat(toTuple(toNumpy(dims).astype(int))).to(args.device)
        
        if(bidx == 0):
            u_out[:,0,:,:,:] = input0
            u_target = uTarget0

        for t_idx in range(tstep):
            uPred = model(inputs[:,-1*args.nic:,:,:])
            if(bidx == 0):
                u_out[:,t_idx+1,:,:,:] = uPred
            
            inputs = inputs[:,-1*int(args.nic-1):,:].detach()
            input0 = uPred.detach()
            inputs = torch.cat([inputs,  input0], dim=1)
        break
        
    u_out = torch.squeeze(u_out)
    u_target = torch.squeeze(u_target)
    
    return u_out, u_target


def vizErr(uPred, uTarget):
    uPred = toNumpy(uPred)
    uTarget = toNumpy(uTarget)
    
    idx_sample = 70
    levels = 30
    cmap = plt.get_cmap('RdBu_r')
    mpl.rcParams.update({'font.family': 'serif', 'font.size': 37})

    xLst = []
    xhatLst = []
    xErrlST = []
    minValLst = []
    maxValLst = []

    for idx in [0, 10, 20, 30, 40, 60, 80, 100]:
        x = uTarget[idx_sample,idx,:,:]
        xhat = uPred[idx_sample,idx,:,:]
        xErr = np.abs(x-xhat)
        minVal = min([np.min(x), np.min(xhat), np.min(xErr)])
        maxVal = max([np.max(x), np.max(xhat), np.max(xErr)])

        xLst.append(x)
        xhatLst.append(xhat)
        xErrlST.append(xErr)        
        minValLst.append(minVal)
        maxValLst.append(maxVal)

    plt.close("all")
    fig, axs = plt.subplots(3, 8, figsize=(100, 30))
    for idx, t in enumerate([0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]):
        norm = mpl.colors.Normalize(vmin=minValLst[idx], vmax=maxValLst[idx])

        if idx == 0:
            ax = axs[0][idx]
            ax.set_xticks([])
            ax.set_yticks([])

            pcm = ax.contourf(xLst[idx], levels=levels, cmap=cmap, norm=norm)
            ax.set_title('Initial condition')
                
            cax, cbar_kwds = mcbar.make_axes(axs[0, idx], shrink=0.95, aspect=17, orientation='vertical')
            ticks = np.linspace(0, 1, 4)
            tickLabels = np.linspace(minValLst[idx], maxValLst[idx], 4)
            tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
            cbar = mcbar.ColorbarBase(cax, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
            cbar.set_ticklabels(tickLabels)

            ax = axs[1][idx]
            ax.set_visible(False)
            ax = axs[2][idx]
            ax.set_visible(False)

        else:
            ax = axs[0][idx]
            pcm = ax.contourf(xLst[idx], levels=levels, cmap=cmap, norm=norm)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('t = {}'.format(t))

            ax = axs[1][idx]
            pcm = ax.contourf(xhatLst[idx], levels=levels, cmap=cmap, norm=norm)
            ax.set_xticks([])
            ax.set_yticks([])

            ax = axs[2][idx]
            pcm = ax.contourf(xErrlST[idx], levels=levels, cmap=cmap, norm=norm)
            ax.set_xticks([])
            ax.set_yticks([])

            cax, cbar_kwds = mcbar.make_axes(axs[:,idx], shrink=0.95, aspect=50, orientation='vertical')
            ticks = np.linspace(0, 1, 7)
            tickLabels = np.linspace(minValLst[idx], maxValLst[idx], 7)
            tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
            cbar = mcbar.ColorbarBase(cax, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
            cbar.set_ticklabels(tickLabels)

    plt.savefig('time_{}.png'.format(idx), bbox_inches='tight')      


def postMSE(uPred, uTarget, ntrain, expName, taskName):
    uPred = uPred.to('cpu')
    uTarget = uTarget.to('cpu')

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
    
    # saveDir = 'figures/{}/{}'.format(taskName, expName)
    # mkdir(saveDir)
    # np.savetxt(saveDir+'/mse_{}.txt'.format(ntrain), mseLst)
    
    pos = np.argsort(np.sum(mseLst, axis=0))
        
    levels = 30
    cmap = plt.get_cmap('RdBu_r')
    mpl.rcParams.update({'font.family': 'serif', 'font.size': 37})
    
    numPlot = 3
    for i in range(numPlot):
        uPred_temp = uPred[pos[i],:,:,:]
        uTarget_temp = uTarget[pos[i],:,:,:]
        
        uP1 = toNumpy(uPred_temp[0,:,:])
        uP2 = toNumpy(uPred_temp[25,:,:])
        uP3 = toNumpy(uPred_temp[50,:,:])
        uP4 = toNumpy(uPred_temp[75,:,:])
        uP5 = toNumpy(uPred_temp[100,:,:])
        
        uT1 = toNumpy(uTarget_temp[0,:,:])
        uT2 = toNumpy(uTarget_temp[25,:,:])
        uT3 = toNumpy(uTarget_temp[50,:,:])
        uT4 = toNumpy(uTarget_temp[75,:,:])
        uT5 = toNumpy(uTarget_temp[100,:,:])
        
        uE1 = np.abs(uP1-uT1)
        uE2 = np.abs(uP2-uT2)
        uE3 = np.abs(uP3-uT3)
        uE4 = np.abs(uP4-uT4)
        uE5 = np.abs(uP5-uT5)
        
        minVal = min([np.min(uP1), np.min(uP2), np.min(uP3), np.min(uP4), np.min(uP5), np.min(uT1), np.min(uT2), np.min(uT3), np.min(uT4), np.min(uT5)])
        maxVal = max([np.max(uP1), np.max(uP2), np.max(uP3), np.max(uP4), np.max(uP5), np.max(uT1), np.max(uT2), np.max(uT3), np.max(uT4), np.max(uT5)])
        norm = mpl.colors.Normalize(vmin=minVal, vmax=maxVal)

        minVal1 = min([np.min(uE1), np.min(uE2), np.min(uE3), np.min(uE4), np.min(uE5)])
        maxVal1 = max([np.max(uE1), np.max(uE2), np.max(uE3), np.max(uE4), np.max(uE5)])
        norm1 = mpl.colors.Normalize(vmin=minVal1, vmax=maxVal1)
                
        plt.close("all")
        fig, axs = plt.subplots(3, 5, figsize=(60, 45))
        for row in range(3):
            if row == 0:
                for col in range(5):
                    ax = axs[row][col]
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if col == 0:
                        pcm = ax.contourf(uP1, levels=levels, cmap=cmap, norm=norm)
                    elif col == 1:
                        pcm = ax.contourf(uP2, levels=levels, cmap=cmap, norm=norm)
                    elif col == 2:
                        pcm = ax.contourf(uP3, levels=levels, cmap=cmap, norm=norm)
                    elif col == 3:
                        pcm = ax.contourf(uP4, levels=levels, cmap=cmap, norm=norm)
                    elif col == 4:
                        pcm = ax.contourf(uP5, levels=levels, cmap=cmap, norm=norm)
                    if col == 4:
                        cax, cbar_kwds = mcbar.make_axes(axs[0, :5], shrink=0.95, aspect=50, orientation='horizontal')
                        ticks = np.linspace(0, 1, 9)
                        tickLabels = np.linspace(minVal, maxVal, 9)
                        tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
                        cbar = mcbar.ColorbarBase(cax, cmap=plt.get_cmap(cmap), orientation='horizontal', ticks=ticks)
                        cbar.set_ticklabels(tickLabels)
                        
            elif row == 1:
                for col in range(5):
                    ax = axs[row][col]
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if col == 0:
                        pcm = ax.contourf(uT1, levels=levels, cmap=cmap, norm=norm)
                    elif col == 1:
                        pcm = ax.contourf(uT2, levels=levels, cmap=cmap, norm=norm)
                    elif col == 2:
                        pcm = ax.contourf(uT3, levels=levels, cmap=cmap, norm=norm)
                    elif col == 3:
                        pcm = ax.contourf(uT4, levels=levels, cmap=cmap, norm=norm)
                    elif col == 4:
                        pcm = ax.contourf(uT5, levels=levels, cmap=cmap, norm=norm)
                    if col == 4:
                        cax, cbar_kwds = mcbar.make_axes(axs[1, :5], shrink=0.95, aspect=50, orientation='horizontal')
                        ticks = np.linspace(0, 1, 9)
                        tickLabels = np.linspace(minVal, maxVal, 9)
                        tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
                        cbar = mcbar.ColorbarBase(cax, cmap=plt.get_cmap(cmap), orientation='horizontal', ticks=ticks)
                        cbar.set_ticklabels(tickLabels) 
                        
            else:       
                for col in range(5):
                    ax = axs[row][col]
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if col == 0:
                        pcm = ax.contourf(uE1, levels=levels, cmap=cmap, norm=norm1)
                    elif col == 1:
                        pcm = ax.contourf(uE2, levels=levels, cmap=cmap, norm=norm1)
                    elif col == 2:
                        pcm = ax.contourf(uE3, levels=levels, cmap=cmap, norm=norm1)
                    elif col == 3:
                        pcm = ax.contourf(uE4, levels=levels, cmap=cmap, norm=norm1)
                    elif col == 4:
                        pcm = ax.contourf(uE5, levels=levels, cmap=cmap, norm=norm1)
                    if col == 4:
                        cax, cbar_kwds = mcbar.make_axes(axs[2,:5], shrink=0.95, aspect=50, orientation='horizontal')
                        ticks = np.linspace(0, 1, 9)
                        tickLabels = np.linspace(minVal1, maxVal1, 9)
                        tickLabels = ["{:02.2f}".format(t0) for t0 in tickLabels]
                        cbar = mcbar.ColorbarBase(cax, cmap=plt.get_cmap(cmap), orientation='horizontal', ticks=ticks)
                        cbar.set_ticklabels(tickLabels)                 
                
        plt.savefig('pred_{}.png'.format(i), bbox_inches='tight')     

        
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
    
    
    