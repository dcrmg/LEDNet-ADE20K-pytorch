import os
import time
import numpy as np
import torch
import torch.utils.data as data
from argparse import ArgumentParser
from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from utils.loss import CrossEntropyLoss2d
import importlib
from utils.iouEval import iouEval
from shutil import copyfile
from core.data.dataloader import get_segmentation_dataset
from torchvision import transforms
from core.utils.distributed import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

def train(args, model, enc=False):
    # image transform
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    data_kwargs = {'dataset_root': args.datadir, 'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size, 'encode': enc}
    train_dataset = get_segmentation_dataset('ade20k', split='train', mode='train', **data_kwargs)
    val_dataset = get_segmentation_dataset('ade20k', split='val', mode='val', **data_kwargs)

    train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=False)
    train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size)
    val_sampler = make_data_sampler(val_dataset, shuffle=False, distributed=False)
    val_batch_sampler = make_batch_data_sampler(val_sampler, args.batch_size)

    loader = data.DataLoader(dataset=train_dataset,
                                   batch_sampler=train_batch_sampler,
                                   num_workers=args.num_workers,
                                   pin_memory=True)
    loader_val = data.DataLoader(dataset=val_dataset,
                                 batch_sampler=val_batch_sampler,
                                 num_workers=args.num_workers,
                                 pin_memory=True)

    criterion = CrossEntropyLoss2d()
    print(type(criterion))

    savedir = f'../save/{args.savedir}'

    if (enc):
        automated_log_path = savedir + "/automated_log_encoder.txt"
        modeltxtpath = savedir + "/model_encoder.txt"
    else:
        automated_log_path = savedir + "/automated_log.txt"
        modeltxtpath = savedir + "/model.txt"    

    if (not os.path.exists(automated_log_path)):    #dont add first line if it exists 
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))

    optimizer = Adam(model.parameters(), args.lr, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)

    start_epoch = 1
    best_acc = 0.0
    if args.resume:
        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
        else:
            filenameCheckpoint = savedir + '/checkpoint.pth.tar'

        assert os.path.exists(filenameCheckpoint)
        checkpoint = torch.load(filenameCheckpoint)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        best_acc = checkpoint['best_acc']
        print("=> Loaded checkpoint at epoch {})".format(checkpoint['epoch']))

    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5) # set up scheduler     ## scheduler 1
    lambda1 = lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.7)                                 ## scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)                                  ## scheduler 2

    for epoch in range(start_epoch, args.num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----"," LR", optimizer.param_groups[0]['lr'], "-----")

        epoch_loss = []
        time_train = []
     
        doIouTrain = args.iouTrain   
        doIouVal =  args.iouVal      

        if (doIouTrain):
            iouEvalTrain = iouEval(args.NUM_CLASSES)

        usedLr = optimizer.param_groups[0]['lr']

        model.train()
        total_train_step = len(train_dataset)//args.batch_size
        total_val_step = len(val_dataset)//args.batch_size
        for step, (images, labels, _) in enumerate(loader):
            start_time = time.time()

            imgs_batch = images.shape[0]
            if imgs_batch != args.batch_size:
                break            
            
            if args.cuda:
                inputs = images.cuda()
                targets = labels.cuda()
            
            outputs = model(inputs, only_encode=enc)

            optimizer.zero_grad()
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            scheduler.step(epoch)  ## scheduler 2

            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)

            if (doIouTrain):
                targets = torch.unsqueeze(targets, 1)
                iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average:0.4} (epoch: {epoch}, step: {step}/{total_train_step})',
                "// Remaining time: %.1f s" % ((total_train_step - step) * sum(time_train) / len(time_train)))
            
        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        
        iouTrain = 0
        if (doIouTrain):
            iouTrain, iou_classes = iouEvalTrain.getIoU()
            print ("EPOCH IoU on TRAIN set: ", iouTrain.item()*100, "%")
        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        epoch_loss_val = []
        time_val = []

        if (doIouVal):
            iouEvalVal = iouEval(args.NUM_CLASSES)

        for step, (images, labels, _) in enumerate(loader_val):
            start_time = time.time()

            imgs_batch = images.shape[0]
            if imgs_batch != args.batch_size:
                break
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
            with torch.no_grad():
                inputs = Variable(images)     
                targets = Variable(labels)
                outputs = model(inputs, only_encode=enc)
                loss = criterion(outputs, targets)
            epoch_loss_val.append(loss.item())
            time_val.append(time.time() - start_time)

            #Add batch to calculate TP, FP and FN for iou estimation
            if (doIouVal):
                targets = torch.unsqueeze(targets, 1)
                iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
                print(f'VAL loss: {average:0.4} (epoch: {epoch}, step: {step}/{total_val_step})',
                "// Remaining time: %.1f s" % ((total_val_step-step) * sum(time_val) / len(time_val)))

        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
        # scheduler.step(average_epoch_loss_val, epoch)  ## scheduler 1   # update lr if needed

        iouVal = 0
        if (doIouVal):
            iouVal, iou_classes = iouEvalVal.getIoU()
            print("EPOCH IoU on VAL set: ", iouVal.item()*100, "%")

        # remember best valIoU and save checkpoint
        if iouVal == 0:
            current_acc = -average_epoch_loss_val
        else:
            current_acc = iouVal
        print('best acc:', best_acc,' current acc:',current_acc.item())
        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)
        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
            filenameBest = savedir + '/model_best_enc.pth.tar'    
        else:
            filenameCheckpoint = savedir + '/checkpoint.pth.tar'
            filenameBest = savedir + '/model_best.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)

        #SAVE MODEL AFTER EPOCH
        if (enc):
            filename = f'{savedir}/model_encoder-{epoch:03}.pth'
            filenamebest = f'{savedir}/model_encoder_best.pth'
        else:
            filename = f'{savedir}/model-{epoch:03}.pth'
            filenamebest = f'{savedir}/model_best.pth'
        if args.epochs_save > 0 and step > 0 and step % args.epochs_save == 0:
            torch.save(model.state_dict(), filename)
            print(f'save: {filename} (epoch: {epoch})')
        if (is_best):
            torch.save(model.state_dict(), filenamebest)
            print(f'save: {filenamebest} (epoch: {epoch})')
            if (not enc):
                with open(savedir + "/best.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))   
            else:
                with open(savedir + "/best_encoder.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))           

        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain, iouVal, usedLr ))
    return(model)

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print("Saving model as best")
        torch.save(state, filenameBest)

def main(args):
    savedir = f'../save/{args.savedir}'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    #Load Model
    assert os.path.exists(args.model + ".py"), "Error: model definition not found"
    model_file = importlib.import_module(args.model)
    model = model_file.Net(args.NUM_CLASSES, outputsize=args.crop_size)
    copyfile(args.model + ".py", savedir + '/' + args.model + ".py")
    
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    if (not args.decoder):
        print("========== ENCODER TRAINING ===========")
        model = train(args, model, True)
    print("========== DECODER TRAINING ===========")

    #model.load_state_dict(torch.load('../save/logs/model_encoder_best.pth'))

    if args.pretrainedEncoder:
        print("Loading encoder pretrained in imagenet")
        from lednet_imagenet import LEDNet as LEDNet_imagenet
        pretrainedEnc = torch.nn.DataParallel(LEDNet_imagenet(1000))
        pretrainedEnc.load_state_dict(torch.load(args.pretrainedEncoder)['state_dict'])
        pretrainedEnc = next(pretrainedEnc.children()).features.encoder
        if (not args.cuda):
            pretrainedEnc = pretrainedEnc.cpu()     #because loaded encoder is probably saved in cuda
    else:
        pretrainedEnc = next(model.children()).encoder
    model = model_file.Net(args.NUM_CLASSES, outputsize=args.crop_size, encoder=pretrainedEnc)  #Add decoder to encoder
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
    model = train(args, model, False)   #Train decoder
    print("========== TRAINING FINISHED ===========")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)  #NOTE: cpu-only not tested yet
    parser.add_argument('--model', default= "lednet")
    parser.add_argument('--datadir', default='./datasets')
    parser.add_argument('--NUM-CLASSES', type=int, default=150,
                        help='ADE20K classes')
    parser.add_argument('--base-size', type=int, default=520,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=480,
                        help='crop image size')
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int, default=50)
    parser.add_argument('--epochs-save', type=int, default=0)    #save model every X epochs
    parser.add_argument('--savedir', type=str, default='logs')
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--pretrainedEncoder') #, default=" ")
    parser.add_argument('--iouTrain', action='store_true', default=False) #recommended: False (takes more time to train otherwise)
    parser.add_argument('--iouVal', action='store_true', default=True)
    parser.add_argument('--resume', default=False)    #Use this flag to load last checkpoint for training

    main(parser.parse_args())
