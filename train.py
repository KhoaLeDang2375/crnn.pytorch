from __future__ import print_function
from __future__ import division
import editdistance 
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from torch.nn import CTCLoss
import os
import utils
import dataset

import models.crnn as crnn
from torch.amp import autocast
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--trainRoot', required=True, help='path to training lmdb')
parser.add_argument('--valRoot', required=True, help='path to validation lmdb')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='height of input image')
parser.add_argument('--imgW', type=int, default=100, help='width of input image')
parser.add_argument('--nh', type=int, default=256, help='size of LSTM hidden state')
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs')

# ====================== THÊM/MỚI ======================
parser.add_argument('--printEvery', type=int, default=100,
                    help='print loss every N steps (default: 100)')
parser.add_argument('--valEvery', type=int, default=500,
                    help='run validation every N steps (default: 500)')
# =====================================================

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--pretrained', default='', help="path to pretrained model")
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--dict', type=str, default='', help='Đường dẫn tới dict.txt')
parser.add_argument('--expr_dir', default='expr', help='folder to save models')
parser.add_argument('--n_test_disp', type=int, default=10)
parser.add_argument('--saveInterval', type=int, default=500, help='save model every N steps')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--adam', action='store_true')
parser.add_argument('--adadelta', action='store_true', default=True)
parser.add_argument('--keep_ratio', action='store_true')
parser.add_argument('--manualSeed', type=int, default=1234)
parser.add_argument('--random_sample', action='store_true')

opt = parser.parse_args()

# Load multi-language alphabet
if opt.dict:
    with open(opt.dict, 'r', encoding='utf-8') as f:
        opt.alphabet = f.read().rstrip('\r\n')

print(opt)

if not os.path.exists(opt.expr_dir):
    os.makedirs(opt.expr_dir)

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True

# Dataset
train_dataset = dataset.lmdbDataset(root=opt.trainRoot)
test_dataset = dataset.lmdbDataset(root=opt.valRoot, transform=dataset.resizeNormalize((100, 32)))

if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batchSize,
    shuffle=True if sampler is None else False,
    sampler=sampler,
    num_workers=opt.workers,
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio),
    pin_memory=True)   # ← khuyến nghị thêm

nclass = len(opt.alphabet) + 1
nc = 1

converter = utils.strLabelConverter(opt.alphabet)
criterion = CTCLoss(zero_infinity=True)

# Model
crnn_model = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)

if opt.pretrained:
    print('loading pretrained model from %s' % opt.pretrained)
    crnn_model.load_state_dict(torch.load(opt.pretrained))

print(crnn_model)

# GPU + DataParallel + AMP
if opt.cuda:
    crnn_model = crnn_model.cuda()
    crnn_model = torch.nn.DataParallel(crnn_model, device_ids=range(opt.ngpu))
    criterion = criterion.cuda()

scaler = torch.amp.GradScaler('cuda')

loss_avg = utils.averager()

# Optimizer
if opt.adam:
    optimizer = optim.Adam(crnn_model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn_model.parameters())
else:
    optimizer = optim.RMSprop(crnn_model.parameters(), lr=opt.lr)




def val(net, dataset, criterion):
    print('\n=== Starting Validation ===')
    net.eval()
    
    # Chỉ tạm tắt gradient cho validation, không ảnh hưởng model gốc
    with torch.no_grad():
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            shuffle=False, 
            batch_size=opt.batchSize,
            num_workers=opt.workers,
            pin_memory=True
        )

        total_loss = 0.0
        n_correct = 0
        total_ed = 0
        total_chars = 0
        count = 0

        for cpu_images, cpu_texts in tqdm(data_loader, desc="Validation"):
            batch_size = cpu_images.size(0)
            text, length = converter.encode(cpu_texts)

            if opt.cuda:
                image = cpu_images.cuda(non_blocking=True)
                text = text.cuda(non_blocking=True)
                length = length.cuda(non_blocking=True)
            else:
                image = cpu_images

            with autocast('cuda'):
                preds = net(image)
                preds = preds.transpose(0, 1)
                preds_size = torch.IntTensor([preds.size(0)] * batch_size)
                if opt.cuda:
                    preds_size = preds_size.cuda()

                cost = criterion(preds.log_softmax(2), text, preds_size, length)

            total_loss += cost.item()
            count += 1

            # Decode
            _, preds_idx = preds.max(2)
            preds_idx = preds_idx.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds_idx, preds_size, raw=False)

            for pred, target in zip(sim_preds, cpu_texts):
                target = target.lower().strip()
                pred = pred.strip()

                if pred == target:
                    n_correct += 1

                ed = editdistance.eval(pred, target)
                total_ed += ed
                total_chars += len(target)

    avg_loss = total_loss / count
    accuracy = n_correct / (count * opt.batchSize)
    cer = (total_ed / total_chars * 100) if total_chars > 0 else 0.0

    print(f'Validation Loss     : {avg_loss:.4f}')
    print(f'Exact Match Acc     : {accuracy:.4f}')
    print(f'Character Error Rate: {cer:.2f}%')
    print('=' * 60)

    return accuracy, cer

def trainBatch(net, criterion, optimizer, scaler, data):
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    
    text, length = converter.encode(cpu_texts)

    if opt.cuda:
        image = cpu_images.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        length = length.cuda(non_blocking=True)
    else:
        image = cpu_images

    optimizer.zero_grad()

    preds = net(image)
    preds = preds.transpose(0, 1)
    preds_size = torch.IntTensor([preds.size(0)] * batch_size)
    if opt.cuda:
        preds_size = preds_size.cuda()

    cost = criterion(preds.log_softmax(2), text, preds_size, length)

    scaler.scale(cost).backward()
    scaler.step(optimizer)
    scaler.update()

    return cost.detach()


# ====================== TRAINING LOOP ======================
best_cer = float('inf')
best_epoch = 0

for epoch in range(opt.nepoch):
    crnn_model.train()
    epoch_loss = 0.0
    step = 0

    print(f"\n=== Epoch {epoch+1}/{opt.nepoch} ===")

    for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        step += 1
        cost = trainBatch(crnn_model, criterion, optimizer, scaler, data)
        
        epoch_loss += cost.item()
        loss_avg.add(cost)

        if step % opt.printEvery == 0:
            tqdm.write(f'[Epoch {epoch+1}/{opt.nepoch}][Step {step}] Loss: {loss_avg.val():.4f}')
            loss_avg.reset()

    # In loss trung bình của epoch
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"⇒ Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}")

    # === Validation chỉ mỗi 5 epoch ===
    if (epoch + 1) % 5 == 0 or (epoch + 1) == opt.nepoch:
        print(f"\n→ Running Validation at epoch {epoch+1} ...")
        accuracy, cer = val(crnn_model, test_dataset, criterion)

        # Lưu model tốt nhất theo CER
        if cer < best_cer:
            best_cer = cer
            best_epoch = epoch + 1
            torch.save(crnn_model.state_dict(), f'{opt.expr_dir}/netCRNN_best.pth')
            print(f"★ BEST MODEL UPDATED! Epoch {best_epoch} - CER: {cer:.2f}% → saved as netCRNN_best.pth")
        else:
            print(f"   Current CER: {cer:.2f}% (Best so far: {best_cer:.2f}% at epoch {best_epoch})")
    else:
        print("   (Validation will run every 5 epochs)")

    print("-" * 70)

print("\nTraining finished!")
print(f"Best model saved at epoch {best_epoch} with CER = {best_cer:.2f}%")
print(f"Model location: {opt.expr_dir}/netCRNN_best.pth")