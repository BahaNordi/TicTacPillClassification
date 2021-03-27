import os
import sys
from datetime import datetime as dt

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

from utils.tb_writer import create_summary_writer, TensorBoardHandler
from data_loader.data_loader import TicTacDataLoader
from train_utils.train_config import TicTacTrainConfig
from models.network import VggBase
from sklearn import metrics


def train(train_data_loader):
    loss_dict = {'loss': 0}
    model.train()
    for itr, batch in enumerate(train_data_loader):
        image, label = batch
        optimizer.zero_grad()
        output = model(image)
        weight_corrects = pow(1 + config.loss_growth_factor, itr)
        weight = torch.tensor([weight_corrects, 1.])
        loss = F.nll_loss(output, label, weight=weight)
        loss.backward()
        optimizer.step()
        if itr % 100 == 0:
            print('Train iteration {} had loss {:.6f}'.format(itr, loss))
        loss_dict['loss'] = loss
        tb_train_handler.write_tb(loss_dict, epoch, itr)

    torch.save(model.state_dict(), os.path.join(log_dir, 'model_epoch{}.pth'.format(epoch)))


def validate(val_data_loader, epoch):
    val_loss_dict = {'val_loss': 0}
    model.eval()
    all_predictions = torch.FloatTensor()
    all_labels = torch.LongTensor()
    with torch.no_grad():
        for itr, batch in enumerate(val_data_loader):
            image, label = batch
            output = model(image)
            val_loss = F.nll_loss(output, label)
            if itr % 10 == 0:
                print('Val iteration {} had loss {:.6f}'.format(itr, val_loss))
            val_loss_dict['val_loss'] = val_loss
            tb_val_handler.write_tb(val_loss_dict, epoch, itr)
            all_predictions = torch.cat([all_predictions, output], dim=0)
            all_labels = torch.cat([all_labels, label], dim=0)
        all_predictions = F.softmax(all_predictions, dim=1)[:, 1].numpy()
        all_labels = all_labels.numpy()
        # all_hard_predictions = (all_predictions > 0.5).astype(int)
        fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_predictions, pos_label=1)
        ap = metrics.average_precision_score(all_labels, all_predictions)
        # tb_val_handler.write_tb({'F1_score': f1_score}, epoch=epoch, per_iteration=False)
        tb_val_handler.write_tb({'val_AUC': metrics.auc(fpr, tpr)}, epoch=epoch, per_iteration=False)
        tb_val_handler.write_tb({'val_AP': ap}, epoch=epoch, per_iteration=False)


if __name__ == "__main__":
    tictac_train_list = sys.argv[1]
    tictac_val_list = sys.argv[2]
    root_dir = sys.argv[3]
    print(root_dir, tictac_train_list)

    config = TicTacTrainConfig()
    data_loader = TicTacDataLoader(config, root_dir, tictac_train_list, tictac_val_list)

    # defining model
    model = VggBase()

    if config.fine_tune:
        model_path = os.path.join(os.getcwd(), 'saved_models', 'model_epoch99.pth')
        model.load_state_dict(torch.load(model_path))

    lr = config.lr
    weight_decay = config.weight_decay
    experiment_name = 'Arch={} LR={} weight_decay{} note={} time={}'.format('mini_VGG', lr, weight_decay,
                                                                            'Augmentation_WeightedLoss',
                                                                            dt.now().strftime("%Y-%m-%d-%H-%M"))
    log_dir = os.path.join(root_dir, experiment_name)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)

    summary_writer = create_summary_writer(model=model, data_loader=data_loader.train_loader,
                                           log_dir=log_dir) if log_dir else None

    tb_train_handler = TensorBoardHandler(summary_writer, data_loader.train_loader, prefix='train')
    tb_val_handler = TensorBoardHandler(summary_writer, data_loader.val_loader, prefix='val')

    scheduler = MultiStepLR(optimizer, milestones=config.epoch_lr_decay_milestones, gamma=0.1)

    # starting the main train and validation loop
    for epoch in range(config.epoch):
        print('Started Epoch {}/{}'.format(epoch + 1, config.epoch))
        scheduler.step()
        train(data_loader.train_loader)
        validate(data_loader.val_loader, epoch)

    print('End of training!')
