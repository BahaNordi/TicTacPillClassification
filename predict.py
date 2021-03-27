import os
import sys
import numpy as np

import torch
import torch.nn.functional as F

from data_loader.data_loader import TicTacDataLoader
from train_utils.train_config import TicTacTrainConfig
from models.network import VggBase


if __name__ == "__main__":
    tictac_test_list = sys.argv[1]

    config = TicTacTrainConfig()
    data_loader = TicTacDataLoader(config, tictac_test_list=tictac_test_list)
    model = VggBase()
    model_path = os.path.join(os.getcwd(), 'saved_models', 'model_epoch99.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    ensemble_rounds = 10
    predictions_ensemble = []
    print('starting predictions...')
    with torch.no_grad():
        for ensemble in range(ensemble_rounds):
            all_predictions = torch.FloatTensor()
            print("Ensemble round {}/{}".format(ensemble + 1, ensemble_rounds))
            for itr, batch in enumerate(data_loader.test_loader):
                image, _ = batch
                output = model(image)
                all_predictions = torch.cat([all_predictions, output], dim=0)
            all_predictions_numpy = F.softmax(all_predictions, dim=1)[:, 1].numpy()
            predictions_ensemble.append(all_predictions_numpy)
    all_predictions = np.mean(predictions_ensemble, axis=0)
    hard_labels = (all_predictions > 0.5).astype(int)
    print('predictions complete')
    print('writing results to disk')
    with open('evaluation.txt', 'w') as f:
        for soft, hard in zip(all_predictions, hard_labels):
            f.write('{:.6f}, {}\n'.format(soft, hard))
    print('Done')
