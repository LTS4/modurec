import os
import pickle

import time
import torch
from torch import optim
import torch.nn.functional as F

from grecom.constants import ROOT_PATH
from grecom.data_utils import create_recom_data
from grecom.model import RecomNet, Decoder
from grecom.parser import parser_recommender
from grecom.utils import common_processing, EarlyStopping

from sklearn.model_selection import train_test_split

TOY = False


def train():
    args = parser_recommender()
    args = common_processing(args)
    torch.set_num_threads(6)

    # Load graph
    recom_data = create_recom_data(is_toy=TOY)

    model = RecomNet(recom_data.n_users + recom_data.n_items)
    decoder = Decoder(200, recom_data.ratings.rating.mean())
    optimizer = optim.Adam(
        list(model.parameters()) + list(decoder.parameters()),
        args.lr, weight_decay=5e-4)
    if args.early_stopping:
        earlyS = EarlyStopping(mode='min', patience=args.early_stopping)

    training_loss = []
    validation_loss = []

    n_ratings = len(recom_data.rating_graph.edge_index[0]) // 2
    train_mask, val_mask = train_test_split(list(range(n_ratings)))
    for epoch in range(args.epochs):
        t0 = time.time()
        # Training
        model.train()
        decoder.train()
        batch_size = 32
        for n_batch, i in enumerate(range(0, len(train_mask), batch_size)):
            t1 = time.time()
            optimizer.zero_grad()
            output = model(
                edge_sim=recom_data.similar_graph.edge_index,
                edge_rat=recom_data.rating_graph.edge_index,
                x=recom_data.similar_graph.x
            )
            h1 = output[recom_data.rating_graph.edge_index[0]]
            h2 = output[recom_data.rating_graph.edge_index[1]]
            train_batch = train_mask[i:i + batch_size]
            pred_rating = decoder(h1[train_batch], h2[train_batch])
            real_rating = recom_data.ratings.rating.iloc[train_batch].values
            real_rating = torch.tensor(real_rating, dtype=torch.float)
            train_loss = F.mse_loss(pred_rating, real_rating)
            train_loss.backward()
            training_loss.append(train_loss)
            optimizer.step()
            print(f"Batch: {n_batch}  --- train_mse={train_loss:.2f}, time={time.time() - t1}")
        # Validation
        model.eval()
        decoder.eval()
        with torch.no_grad():
            pred_rating = decoder(h1[val_mask], h2[val_mask])
            real_rating = recom_data.ratings.rating.iloc[val_mask].values
            real_rating = torch.tensor(real_rating, dtype=torch.float)
            val_loss = F.mse_loss(pred_rating, real_rating)
            if args.early_stopping:
                if earlyS.step(val_loss.cpu().numpy()):
                    break
            validation_loss.append(val_loss)
        print(f"Epoch: {epoch}  --- train_mse={train_loss:.2f}, val_mse={val_loss:.2f}, time={time.time() - t0}")

    # Save models
    torch.save(model.state_dict(), os.path.join(ROOT_PATH, 'models', 'recom_model.pt'))
    torch.save(decoder.state_dict(), os.path.join(ROOT_PATH, 'models', 'recom_decoder.pt'))

    # Save results
    with open(os.path.join(ROOT_PATH, 'results', 'training_mse.pkl'), 'wb') as f:
        pickle.dump(training_loss, f)
    with open(os.path.join(ROOT_PATH, 'results', 'validation_mse.pkl'), 'wb') as f:
        pickle.dump(validation_loss, f)


if __name__ == "__main__":
    train()
