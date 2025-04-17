import time
import torch
import logging
import os
import shutil
import numpy as np
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from lib import config_loader
from lib.metrics import masked_mae, masked_mape, masked_rmse
from lib.dataloader import load_feature_matrix
from lib.utils import getLogger, getSaveDir, set_seed, save_checkpoint
from model.data_embedding import DataEmbedding
from model.dynamic_adj_matrix import DynamicAdjMatrix
from model.dynamic_gcn import DynamicChebNet
from model.fc_output import FCOutputLayer
from model.fusion_adj_matrix import FusionAdjacencyMatrix
from model.fusion_feature_matrix import GatedFusion
from model.multi_scale_conv import MultiScaleConvTemporalModel
from model.temporal_attention import TemporalAttention
from torch.utils.tensorboard import SummaryWriter

# ================================ Prepare ================================
args = config_loader.setTerminal()
data_config, train_config = config_loader.getConfig(args)

# config data info
adj_filename = data_config['adj_filename']
dataset_name = data_config['dataset_name']
num_predict = int(data_config['num_predict'])
num_sensors = int(data_config['num_sensors'])
num_weeks = int(data_config['num_weeks'])
num_days = int(data_config['num_days'])
num_hours = int(data_config['num_hours'])

continue_training = train_config.getboolean('continue_training')
continue_training_name = train_config['continue_training_name']
save_path = train_config['save_path']
save_name = train_config['save_name']
device = train_config['device']
seed = int(train_config['seed'])
learning_rate = float(train_config['learning_rate'])
weight_decay = float(train_config['weight_decay'])
batch_size = int(train_config['batch_size'])
epochs = int(train_config['epochs'])
dropout = float(train_config['dropout'])
embed_dim = int(train_config['embed_dim'])
attention_num_layer = int(train_config['attention_num_layer'])
attention_hidden_dim = int(train_config['attention_hidden_dim'])
multi_scale_conv_unit = list(map(int, train_config['multi_scale_conv_unit'].strip('[]').split(',')))
dynamic_adj_matrix_conv_out_channel = int(train_config['dynamic_adj_matrix_conv_out_channel'])
dynamic_chebnet_num_layer = int(train_config['dynamic_chebnet_num_layer'])
dynamic_chebnet_K = int(train_config['dynamic_chebnet_K'])
dynamic_chebnet_hidden_dim = int(train_config['dynamic_chebnet_hidden_dim'])
dynamic_chebnet_output_dim = int(train_config['dynamic_chebnet_output_dim'])
fc_hidden_unit = list(map(int, train_config['fc_hidden_unit'].strip('[]').split(',')))

# save train
save_dir = getSaveDir(save_path, save_name, continue_training, continue_training_name)
getLogger(save_dir)

# log start info
logging.info(f"dataset: {dataset_name}")

# backup parameters
shutil.copyfile(os.path.join('./configuration', f'{dataset_name}.conf'),
                f"{save_dir}/{dataset_name}_config_backup.conf")
logging.info(f"save parameters: {save_dir}/{dataset_name}_config_backup.conf")

# read file
dataset_dir = f'./dataset/{dataset_name}/{dataset_name}_w{num_weeks}_d{num_days}_h{num_hours}.npz'
dataset = np.load(dataset_dir)

cost_adj_matrix_dir = f'./dataset/{dataset_name}/{dataset_name}_cost_adj_matrix.pt'
similarity_adj_matrix_dir = f'./dataset/{dataset_name}/{dataset_name}_similarity_adj_matrix.pt'

# ================================ Load Data ================================
# feature matrix
train_loader, val_loader, _ = load_feature_matrix(dataset, device, batch_size)
# log start info
logging.info(f"dataset.files: {dataset.files}")
logging.info(f"train_x.shape: {dataset['train_x'].shape} -> (B, N, F, T)")
logging.info(f"train_target.shape: {dataset['train_target'].shape} -> (B, N, T)")
# static adj matrix
cost_adj_matrix = torch.load(cost_adj_matrix_dir).to(device)

similarity_adj_matrix = torch.load(similarity_adj_matrix_dir)
threshold = torch.quantile(similarity_adj_matrix, 0.75).item()
similarity_adj_matrix = (similarity_adj_matrix > threshold).float().to(device)
# log adj matrix shape
logging.info(f"cost_adj_matrix.shape: {cost_adj_matrix.shape} -> (N, N)")
logging.info(f"similarity_adj_matrix.shape: {similarity_adj_matrix.shape} -> (N, N) | threshold = {threshold}")

# ================================ Build Models ================================
_, _, num_features, time_steps = dataset['train_x'].shape
# Data Embedding Layer
data_embedding_layer = DataEmbedding(num_features, embed_dim, dropout)
# Temporal Attention Layer
temporal_attention_week = TemporalAttention(embed_dim, attention_hidden_dim, dropout, attention_num_layer)
temporal_attention_day = TemporalAttention(embed_dim, attention_hidden_dim, dropout, attention_num_layer)
temporal_attention_hour = TemporalAttention(embed_dim, attention_hidden_dim, dropout, attention_num_layer)
# Multi Scale Conv Layer
multi_scale_conv_week = MultiScaleConvTemporalModel(embed_dim, multi_scale_conv_unit)
multi_scale_conv_day = MultiScaleConvTemporalModel(embed_dim, multi_scale_conv_unit)
multi_scale_conv_hour = MultiScaleConvTemporalModel(embed_dim, multi_scale_conv_unit)
# Gated Fusion
fusion_feature_matrix_layer = GatedFusion(embed_dim, embed_dim)
# Dynamic Adjacency Matrix
dynamic_adj_matrix_layer = DynamicAdjMatrix(dynamic_adj_matrix_conv_out_channel)
# Multi Adjacency Matrix Fusion
fusion_adj_matrix_layer = FusionAdjacencyMatrix(cost_adj_matrix, similarity_adj_matrix)
# Dynamic ChebNet
dynamic_gcn = DynamicChebNet(time_steps * embed_dim, dynamic_chebnet_hidden_dim, dynamic_chebnet_output_dim, dynamic_chebnet_num_layer, dropout, dynamic_chebnet_K)
# Fully Connected Decoding Layer
fc_output_layer = FCOutputLayer(dynamic_chebnet_output_dim, fc_hidden_unit, num_predict)

models = {
    'data_embedding_layer': data_embedding_layer,
    'temporal_attention_week': temporal_attention_week,
    'temporal_attention_day': temporal_attention_day,
    'temporal_attention_hour': temporal_attention_hour,
    'multi_scale_conv_week': multi_scale_conv_week,
    'multi_scale_conv_day': multi_scale_conv_day,
    'multi_scale_conv_hour': multi_scale_conv_hour,
    'fusion_feature_matrix_layer': fusion_feature_matrix_layer,
    'dynamic_adj_matrix_layer': dynamic_adj_matrix_layer,
    'fusion_adj_matrix_layer': fusion_adj_matrix_layer,
    'dynamic_gcn': dynamic_gcn,
    'fc_output_layer': fc_output_layer,
}

writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard'))
set_seed(seed)
criterion = nn.HuberLoss().to(device)
optimizer = torch.optim.AdamW([param for layer in models.values() for param in layer.parameters()],
                              lr=learning_rate, weight_decay=weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

# ================================ Train ================================
for layer in models.values():
    layer.to(device)

start_epoch = 0
best_val_loss = np.inf
if continue_training:
    logging.info(f"this training continues from the {save_dir}")
    last_checkpoint_path = os.path.join(save_dir, 'checkpoint', 'last.pt')
    checkpoint = torch.load(last_checkpoint_path)
    for name, model in models.items():
        model.load_state_dict(checkpoint['models_state_dict'][name])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint.get('best_val_loss', np.inf)
    logging.info(f"Checkpoint loaded, resuming from epoch {start_epoch + 1}")
else:
    logging.info(f"this training is a new training {save_dir}")

for epoch in range(start_epoch, epochs):
    train_start_time = time.time()

    logging.info(f"{'*' * 32}Epoch:{epoch + 1}/{epochs}{'*' * 32}")

    for layer in models.values():
        layer.train()

    train_total_loss = 0.0
    train_total_mae = 0.0
    train_total_mape = 0.0
    train_total_rmse = 0.0

    for b_idx, batch in enumerate(train_loader):
        train_x, train_target = batch

        # 1.Data Embedding Layer
        # train_x_embed -> (B, N, T, embed_dim)
        train_x_embed = data_embedding_layer(train_x)

        # 2.Temporal Feature Extraction Block
        # feature_matrix_fusion -> (B, N, T, embed_dim)
        input_week, input_day, input_hour = torch.split(train_x_embed, num_predict, dim=-2)

        # 2.1.Temporal Attention Layer
        output_week_TA = temporal_attention_week(input_week)
        output_day_TA = temporal_attention_day(input_day)
        output_hour_TA = temporal_attention_hour(input_hour)

        # 2.2.Multi Scale Conv Layer
        output_week_MC = multi_scale_conv_week(input_week)
        output_day_MC = multi_scale_conv_day(input_day)
        output_hour_MC = multi_scale_conv_hour(input_hour)

        # 2.3.Concat
        feature_matrix_TA = torch.cat([output_week_TA, output_day_TA, output_hour_TA], dim=-2)
        feature_matrix_MC = torch.cat([output_week_MC, output_day_MC, output_hour_MC], dim=-2)

        # 3.Gated Fusion
        feature_matrix_fusion = fusion_feature_matrix_layer(feature_matrix_TA, feature_matrix_MC)
        feature_matrix_fusion = feature_matrix_fusion + train_x_embed

        # 4.Multi Adjacency Matrix Fusion
        # adj_matrix_fusion -> (B, N, N)

        # 4.1.Dynamic Adjacency Matrix
        X_flow = train_x[:, :, 0, :]
        dynamic_adj_matrix = dynamic_adj_matrix_layer(X_flow)

        # 4.2.Fusion Adj Matrix
        adj_matrix_fusion = fusion_adj_matrix_layer(dynamic_adj_matrix)

        # 5.Dynamic ChebNet
        # output_gcn -> (B, N, gcn_output_dim)
        output_gcn = dynamic_gcn(feature_matrix_fusion, adj_matrix_fusion)

        # 6.Fully Connected Decoding Layer
        # train_output -> (B, N, T)
        train_output = fc_output_layer(output_gcn)

        loss = criterion(train_output, train_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_mae = masked_mae(train_output, train_target, 0.0).item()
        train_mape = masked_mape(train_output, train_target, 0.0).item() * 100
        train_rmse = masked_rmse(train_output, train_target, 0.0).item()

        if (b_idx + 1) % 50 == 0:
            logging.info(f"Batch: {(b_idx + 1):<8}\t"
                         f"Loss: {loss.item():<8.2f}\t"
                         f"MAE: {train_mae:<8.2f}\t"
                         f"MAPE: {train_mape:<8.2f}\t"
                         f"RMSE: {train_rmse:<8.2f}")

        train_total_loss += loss.item()
        train_total_mae += train_mae
        train_total_mape += train_mape
        train_total_rmse += train_rmse

    scheduler.step()

    train_avg_loss = train_total_loss / len(train_loader)
    train_avg_mae = train_total_mae / len(train_loader)
    train_avg_mape = train_total_mape / len(train_loader)
    train_avg_rmse = train_total_rmse / len(train_loader)

    train_time = time.time() - train_start_time

    logging.info(f"Train | "
                 f"Loss: {train_avg_loss:.2f} | "
                 f"MAE: {train_avg_mae:.2f} | "
                 f"MAPE: {train_avg_mape:.2f} | "
                 f"RMSE: {train_avg_rmse:.2f} | "
                 f"Time: {train_time:.2f}s")

    writer.add_scalar('Train/Loss', train_avg_loss, epoch + 1)
    writer.add_scalar('Train/MAE', train_avg_mae, epoch + 1)
    writer.add_scalar('Train/MAPE', train_avg_mape, epoch + 1)
    writer.add_scalar('Train/RMSE', train_avg_rmse, epoch + 1)

    # ================================ Validation ================================
    val_start_time = time.time()

    for layer in models.values():
        layer.eval()

    val_total_loss = 0.0
    val_total_mae = 0.0
    val_total_mape = 0.0
    val_total_rmse = 0.0

    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            val_x, val_target = batch

            val_x_embed = data_embedding_layer(val_x)

            input_week, input_day, input_hour = torch.split(val_x_embed, num_predict, dim=-2)

            output_week_TA = temporal_attention_week(input_week)
            output_day_TA = temporal_attention_day(input_day)
            output_hour_TA = temporal_attention_hour(input_hour)

            output_week_MC = multi_scale_conv_week(input_week)
            output_day_MC = multi_scale_conv_day(input_day)
            output_hour_MC = multi_scale_conv_hour(input_hour)

            feature_matrix_TA = torch.cat([output_week_TA, output_day_TA, output_hour_TA], dim=-2)
            feature_matrix_MC = torch.cat([output_week_MC, output_day_MC, output_hour_MC], dim=-2)

            feature_matrix_fusion = fusion_feature_matrix_layer(feature_matrix_TA, feature_matrix_MC)

            feature_matrix_fusion = feature_matrix_fusion + val_x_embed

            X_flow = val_x[:, :, 0, :]
            dynamic_adj_matrix = dynamic_adj_matrix_layer(X_flow)

            adj_matrix_fusion = fusion_adj_matrix_layer(dynamic_adj_matrix)

            output_gcn = dynamic_gcn(feature_matrix_fusion, adj_matrix_fusion)

            val_output = fc_output_layer(output_gcn)

            val_loss = criterion(val_output, val_target)

            val_mae = masked_mae(val_output, val_target, 0.0).item()
            val_mape = masked_mape(val_output, val_target, 0.0).item() * 100
            val_rmse = masked_rmse(val_output, val_target, 0.0).item()

            val_total_loss += val_loss.item()
            val_total_mae += val_mae
            val_total_mape += val_mape
            val_total_rmse += val_rmse

    val_avg_loss = val_total_loss / len(val_loader)
    val_avg_mae = val_total_mae / len(val_loader)
    val_avg_mape = val_total_mape / len(val_loader)
    val_avg_rmse = val_total_rmse / len(val_loader)

    if val_avg_loss < best_val_loss:
        best_val_loss = val_avg_loss
        save_checkpoint(epoch, models, optimizer, best_val_loss,
                        val_avg_loss, val_avg_mae, val_avg_mape, val_avg_rmse,
                        save_dir, name='best')

    save_checkpoint(epoch, models, optimizer, best_val_loss,
                    val_avg_loss, val_avg_mae, val_avg_mape, val_avg_rmse,
                    save_dir, name='last')

    val_time = time.time() - val_start_time

    logging.info(f"Validation | "
                 f"Loss: {val_avg_loss:.2f} | "
                 f"MAE: {val_avg_mae:.2f} | "
                 f"MAPE: {val_avg_mape:.2f} | "
                 f"RMSE: {val_avg_rmse:.2f} | "
                 f"Time: {val_time:.2f}s")

    writer.add_scalar('Validation/Loss', val_avg_loss, epoch + 1)
    writer.add_scalar('Validation/MAE', val_avg_mae, epoch + 1)
    writer.add_scalar('Validation/MAPE', val_avg_mape, epoch + 1)
    writer.add_scalar('Validation/RMSE', val_avg_rmse, epoch + 1)

writer.close()
