import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        x,
        exog_attn=None,
        attn_alpha=0.5,
        attn_mask=None,
        tau=None,
        delta=None,
        channel_mask=None,
        residual_beta=1.0,
    ):
        new_x, attn = self.attention(
            x,
            x,
            x,
            exog_attn=exog_attn,
            attn_alpha=attn_alpha,
            attn_mask=attn_mask,
            tau=tau,
            delta=delta,
            channel_mask=channel_mask,
            residual_beta=residual_beta,
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(
        self,
        x,
        attn_mask=None,
        tau=None,
        delta=None,
        exog_attns=None,
        attn_alpha=0.5,
        channel_mask=None,
        residual_beta=1.0,
    ):
        exog_attns = exog_attns or [None] * len(self.attn_layers)
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer, exog_attn) in enumerate(zip(self.attn_layers, self.conv_layers, exog_attns)):
                layer_delta = delta if i == 0 else None
                x, attn = attn_layer(
                    x,
                    exog_attn=exog_attn,
                    attn_alpha=attn_alpha,
                    attn_mask=attn_mask,
                    tau=tau,
                    delta=layer_delta,
                    channel_mask=channel_mask,
                    residual_beta=residual_beta,
                )
                x = conv_layer(x)
                attns.append(attn)
        else:
            for i, (attn_layer, exog_attn) in enumerate(zip(self.attn_layers, exog_attns)):
                layer_delta = delta if i == 0 else None
                x, attn = attn_layer(
                    x,
                    exog_attn=exog_attn,
                    attn_alpha=attn_alpha,
                    attn_mask=attn_mask,
                    tau=tau,
                    delta=layer_delta,
                    channel_mask=channel_mask,
                    residual_beta=residual_beta,
                )
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
        return x, attns
