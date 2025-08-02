import math
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, input_dims, heads):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dims = int(input_dims / heads)
        self.input_dims = input_dims

        self.query = nn.Linear(self.head_dims, self.head_dims)
        self.key = nn.Linear(self.head_dims, self.head_dims)
        self.value = nn.Linear(self.head_dims, self.head_dims)
        self.fc = nn.Linear(self.head_dims * heads, self.input_dims)

    def forward(self, query, key, value, mask):
        Batch, Seq_len, embed = query.shape
        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]

        query = query.reshape(Batch, query_len, self.heads, self.head_dims)
        key = key.reshape(Batch, key_len, self.heads, self.head_dims)
        value = value.reshape(Batch, value_len, self.heads, self.head_dims)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        score = torch.einsum('bqhd,bkhd->bhqk', [query, key])
        if mask is not None:
            score = score.masked_fill(mask == 0, float('-1e20'))

        attention_score = nn.Softmax(dim=-1)(score / ((self.head_dims) ** (1 / 2)))
        out = torch.einsum('bhqv,bvhd->bqhd', [attention_score, value]).reshape(Batch, query_len,
                                                                                self.head_dims * self.heads)
        out = self.fc(out)

        return out


class TransformerBasicLayer(nn.Module):
    def __init__(
            self,
            heads,
            embedding_dims,
            dropout,
            forward_expansion,
            layer_norm_eps
    ):
        super(TransformerBasicLayer, self).__init__()
        self.embedding_dims = embedding_dims
        self.attention = SelfAttention(embedding_dims, heads)
        self.layer_norm1 = nn.LayerNorm(embedding_dims, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(embedding_dims, eps=layer_norm_eps)
        self.feed_forward = nn.Sequential(
            *[
                nn.Linear(embedding_dims, embedding_dims * forward_expansion),
                nn.GELU(),
                nn.Linear(embedding_dims * forward_expansion, embedding_dims)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_block = self.attention(x, x, x, mask)
        add = self.dropout(self.layer_norm1(attention_block + x))
        feed_forward = self.feed_forward(add)
        out = self.dropout(self.layer_norm2(feed_forward + add))
        return out


class Transformer(nn.Module):
    def __init__(
            self,
            embedding_dims,
            dropout,
            heads,
            num_of_layers,
            forward_expansion,
            max_len,
            layer_norm_eps=1e-5
    ):
        """
        :param embedding_dims:
        :param dropout:
        :param heads:
        :param num_of_layers:
        :param forward_expansion:
        :param max_len: the maximum length of sequence
        :param layer_norm_eps:
        """
        super(Transformer, self).__init__()
        self.embedding_dims = embedding_dims
        self.positional_embeddings = nn.Parameter(torch.zeros(1, max_len, embedding_dims))
        self.dropout = nn.Dropout(dropout)
        self.basic_layers = nn.ModuleList(
            [
                TransformerBasicLayer(
                    heads,
                    embedding_dims,
                    dropout,
                    forward_expansion,
                    layer_norm_eps
                )
                for _ in range(num_of_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(embedding_dims, eps=layer_norm_eps)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_of_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def casual_mask(self, x):
        """
        create lower triangular matrix as casual mask
        :param x:
        :return:
        """
        # shape[0], shape[1] = batchsize, seq_length
        mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1]))).unsqueeze(1)
        return mask

    def forward(self, x):
        casual_mask = self.casual_mask(x)
        # [bs, seqLength, embedding_dims]
        seq_len = x.shape[1]
        y = self.dropout(x + self.positional_embeddings[:, :seq_len, :])
        for block in self.basic_layers:
            y = block(y, casual_mask)
        y = self.layer_norm(y)

        return y


class Projector_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048, last_batch_norm=False):
        super(Projector_MLP, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        if last_batch_norm:
            self.layer2 = nn.Sequential(
                nn.Linear(hidden_dim, out_dim),
                nn.BatchNorm1d(hidden_dim)
            )
        else:
            self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        b, t, d = x.size()
        x = x.reshape(b * t, d)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(b, t, -1)
        return x


class BidirectionalTemporalEncoder(nn.Module):
    def __init__(
            self,
            embedding_dims,
            hidden_dims,
            dropout,
            heads,
            num_of_layers,
            forward_expansion,
            max_len,
            clip_length=5,
            layer_norm_eps=1e-5,

    ):
        super(BidirectionalTemporalEncoder, self).__init__()
        self.projector1 = Projector_MLP(embedding_dims, hidden_dims, hidden_dims, last_batch_norm=True)

        # Initialize forward and backward transformers
        # Two methods for implementing f_transformer and b_transformer:
        # 1) Use lower triangular matrix for f_transformer (predicting based on preceding frames) and
        # upper triangular matrix for b_transformer (predicting based on subsequent frames).
        # 2) Use lower triangular matrix as the causal mask for both transformers, but input the original
        # sequence to f_transformer and the reversed sequence to b_transformer.
        # Here, We apply the second method.
        self.f_transformer = Transformer(hidden_dims, dropout, heads, num_of_layers, forward_expansion, max_len,
                                         layer_norm_eps=layer_norm_eps)
        self.b_transformer = Transformer(hidden_dims, dropout, heads, num_of_layers, forward_expansion, max_len,
                                         layer_norm_eps=layer_norm_eps)

        self.projector2 = Projector_MLP(hidden_dims, embedding_dims, embedding_dims, last_batch_norm=False)

        # Fusion layer to combine forward and backward predictions
        self.fusion_layer = Projector_MLP(embedding_dims, embedding_dims, embedding_dims, last_batch_norm=False)

        self.clip_length = clip_length

        # used for generating clip-level features for calculating contrastive loss
        self.fb_ta = nn.Linear(embedding_dims, 1)
        self.fb_cntrst_projector3 = Projector_MLP(embedding_dims, hidden_dims, hidden_dims, last_batch_norm=False)

    def forward_transformer(self, x):
        proj_x = self.projector1(x)

        # Apply forward transformer to the original sequence (excluding the last frame)
        f_proj_x = proj_x[:, :-1, :]
        # Reverse the sequence and apply backward transformer (excluding the first frame)
        b_proj_x = torch.flip(proj_x[:, 1:, :], dims=[1])

        f_pred_y = self.f_transformer(f_proj_x)
        b_pred_y = self.b_transformer(b_proj_x)

        # Make frame predictions based on preceding frames
        f_pred_y = self.projector2(f_pred_y)
        # Make frame predictions based on subsequent frames
        b_pred_y = self.projector2(b_pred_y)

        f_pred_y = torch.cat([x[:, :1, :], f_pred_y], dim=1)
        b_pred_y = torch.cat([x[:, x.shape[1] - 1:, :], b_pred_y], dim=1)
        b_pred_y = torch.flip(b_pred_y, dims=[1])

        # Fuse forward and backward predictions
        fb_merge = 1 / 2 * (f_pred_y + b_pred_y)
        fb_pred_y = self.fusion_layer(fb_merge)

        return f_pred_y, b_pred_y, fb_pred_y

    def forward(self, x):
        # Generate frame feature predictions
        f_pred_y, b_pred_y, fb_pred_y = self.forward_transformer(x)

        # Generate clip-level features for contrastive loss computation
        b, t, d = fb_pred_y.size()
        fb_cntrst_y = fb_pred_y.unsqueeze(2)

        fb_clip_features = None

        if self.clip_length:
            num_clips = t // self.clip_length
            fb_clip_features = fb_cntrst_y.reshape([b, num_clips, self.clip_length, -1])
            fb_clip_features = fb_clip_features.reshape([b * num_clips, self.clip_length, -1])
            fb_clip_weights = self.fb_ta(fb_clip_features)
            fb_clip_features = self.fb_cntrst_projector3(fb_clip_features)
            fb_clip_features = torch.mul(fb_clip_features, fb_clip_weights)
            fb_clip_features = torch.sum(fb_clip_features, axis=1).unsqueeze(1)
            fb_clip_features = fb_clip_features.reshape([b, num_clips, -1])

        return f_pred_y, b_pred_y, fb_pred_y, fb_clip_features


class ClipLevelContrastiveLossModule(nn.Module):
    """
    A module for computing clip-level contrastive loss.
    It aims to pull adjacent clips (positive pairs) closer together, while pushing non-adjacent clips (negative pairs) further apart.
    """

    def __init__(self, temperature=1.0):
        super(ClipLevelContrastiveLossModule, self).__init__()

        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduce=None)
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, clip_num):
        """
        Generate a mask to identify negative samples.
        :param clip_num: The number of clips.
        :return: A boolean mask tensor where `True` represents negative pairs.
        """
        N = clip_num
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(clip_num):
            if i != 0:
                mask[i][i - 1] = 0
            if i != (clip_num - 1):
                mask[i][i + 1] = 0

        return mask

    def forward(self, x):
        """
        Compute the contrastive loss based on the clip-level features.
        :param x: Tensor of shape (N, feature_dim), where N is the number of clips.
        :return: The computed contrastive loss.
        """
        N = x.size()[0]
        sim = self.similarity_f(x.unsqueeze(0), x.unsqueeze(1))

        sim_i_j = torch.diag(sim, 1)
        sim_j_i = torch.diag(sim, -1)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0)
        negative_mask = self.mask_correlated_samples(N)
        negative_samples = sim[negative_mask]

        positive_samples_first = positive_samples[:1]
        positive_samples_mid = positive_samples[1:-1]
        positive_samples_last = positive_samples[-1:]

        negative_samples_first = negative_samples[:N - 2]
        negative_samples_mid = negative_samples[N - 2:negative_samples.size(0) - (N - 2)]
        negative_samples_last = negative_samples[-(N - 2):]

        # the first clip
        nominator_first = torch.exp(positive_samples_first / self.temperature)
        denominator_first = torch.exp(negative_samples_first / self.temperature)  # 2*bs, 2*bs
        loss_partial_first = -torch.log(nominator_first / torch.sum(denominator_first))
        loss_first = torch.sum(loss_partial_first) / 1.0

        # the second first clip ~ the last second clip
        positive_samples_mid = positive_samples_mid.reshape(N - 2, -1)
        negative_samples_mid = negative_samples_mid.reshape(N - 2, -1)
        nominator_mid = torch.exp(positive_samples_mid / self.temperature)
        denominator_mid = torch.exp(negative_samples_mid / self.temperature)
        loss_partial_mid = -torch.log(nominator_mid / torch.sum(denominator_mid, dim=1).unsqueeze(1))
        loss_mid = torch.mean(loss_partial_mid, dim=1)
        loss_mid = torch.sum(loss_mid)

        # the last clip
        nominator_last = torch.exp(positive_samples_last / self.temperature)
        denominator_last = torch.exp(negative_samples_last / self.temperature)  # 2*bs, 2*bs
        loss_partial_last = -torch.log(nominator_last / torch.sum(denominator_last))
        loss_last = torch.sum(loss_partial_last) / 1.0

        loss = (loss_first + loss_mid + loss_last) / (1.0 * N)

        return loss