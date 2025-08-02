# MulSclTE
Official pytorch implementation of Unsupervised Action Segmentation via Multi-scale Temporal-interaction Enhancement.

## Overview
MulSclTE is an unsupervised action segmentation (UAS) framework designed to enhance performance by leveraging multi-scale temporal interactions across global, clip, and frame levels. The framework operates in two main stages:

1. Bi-Encoder Training: Utilizes a self-supervised temporal loss function, which integrates global-level prediction loss and clip-level contrastive loss to enhance representation capability.
2. Inference with PS-Seg: Enhances frame-level interactions by combining frame prediction errors with adjacent frame similarities for improved action boundary detection. It then refines the results by clustering to merge related action segments and mitigate over-segmentation. Implementation will be released later.

## Requirements
- Python 3.7.10 
- Pytorch 1.13.1
- CUDA 11.7

## Datasets
MulSclTE supports the following widely-used UAS datasets:
- Breakfast **[1]**
- YouTube Instructions (YTI) **[2]**
- 50Salads **[3]**
- EPIC-KITCHENS **[4]**

## Usage
To train the Bi-Encoder using the self-supervised temporal loss, using the following command:
```
python -u pretrain.py --model_type "cntrst_bi_encoder" \
    --operate train \
    --ds_name Breakfast \
    --data_root [breakfast_data_root] \
    --n_epochs 1 \
    --batch_size 12 \
    --lr 1e-4 \
    --using_clip \
    --clip_window_size 500 \
    --clip_window_step 100 \
    --lr_scheduler step \
    --step_size 8 \
    --warmup \
    --warmup_max_steps 500 \
    --log_interval 10 \
    --max_seq_len 10000 \
    --hidden_dims 1024 \
    --num_of_layers 2 \
    --heads 2 \
    --save_path [save_root] \
    --use_cntrst \
    --cntrst_clip_width 5 \
    --cntrst_loss_weight 0.5 \
    --pred_loss_weight 0.5 \
    --zero_padding \
    --gpu 7 
```

To encode the frame features, using the following command:
```
python -u pretrain.py --model_type "cntrst_bi_encoder" \
    --ds_name Breakfast \
    --data_root [breakfast_data_root] \
    --max_seq_len 10000 \
    --hidden_dims 1024 \
    --num_of_layers 2 \
    --heads 2 \
    --save_path [save_path] \
    --init_weights [model_weigths_path] \
    --feature_save_root [feature_save_root] \
    --operate test-encode \
    --gpu 7 
```

## Comprehensive documentation and detailed usage instructions are in progress. The codebase is currently undergoing refinement and optimization. We appreciate your patience as we work to provide a more robust and well-documented implementation. Updates will be made available as soon as possible.

## References

[1] Kuehne, H., Arslan, A., & Serre, T. (2014). The language of actions: Recovering the syntax and semantics of goal-directed human activities. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 780-787).

[2] Alayrac, J.-B., Bojanowski, P., Agrawal, N., Sivic, J., Laptev, I., & Lacoste-Julien, S. (2016). Unsupervised learning from narrated instruction videos. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 4575-4583).

[3] Stein, S., & McKenna, S. J. (2013). Combining embedded accelerometers with computer vision for recognizing food preparation activities. In *Proceedings of the 2013 ACM International Joint Conference on Pervasive and Ubiquitous Computing* (pp. 729-738).

[4] D. Damen, H. Doughty, G. M. Farinella, et al., The epic-kitchens dataset: Collection, Challenges and Baselines, TPAMI, vol. 43, no. 11, pp. 4125-4141, 2021.