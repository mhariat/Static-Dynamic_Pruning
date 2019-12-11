# A new pruning framework

Many methods have been used in the context of pruning. But they are very often
tackling only one aspect of the issue. Some of them try to prune the network in a static
way by maintaining a good performance with respect to the worst case. Whereas others
have a more dynamic approach by providing a good accuracy in average. Methods that
are static oriented tend to focus more on the storage aspect of the compression process.
While dynamic methods are more willing to reduce the number of floating point
operations. Our method aims at taking advantage of these two aspects. We propose a
framework mixing both static pruning and dynamic pruning. The static pruning part is
based on the Hessian of the loss function and the study of its eigenvalues in order to
find the best way to reduce the size of the network and inspired of 
[EigenDamage: Structured Pruning in the Kronecker-Factored Eigenbasis](https://arxiv.org/abs/1905.05934).  In this
part, parameters are effectively removed from the network without adding any sparsity,
reducing considerably the storage of the parameters in hardware. Dynamic
compression is managed thanks to smart switches (gating network) spread evenly all
over the network and routing images through a subset of all layers on a per input basis.
The switches take advantage of the fact that some images are easier to recognise than
others amongst the dataset. In this part we used the paper
 [SkipNet: Learning Dynamic Routing in Convolutional Networks](https://arxiv.org/abs/1711.09485) and the corresponding
 code


## Table Results CIFAR-100 & Resnet-32


| Methods                      | Accuracy | FLOPS (in GFLOPS) | Compression |
|------------------------------|----------|-------------------|-------------|
| w/o Pruning                  | 78.19%   | 2.20              | 0%          |
| w/ Static Pruning only       | 77.48%   | 0.51              |  60%        |
| w/ Static and Dynamic Pruning| 77.36%   | 0.4               | 60%%      |

Our purpose is to investigate the adding value of the smart switches. To
do so, a hard threshold of one percent loss on the accuracy is imposed. Then the results
obtained on the one hand with static compression only and on the other hand with both
static and dynamic compression are compared under this hard threshold. We
investigate more particularly the number of flops and the number of parameters left.

## Run

### Prerequisite:
Download CIFAR-100 dataset.


### How to run:
1. Training the overall architecture (with switches)
```
CUDA_VISIBLE_DEVICES=0 python main_sp.py --network resnet \
                                         --depth 32 \
                                         --epochs 200 \
                                         --lr 0.0001 \
                                         --bs 128 \
                                         --num_workers 16 \
                                         --momentum 0.9 \
                                         --weight_decay 0.0005 \
                                         --img_type png \
                                         --resize 32 \
                                         --result_dir /app/results \
                                         --data_dir /usr/share/bind_mount/data/cifar_100

```

2. Static Pruning

```
CUDA_VISIBLE_DEVICES=0 python main_prune.py --config_path config.json
```

3. Dynamic Pruning

```
CUDA_VISIBLE_DEVICES=0 python main_rl_prune.py --network resnet \
                                               --depth 32 \
                                               --epochs 100 \
                                               --lr 0.001 \
                                               --bs 256 \
                                               --num_workers 16 \
                                               --momentum 0.9 \
                                               --weight_decay 0.0005 \
                                               --img_type png \
                                               --resize 32 \
                                               --result_dir /app/results \
                                               --data_dir /usr/share/bind_mount/data/cifar_100 \
                                               --alpha 0.1 \
                                               --gamma 1 \
                                               --rl_weight 0.1

```



