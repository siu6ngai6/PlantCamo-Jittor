# ZoomNeXt: A Unified Collaborative Pyramid Network for Camouflaged Object Detection (TPAMI 2024)

<div align="center">
  <img src="https://github.com/lartpang/ZoomNeXt/assets/26847524/f43f773b-a81f-4c64-a809-9764b53dd52c" alt="Logo">
</div>


```bibtex
@ARTICLE {ZoomNeXt,
    title   = {ZoomNeXt: A Unified Collaborative Pyramid Network for Camouflaged Object Detection},
    author  ={Youwei Pang and Xiaoqi Zhao and Tian-Zhu Xiang and Lihe Zhang and Huchuan Lu},
    journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
    year    = {2024},
    doi     = {10.1109/TPAMI.2024.3417329},
}
```

## Error
```bibtex
Traceback (most recent call last):
  File "/kaggle/working/main_for_image.py", line 408, in <module>
    main()
  File "/kaggle/working/main_for_image.py", line 398, in main
    train(model=model, cfg=cfg)
  File "/kaggle/working/main_for_image.py", line 278, in train
    optimizer.backward(loss)
  File "/opt/conda/lib/python3.10/site-packages/jittor/optim.py", line 194, in backward
    grads = jt.grad(loss, params_has_grad, retain_graph)
  File "/opt/conda/lib/python3.10/site-packages/jittor/__init__.py", line 465, in grad
    return core.grad(loss, targets, retain_graph)
RuntimeError: Wrong inputs arguments, Please refer to examples(help(jt.grad)).

Types of your inputs are:
 self	= module,
 args	= (Var, list, bool, ),

The function declarations are:
 vector<VarHolder*> _grad(VarHolder* loss, const vector<VarHolder*>& targets, bool retain_graph=true)

Failed reason:[f 0220 06:18:21.440805 60 reshape_op.cc:45] Check failed x_items(73728) == y_items(294912) reshape shape is invalid for input of size

```

## Prepare Data

> [!note]
>
> - CAD dataset can be found at https://drive.google.com/file/d/1XhrC6NSekGOAAM7osLne3p46pj1tLFdI/view?usp=sharing
> - COD dataset for testing can be found at https://drive.google.com/file/d/1V0iSEdYJrT0Y_DHZfVGMg6TySFRNTy4o/view?usp=sharing
>
> Based on the following data setup, the performance of the VCOD dataset evaluated directly using the training script is now consistent with the paper.

## Evaluation

```shell
# ICOD
python main_for_image.py --config configs/icod_train.py --model-name <MODEL_NAME> --evaluate --load-from <TRAINED_WEIGHT>
# VCOD
python main_for_video.py --config configs/vcod_finetune.py --model-name <MODEL_NAME> --evaluate --load-from <TRAINED_WEIGHT>
```

## Training

### Image Camouflaged Object Detection

```shell
python main_for_image.py --config configs/icod_train.py --pretrained --model-name PvtV2B2_ZoomNeXt
python main_for_image.py --config configs/icod_train.py --pretrained --model-name PvtV2B3_ZoomNeXt
python main_for_image.py --config configs/icod_train.py --pretrained --model-name PvtV2B4_ZoomNeXt
python main_for_image.py --config configs/icod_train.py --pretrained --model-name PvtV2B5_ZoomNeXt
```

### Video Camouflaged Object Detection

1. Pretrain on COD10K-TR: `python main_for_image.py --config configs/icod_pretrain.py --info pretrain --model-name PvtV2B5_ZoomNeXt --pretrained`
2. Finetune on MoCA-Mask-TR: `python main_for_video.py --config configs/vcod_finetune.py --info finetune --model-name videoPvtV2B5_ZoomNeXt --load-from <PRETAINED_WEIGHT>`

> [!note]
> If you meets the OOM problem, you can try to reduce the batch size or switch on the `--use-checkpoint` flag:
> `python main_for_image.py/main_for_video.py <your config> --use-checkpoint`
