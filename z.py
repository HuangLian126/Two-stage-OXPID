'''

    conda activate HL_rtdetr

    CUDA_VISIBLE_DEVICES=1 python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_oxpid_s.yml --amp

python twoStage.py train \
  --json /home/abc/HL/two_OXPID/data/OXPID_M/train.json \
  --img-root /home/abc/HL/two_OXPID/data/OXPID_M/train \
  --yw-ckpt  /home/abc/HL/two_OXPID/output/rtdetr_r50vd_6x_ldxray/checkpoint0030.pth \
  --rtdetr-config /home/abc/HL/two_OXPID/configs/rtdetr/rtdetr_r50vd_6x_pidray.yml \
  --rtdetr-root   /home/abc/HL/two_OXPID \
  --out /home/abc/HL/two_OXPID/output/outputTwoStage/box_classifier_clip_edl_rtdetr.pt \
  --clip-model ViT-B-16 --clip-pretrained openai --prompt-style xray \
  --unfreeze-backbone --backbone-lr 1e-5 --freeze-epochs 3 --lr 5e-4 \
  --epochs 50 --batch-size 64 --adapter-ratio 4 --adapter-beta 0.5 --kl-weight 0.1 \
  --geom-weight 0.01 --geom-anneal 0.5 --geom-feature-weight 0.0




'''