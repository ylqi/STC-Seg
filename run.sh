
INPUT=${1:-inputs}
RESNET=101  # 50, 101, 101_BiFPN_dcni3
MODEL=model_final

python demo/predict.py \
       --input_dir ${INPUT} \
       --output_dir results \
       --tracking \
       --confidence-threshold 0.5 \
       --tracking_mode DoublePoint+SD \
       --config-file models/STC-Seg_MS_R_${RESNET}_kitti_mots/config.yaml \
       --opts MODEL.WEIGHTS models/STC-Seg_MS_R_${RESNET}_kitti_mots/${MODEL}.pth
