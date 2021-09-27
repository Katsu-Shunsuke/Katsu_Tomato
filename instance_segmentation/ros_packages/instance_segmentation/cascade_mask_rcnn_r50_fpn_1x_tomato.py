# path relative to location of this config file
# _base_ = "../cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py"
_base_ = "../../../mmdetection/configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py"

# path relative to work_dir (mmdetection), which is set to the directory where you run tools/dist_train.sh
dataset_type = 'CocoDataset'
classes = ('stem', 'tomato', 'pedicel', 'sepal') # extremely weird but order matters
optimizer = dict(lr=0.01)
runner = dict(type='EpochBasedRunner', max_epochs=400)
checkpoint_config = dict(interval=5)
data = dict(
    samples_per_gpu=2, # batch size= samples_per_gpu * n_gpus
    workers_per_gpu=2,
    train=dict(
        type = dataset_type,
        img_prefix='../ml_dataset/annotation_images/tomato_split/agrimind3_09_01_2021/train_images',
        classes=classes,
        ann_file='../ml_dataset/annotation_json/tomato_split/superannotate/agrimind3_09_01_2021/train_coco.json'),
    val=dict(
        type=dataset_type,
        img_prefix='../ml_dataset/annotation_images/tomato_split/agrimind3_09_01_2021/val_images',
        classes=classes,
        ann_file='../ml_dataset/annotation_json/tomato_split/superannotate/agrimind3_09_01_2021/val_coco.json'),
    test=dict(
        type=dataset_type,
        img_prefix='../ml_dataset/annotation_images/tomato_split/agrimind3_09_01_2021/test_images',
        classes=classes,
        ann_file='../ml_dataset/annotation_json/tomato_split/superannotate/agrimind3_09_01_2021/test_coco.json'))

# overwrite number of classes in _base_ config file (_base_/models/cascade_rcnn_r50_fpn.py)
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                # explicitly over-write all the `num_classes` field from default 80 to 5.
                num_classes=4),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=4),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=4)],
        mask_head=dict(num_classes=4))) # do the same for mask, just like with bbox


# load_from = '~/Desktop/tomato/mmdetection/checkpoints/detectors_cascade_rcnn_r50_1x_coco-32a10ba0.pth'

# resume_from = '~/Desktop/tomato/mmdetection/work_dirs/xxxxxxxxxxxxxxxx'





