



pip install paddlepaddle-gpu==1.8.4.post107 -i https://mirror.baidu.com/pypi/simple



# 解压预训练模型
cd ~/w*
cp ../data/data53741/output.zip ./output.zip
unzip output.zip
rm -f output.zip




# 安装依赖、解压COCO2017数据集
nvidia-smi
cd ~
pip install pycocotools
cd data
cd data7122
unzip ann*.zip
unzip val*.zip
unzip tes*.zip
unzip image_info*.zip
unzip train*.zip
cd ~/w*






-------------------------------- SOLOv2 --------------------------------
训练
python tools/train.py -c configs/solov2/solov2_light_448_r50_fpn_8gpu_3x.yml --eval --eval_iter 30000




（二）恢复训练
接着上次继续训练。比如上次训练时留下了一个最新的模型，代号是1000，恢复训练时加-r参数：
python tools/train.py -r output/solov2_light_448_r50_fpn_8gpu_3x/1000 -c configs/solov2/solov2_light_448_r50_fpn_8gpu_3x.yml --eval --eval_iter=30000



预测
python tools/infer.py -c configs/solov2/solov2_r50_fpn_8gpu_3x.yml --infer_dir=./test_imgs/


python tools/infer.py -c configs/solov2/solov2_light_448_r50_fpn_8gpu_3x.yml --infer_dir=./test_imgs/


结果压缩成zip方便下载：
rm -f out.zip
zip -r out.zip output/*.jpg



验证
python tools/eval.py -c configs/solov2/solov2_r50_fpn_8gpu_3x.yml


python tools/eval.py -c configs/solov2/solov2_light_448_r50_fpn_8gpu_3x.yml



导出
python tools/export_model.py  -c configs/solov2/solov2_r50_fpn_8gpu_3x.yml --output_dir=./inference_model -o weights=output/solov2_r50_fpn_8gpu_3x/model_final


python tools/export_model.py  -c configs/solov2/solov2_light_448_r50_fpn_8gpu_3x.yml --output_dir=./inference_model -o weights=output/solov2_light_448_r50_fpn_8gpu_3x/model_final




用导出后的模型预测1张图片：（测速）
python deploy/python/infer.py --model_dir=./inference_model/solov2_r50_fpn_8gpu_3x --image_file=./test_imgs/000000000019.jpg --use_gpu=True


python deploy/python/infer.py --model_dir=./inference_model/solov2_light_448_r50_fpn_8gpu_3x --image_file=./test_imgs/000000000019.jpg --use_gpu=True



用导出后的模型预测./test_imgs/目录下的图片：
python deploy/python/infer.py --model_dir=./inference_model/solov2_light_448_r50_fpn_8gpu_3x --image_dir=./test_imgs/ --use_gpu=True



用导出后的模型预测视频：
python deploy/python/infer.py --model_dir=./inference_model/solov2_light_448_r50_fpn_8gpu_3x --video_file D://PycharmProjects/moviepy/che3.mp4 --use_gpu=True



用导出后的模型播放视频：（按esc键停止播放）
python deploy/python/infer.py --model_dir=./inference_model/solov2_light_448_r50_fpn_8gpu_3x --play_video D://PycharmProjects/moviepy/che3.mp4 --use_gpu=True



















