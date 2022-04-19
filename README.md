# Image Inpainting Model to discover the hidden lane geometry
This project is presenting the baseline model and the corresponding paper. 
Furthermore, this repository contains several modified models (Image inpainting models) that improved the image quality. 
The model modification against the vanila model has been done in architecture and in loss functions. 
The models implementing different architecture and different loss functions are separated to different train_<XYZ>.py files. 
The ideas and experiments are explained in summary paper. 
  
https://github.com/SoojungHong/image_inpainting_model_for_lane_geomery_discovery/blob/main/summary/Image_inpainting_model_paper_SoojungHong_2021.pdf

# Baseline model
The baseline model is GLCIC (GLobally Locally Consistent Image Completion). The paper and source code are in following URL. 
http://iizuka.cs.tsukuba.ac.jp/projects/completion/data/completion_sig2017.pdf


# Experiment models 
Based on the baseline model (GLCIC), several experiments have been conducted. The summary of these experiments are written in confluence page.
(probably you can't read it since it's inside of HERE Technologies' intranet)  
https://confluence.in.here.com/pages/viewpage.action?pageId=1135190018


# Hyperparameters for train
Since there are many experiments conducted and each experiment has different target to achieve. Therefore, before running the experiment, following features of the model need to be double checked in the parameter setting code.

1. Parallel traininig using multiple GPU 
parser.add_argument('--data_parallel', action='store_true', default=False)

2. Save model parameter and optimizer for future training resume in designated path
- parser.add_argument('--init_model_cn', type=str, default=None) #default='/home/shong/<path>'
- parser.add_argument('--init_model_cd', type=str, default=None) #default='/home/shong/<path>'
- parser.add_argument('--init_opt_cn', type=str, default=None) #default='/home/shong/<path>'
- parser.add_argument('--init_opt_cd', type=str, default=None) #default='/home/shong/<path>'
- parser.add_argument('--resume_step', type=int, default=None) #default=4) # resume_step should be the last saved step 

3. Batch size 
- parser.add_argument('--bsize', type=int, default=1)

4. loss function 
In case the experiment is conducting using different loss function, please double check the used loss function in completion network and context discriminator

5. In case, resuming the training with previously trained model, check Optimizer is loaded and move to GPU correctly
  
# Dependencies
  requirements.txt
   
    Python: 3.7.6
    torch: 1.9.0 (cuda 11.1)
    torchvision: 0.10.0
    tqdm: 4.61.1
    Pillow: 8.2.0
    opencv-python: 4.5.2.54
    numpy: 1.19.2
    GPU: Geforce GTX 1080Ti (12GB RAM) X 4

  
# Train 
  train.py is for training with baseline model.
  
  train_with_deep_dual_conv.py is for training with improved architecture which use dual deep dilated convolution layers.
  
  rain_with_various_loss.py is for training the various loss function. The newly adapted loss function is explained in the paper in summary folder.
 
  # in <path-to-this-repo>
python train.py datasets/img_align_celeba results/demo/
  
# Inference
  
  python predict.py model_cn config.json images/test_2.jpg test_2_out.jpg


  
# Data 
  Data for train and evaluation can be downloaded in the Google Drive folders (refer the wiki 'data' page)
