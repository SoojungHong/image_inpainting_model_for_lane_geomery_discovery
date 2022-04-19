# Image Inpainting Model
This project is presenting the baseline model and the corresponding paper. 
Furthermore, this repository contains several modified models (Image inpainting models) that improved the image quality. 
The model modification against the vanila model has been done in architecture and in loss functions. 
The models implementing different architecture and different loss functions are separated to different train_<XYZ>.py files. 


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
  
# Train 
  train.py is the training with baseline model
  