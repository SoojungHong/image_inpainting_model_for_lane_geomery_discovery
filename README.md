# Image Inpainting Model
This project is the baseline and modified model (Image inpainting model) experiment.


# Baseline model
The baseline model is GLCIC (GLobally Locally Consistent Image Completion). The paper and source code are in following URL. 
http://iizuka.cs.tsukuba.ac.jp/projects/completion/data/completion_sig2017.pdf


# Experiment models 
Based on the baseline model (GLCIC), several experiments have been conducted. The summary of these experiments are written in confluence page. 
https://confluence.in.here.com/pages/viewpage.action?pageId=1135190018


# Training Features
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
In case the experiment is about loss function, double check the used loss function in completion network and context discriminator

5. In case, resuming the training with previously trained model, check Optimizer is loaded and move to GPU correctly

def optimizer_to(optim, device)

# Train py for each experiment 
1. loss function experiments 
- train_with_various_loss.py 
- gan_losses.py 


2. dilated convolution layer experiments 
- increasing layer -->  train_with_dilatedconv.py 
- dual layer --> train_with_deep_dual_conv.py 


3. enable that model can resume training
- train_enable_resume.py
