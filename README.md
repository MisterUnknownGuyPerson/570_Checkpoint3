# 570_Checkpoint3
Those who know, know

Hello, this is my code for checkpoint 3. All of the code is here.

DEMO VIDEO

DEMO VIDEO is located on google drive https://drive.google.com/file/d/1l811tM6O5mRNnPH54qx6yOt_EQHtCn_X/view?usp=drive_link

DEMO VIDEO

A copy of the code INCLUDING THE MODELS (they are too large files for github) are all locared in the google drive linked bellow.
You can download the models here: https://drive.google.com/drive/folders/1QHVOziEfOcHl37DD9FxHPBuP8jrvPP2F?usp=sharing

The best model i managed to train was model_final_2.pt , you can look at the results folder

The code extends the work of Zhu et Al's Denoising Diffusion Models for Plug-and-Play Image Restoration. The code I built on top of is located at: 

https://github.com/yuanzhi-zhu/DiffPIR.git

Additionally for training the diffusion model I used OpenAI's guded diffusion: Code for training is based on: 

https://github.com/openai/guided-diffusion.git

I used images 60000 - 69000 from FFHQ and the testset I used is located at: 

https://github.com/NVlabs/ffhq-dataset.git


The model is large to upload to github so instead it is located on google drive found here, same as the video, if it is not uploaded on circuit you can find it on this google drive.
Model download:

The way I developed and used this code, and they way you can go about figuring out how it works is:

Copying, uploading and looking throught this colab file which is used for the code to run.

I put all of the code on my google drive, along with zip folders of the images. 
At runtime on colab there are code blocks in the file above that mount google drvie and move the images and unzip them in the runtime memory.
The unzipped image folders may contain subfolders so use the flatten code block to remove all the subfolders and place all images in one folder.

Now the model can be trained, all the way at the bottom , second to last code block is used to move the terminal to the code directory. (change to the directoy of your code)
And the final code block in the colab file starts the training process. YOu can modify it however you would like. The code that I wrote that does the training is in train_dark2bright.py
Also check where the checkpoints are going to save and make sure the image adresses are correct for your setup.

After training, you can move on to inference, for this you could also use colab, I used my laptop, It doesn't really matter, colab may likely be faster.

The may thing to do before running inference is making sure the file paths are correct, there are 3 paths you need to make sure are correct for your setup and those are in the script I wrote main_ddpir_brighten.py
Specifically in main(): "model_path"; ; "input_dark_dir"; "output_dir":

Here you can modify the parameters in main depending on the model you use. Timestep respacing can be tweeked for any model, 25 is fast but noisy output, 1000 is slow but good output. You can use any values in the range.

Good Luck Have Fun
