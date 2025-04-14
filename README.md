# 570_Checkpoint3
> *Those who know, know*

This repository contains the full implementation for **Checkpoint 3** of the ECE570 course project. The code builds upon the work of Zhu et al.'s *Denoising Diffusion Models for Plug-and-Play Image Restoration* and adapts it for the novel task of **illumination enhancement** using conditional diffusion.

[Demo Video on Google Drive](https://drive.google.com/file/d/1l811tM6O5mRNnPH54qx6yOt_EQHtCn_X/view?usp=drive_link)

## 📦 Code and Models
All code is provided in this repository. However, due to file size limitations, **trained model checkpoints** are hosted externally, you can download the code from this github, or the google drive bellow, however the models are only on the google drive.

📁 **Download models and full project folder here**:  
[Google Drive - Models & Files](https://drive.google.com/drive/folders/1QHVOziEfOcHl37DD9FxHPBuP8jrvPP2F?usp=sharing)

**Best performing model**: `model_final_2.pt`  
(Results are located in the `results` folder.)

## Built On

- 🔁 Zhu et al.’s **DiffPIR**:
  https://github.com/yuanzhi-zhu/DiffPIR.git
- 🧪 OpenAI’s **guided-diffusion**:
  https://github.com/openai/guided-diffusion.git
- 🧍‍♂️ Face dataset (FFHQ):  
  https://github.com/NVlabs/ffhq-dataset.git  
  (Images 60000–69000 were used.)

## How to Use

### Environment

- Run on **Google Colab Pro (A100 GPU)** for training - Use the .ipynb file and open google colab colab
- Run on anything for inference

### Training Workflow

1. **Mount your Google Drive** in Colab.
2. **Place all code and zip folders** of the images in Drive.
3. Use code blocks in the Colab notebook to:
   - Mount Drive
   - **Unzip image folders** into Colab memory
   - (If subfolders exist, use flattening block to consolidate)
   - change directory
   - start traininig
  
I put all of the code on my google drive, along with zip folders of the images. 
At runtime on colab there are code blocks in the file above that mount google drvie and move the images and unzip them in the runtime memory.
The unzipped image folders may contain subfolders so use the flatten code block to remove all the subfolders and place all images in one folder.

Now the model can be trained, all the way at the bottom , second to last code block is used to move the terminal to the code directory. (change to the directoy of your code)
And the final code block in the colab file starts the training process. You can modify it however you would like. The code that I wrote that does the training is in train_dark2bright.py
Also check where the checkpoints are going to save and make sure the image adresses are correct for your setup.

After training, you can move on to inference, for this you could also use colab, I used my laptop, It doesn't really matter, colab may likely be faster.

The may thing to do before running inference is making sure the file paths are correct, there are 3 paths you need to make sure are correct for your setup and those are in the script I wrote main_ddpir_brighten.py
Specifically in main(): "model_path"; ; "input_dark_dir"; "output_dir":

Here you can modify the parameters in main depending on the model you use. Timestep respacing can be tweeked for any model, 25 is fast but noisy output, 1000 is slow but good output. You can use any values in the range.

You can mount the colab notebook and run all the code blocks, scripts, training and inference from there.

Good Luck Have Fun
