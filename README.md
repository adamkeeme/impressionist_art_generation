How to Run the UI and Upload your Image for Impressionist Painting:

1. ```git clone https://github.com/adamkeeme/impressionist_art_generation.git```
2. ```pip install -r requirements.txt```
3. ```python gui.py```
4. go to link http://127.0.0.1:7860 unless you specify a different port
5. upload your images and enjoy your new images stylized to impressionist paintings.

How to Run the Model:

Training:

python run.py --mode train --data_dir data/wikiart/Impressionism

Stylizing: (applying the trained model to stylize the images from "input" folder and output into "output" directory as impressionist style paintings.)

python run.py --mode stylize



The results:

As different impressionist painters had a rather varied and distinct styles, i only trained the model exclusively on the famed Claude Monet's impressionist paintings. That means 13,000 of his impressionist paintings were used in the training (from Wikiart), and with all the limitations of a lightweight VAE, I saw some success(could be an overstatement) with turning a real life photo of a woman holding a parasol into his most known work, "Woman with a Parasol (1875)".
While crude, if you squint your eyes (really hard) on all three images (including the acutal painting at the bottom), you can see the slight resemblance.

Input:

![woman_parasol (1)](https://github.com/user-attachments/assets/d75fdfc1-de02-499f-982d-01862d571919)



Output:

![woman_parasol](https://github.com/user-attachments/assets/4035b374-a033-4b2a-a832-13a5e76ea054)


Actual Painting:

![woman_parasol (2)](https://github.com/user-attachments/assets/345d0239-da3c-460c-9478-df1877d20ee9)



There are 5 more pairs of produced images to original paintings/photos for comparisons in the input and output folders. With 4 of his actual paintings also placed in the "real_paintings_for_reference" folder for your reference.
