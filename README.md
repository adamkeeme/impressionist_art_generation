How to Run:

Training:
python run.py --mode train --data_dir data/wikiart/Impressionism

Stylizing: (applying the trained model to stylize the images from "input" folder and output into "output" directory as impressionist style paintings.)
python run.py --mode stylize

The results:
As different impressionist painters had a rather varied and distinct styles, i only trained the model exclusively on the famed Claude Monet's impressionist paintings. That means 13,000 of his impressionist paintings were used in the training (from Wikiart), and with all the limitations of a lightweight VAE, I saw some success with turning a real life photo of a woman holding a parasol into his most popular work "Woman with a Parasol (1875)".
While crude, if you squint your eyes on all three images (including the acutal painting at the bottom), you can see the resemblance.
Input:
![woman_parasol](https://github.com/user-attachments/assets/587ecfe0-915b-4a66-9f93-5d1dc9f51ca0)

Output:
![woman_parasol](https://github.com/user-attachments/assets/4035b374-a033-4b2a-a832-13a5e76ea054)

Actual Painting:
![woman_parasol](https://github.com/user-attachments/assets/eb60a8f9-1169-41d2-b23e-0ce360dc43e5)



