# https://huggingface.co/THUDM/CogView4-6B

# pip install git+https://github.com/huggingface/diffusers.git
# cd diffusers
# pip install -e .

import os, ollama, torch
from diffusers import CogView4Pipeline


response = ollama.chat(
    model='llama3.2-vision',
    messages=[{
        'role': 'user',
        'content': 'What is in this image?',
        'images': ['cogview4-o.png']
    }]
)

prompt = response.message.content

print(prompt)

# There is an orange sports car driving along a coastal road. 
# The car has two seats and a sleek, aerodynamic design. 
# It appears to be moving at high speed, with the ocean waves visible in the background. 
# The sky above is blue with white clouds. The overall atmosphere suggests a sunny day and 
# a sense of freedom and adventure.

os.system(f"pidof ollama > pidollama.txt")

with open("pidollama.txt", 'r') as f:
    pid_ollama = f.read()

os.system(f"sudo kill {pid_ollama}") # sudo python image_generate_with_CogView4.py 


torch.cuda.empty_cache()

pipe = CogView4Pipeline.from_pretrained("THUDM/CogView4-6B", torch_dtype=torch.bfloat16)

# Open it for reduce GPU memory usage
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# prompt = """A vibrant orange sports car sits proudly under the gleaming sun, its polished 
#             exterior smooth and flawless, casting a mirror-like reflection. The car features a low, 
#             aerodynamic body, angular headlights that gaze forward like predatory eyes, and a set 
#             of black, high-gloss racing rims that contrast starkly with the red. A subtle hint of 
#             chrome embellishes the grille and exhaust, while the tinted windows suggest a luxurious 
#             and private interior. The scene conveys a sense of speed and elegance, the car appearing 
#             as if it's about to burst into a sprint along a coastal road, with the Netherland's mills and 
#             North sea's azure waves crashing in the background.""" # > cogview4-o.png

image = pipe(
            prompt=prompt,
            guidance_scale=3.5,
            num_images_per_prompt=1,
            num_inference_steps=50,
            width=1024,
            height=1024,
        ).images[0]

image.save("cogview4-o1.png")
