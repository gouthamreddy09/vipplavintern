🎨 Stable Diffusion Fine-tuning with Flickr30k Dataset

🚀 Overview

This project demonstrates fine-tuning Stable Diffusion v1.4 on the Flickr30k dataset using Google Colab. The pipeline is designed for custom image generation, multilingual prompt support (Telugu included), and efficient training in a constrained GPU environment.

Key highlights:
	•	✅ Fine-tuned Stable Diffusion on 30,000+ images with captions
	•	✅ Added Telugu prompt support (via Google Translate)
	•	✅ Optimized training for Google Colab GPUs (6–8GB VRAM)
	•	✅ Reusable modular pipeline with Hugging Face Diffusers

⸻

📦 Prerequisites & Setup

Install Dependencies

pip install datasets transformers sentencepiece diffusers torch accelerate huggingface_hub googletrans

Authenticate with Hugging Face

from huggingface_hub import login
login()  # Requires your Hugging Face token


⸻

📂 Dataset Preparation

Dataset: Flickr30k
	•	Images: 31,783
	•	Captions: 158,915 (5 per image)
	•	Size: ~1.2 GB

Steps:
	1.	Clone and unzip Flickr30k dataset
	2.	Parse captions into separate rows
	3.	Create Hugging Face dataset for efficient loading

!git clone https://huggingface.co/datasets/nlphuji/flickr30k
!unzip flickr30k/flickr30k-images.zip -d flickr30k/images


⸻

🧩 Model Architecture
	•	Base Model: CompVis/stable-diffusion-v1-4
	•	Components:
	•	VAE → encodes & decodes latent space
	•	CLIP Text Encoder → converts prompts to embeddings
	•	UNet → denoising network (fine-tuned)
	•	Scheduler → controls noise schedule

⸻

🎯 Fine-tuning

Training Configuration:
	•	Learning Rate: 1e-5
	•	Batch Size: 4
	•	Precision: fp16 (mixed precision)
	•	Optimizer: AdamW
	•	Epochs: 3

Pipeline:
	1.	Resize images to 512×512
	2.	Normalize to [-1, 1]
	3.	Encode latents with VAE
	4.	Train UNet with text-image pairs

⸻

🖼️ Testing & Inference

Baseline (Before Fine-tuning)

from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")
prompt = "royal enfield gt650"
image = pipe(prompt).images[0]
image.show()

Fine-tuned Model (After Training)

fine_tuned_pipe = StableDiffusionPipeline(
    vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=noise_scheduler
).to("cuda")

prompt = "A photo of a person riding a bicycle"
image = fine_tuned_pipe(prompt).images[0]
image.show()


⸻

🌍 Multilingual Support (Telugu Example)

from googletrans import Translator
translator = Translator()

telugu_prompt = "క్రికెట్ ఆడుతున్న బాలుడు"  # "A boy playing cricket"
english_prompt = translator.translate(telugu_prompt, src='te', dest='en').text

image = fine_tuned_pipe(english_prompt).images[0]
image.show()


⸻

📊 Performance

Component	GPU Memory	Time per step
VAE Encoding	~2GB	~0.1s/image
Text Encoding	~1GB	~0.01s/prompt
UNet Forward	~4GB	~0.5s/step
Training (epoch)	~8GB	~30 min


⸻

🐞 Troubleshooting
	•	CUDA OOM → Reduce batch size, enable fp16, use gradient accumulation
	•	Dataset issues → Check paths & formats (os.path.exists)
	•	Training instability → Use gradient clipping & lower LR
	•	Slow data loading → Use Hugging Face datasets with transforms

⸻

🌟 Future Enhancements
	•	LoRA fine-tuning (efficient training)
	•	DreamBooth for personal object/face training
	•	Advanced schedulers (DDIM, DPM-Solver++)
	•	Gradio/Streamlit web interface
	•	Automated evaluation with FID & CLIP scores

⸻

📖 License

This project is released under the MIT License.
