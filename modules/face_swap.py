import sys
from PIL import Image
import numpy as np
sys.path.append('../inswapper')

from inswapper.swapper import process

def perform_face_swap(images, inswapper_source_image, inswapper_target_image_index):      
  swapped_images = []

  for item in images:
      print(type(item))
      source_image = Image.fromarray(inswapper_source_image)
      print(f"Target index: {inswapper_target_image_index}")
      result_image = process([source_image], item, "-1", f"{int(inswapper_target_image_index)}", "../inswapper/checkpoints/inswapper_128.onnx")      

  if True:
      from inswapper.restoration import face_restoration,check_ckpts,set_realesrgan,torch,ARCH_REGISTRY,cv2
      
      # make sure the ckpts downloaded successfully
      check_ckpts()
      
      # https://huggingface.co/spaces/sczhou/CodeFormer
      upsampler = set_realesrgan()
      device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

      codeformer_net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512,
                                                        codebook_size=1024,
                                                        n_head=8,
                                                        n_layers=9,
                                                        connect_list=["32", "64", "128", "256"],
                                                      ).to(device)
      ckpt_path = "CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth"
      checkpoint = torch.load(ckpt_path)["params_ema"]
      codeformer_net.load_state_dict(checkpoint)
      codeformer_net.eval()
      
      result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
      result_image = face_restoration(result_image, 
                                      True, 
                                      True, 
                                      1, 
                                      1,
                                      upsampler,
                                      codeformer_net,
                                      device)
      result_image = Image.fromarray(result_image)

      swapped_images.append(result_image)      
  
  return swapped_images

