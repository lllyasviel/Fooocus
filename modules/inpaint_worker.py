class InpaintWorker:
    def __init__(self, image, mask):
        self.image = image
        self.mask = mask
        # Fooocus inpaint logic
        # 1. ensure that diffusion area cover all masks.
        # 2. ensure that diffusion area cover at lease 30% part of images.
        # 3. ensure that diffusion area has at least 1k resolution.
        raise NotImplemented
