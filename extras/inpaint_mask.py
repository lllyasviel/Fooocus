from rembg import remove, new_session


def generate_mask_from_image(image, mask_model):
    if image is None:
        return

    if 'image' in image:
        image = image['image']

    return remove(
        image,
        session=new_session(mask_model),
        only_mask=True
    )
