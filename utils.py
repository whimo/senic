def apply_to_patch(img, fn, coord, patch_size):
    '''
    Applies given function to a square patch of an image
    '''
    y, x = coord

    patch = img[y:y + patch_size,
                x:x + patch_size]
    patch = fn(patch)

    img[y:y + patch_size,
        x:x + patch_size] = patch

    return img


def apply_patchwise(img, fn, patch_size=256):
    '''
    Function that splits image into square patches and apples given function to every of them
    '''
    height, width, _ = img.shape

    offset_height, offset_width = 0, 0

    while offset_height <= height - patch_size:
        while offset_width < width - patch_size:
            apply_to_patch(img, fn, (offset_height, offset_width), patch_size)
            offset_width += patch_size
        apply_to_patch(img, fn, (offset_height, width - patch_size), patch_size)

        offset_height += patch_size
        offset_width = 0

        if offset_height > height - patch_size and offset_height < height:
            offset_height = height - patch_size

    return img
