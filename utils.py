def apply_patchwise(img, fn, patch_size=256):
    '''
    Function that takes an image and applies given function to every square patch of it
    '''
    height, width, _ = img.shape

    offset_height, offset_width = 0, 0

    while offset_height <= height - patch_size:
        while offset_width <= width - patch_size:
            patch = img[offset_height:offset_height + patch_size,
                        offset_width:offset_width + patch_size]
            patch = fn(patch)

            img[offset_height:offset_height + patch_size,
                offset_width:offset_width + patch_size] = patch

            offset_width += patch_size

        offset_height += patch_size
        offset_width = 0
