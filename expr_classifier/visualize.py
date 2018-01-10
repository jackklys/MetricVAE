import numpy as np

# https://stackoverflow.com/questions/43531495/tiling-images-in-a-grid-i-e-with-wrapround-in-tensorflow
def reshape_row(arr):
    return reduce(lambda x, y: np.concatenate((x,y), axis=1), arr)

def reshape_col(arr):
    return reduce(lambda x, y: np.concatenate((x,y), axis=0), arr)

def visualize_grid(arr, num_rows=None):
    num_images, height, width = arr.shape

    if num_rows is None:
        num_rows = int(np.sqrt(num_images))
    num_cols = num_images // num_rows
    rows = []
    for i in range(num_rows):
        row_image = arr[i*num_cols:i*num_cols+num_cols]
        if row_image.shape[0] != num_cols:
            for _ in range(num_cols - row_image.shape[0]):
                row_image = np.concatenate((row_image, np.expand_dims(np.zeros((height, width, 1)), axis=0)), axis=0)
        row_image = reshape_row(row_image)
        rows.append(row_image)
    mosaic = reshape_col(rows)
    return mosaic
