import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np

# Read image and convert it to a torch tensor (channels, rows, columns)
imt = torchvision.io.read_image('/Users/viktorladehoff/Desktop/Personal stuff/Viktorer/chad.png')

# Scale the image to range [0 ; 1]
imt = imt/255.0

# in Torch the image dimensions should be (batch index, channels, rows, columns)
# Therefore we add an empty dimension at location 0
imt = imt.unsqueeze(0)

with torch.inference_mode():

    w = np.ones((imt.shape[1], 5, 5)) / (imt.shape[1]*5*5) # creat kernel (a mean-filter)

    w = w.astype(np.float32) # convert weights to float32-type

    weight = torch.from_numpy(w).unsqueeze(0) # convert to Pytorch tensor and make the filter (N, Channels, Rows, Columns)

    result = torch.conv2d(imt, weight) # convolve the image with the filter

    result = result.squeeze(0).permute(1, 2, 0) # remove empty dimension and change dimensions back to (row, column, channel)

plt.imshow(result.numpy(), cmap='gray')
plt.show()

print('done')