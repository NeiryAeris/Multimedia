from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Define a preprocessing pipeline
preprocess = transforms.Compose(
    [
        transforms.Resize(256),  # Resize the shorter side to 256 pixels
        transforms.CenterCrop(224),  # Crop a 224x224 patch from the center
        transforms.ToTensor(),  # Convert image to PyTorch tensor (scales to [0, 1])
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet statistics
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

save_path = "./save/"

# Load your image
img_path = "./dummy.jpg"  # Replace with your image path
image = Image.open(img_path).convert("RGB")

filterImage = cv2.imread(img_path)
filterImage = cv2.cvtColor(filterImage, cv2.COLOR_BGR2RGB)

gaussian_blur = cv2.GaussianBlur(filterImage, (5, 5), 0)
median_blur = cv2.medianBlur(filterImage, 5)
bilateral_filter = cv2.bilateralFilter(filterImage, 9, 75, 75)
gaussian_for_sharpen = cv2.GaussianBlur(filterImage, (9, 9), 10)
sharpened = cv2.addWeighted(filterImage, 1.5, gaussian_for_sharpen, -0.5, 0)

# Apply preprocessing
img_preprocessed = preprocess(image)


# Check the tensor shape (should be [3, 224, 224] for an RGB image)
print("Preprocessed image shape:", img_preprocessed.shape)


# Optionally, visualize the processed image (after de-normalizing)
def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


# Clone and denormalize for display purposes
img_display = denormalize(
    img_preprocessed.clone(), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
)
img_display = img_display.permute(1, 2, 0).numpy()  # Rearrange dimensions to H x W x C

img_display_clamped = np.clip(img_display, 0, 1)
img_display_uint8 = (img_display_clamped * 255).astype(np.uint8)

titles = [
    "Original",
    "Gaussian Blur",
    "Median Blur",
    "Bilateral Filter",
    "Sharpened",
    "Preprocessed",
]
images = [image, gaussian_blur, median_blur, bilateral_filter, sharpened, img_display]

Image.fromarray(gaussian_blur).save(f"{save_path}gaussian_blur.jpg")
Image.fromarray(median_blur).save(f"{save_path}median_blur.jpg")
Image.fromarray(bilateral_filter).save(f"{save_path}bilateral_filter.jpg")
Image.fromarray(sharpened).save(f"{save_path}sharpened.jpg")
Image.fromarray(img_display_uint8).save(f"{save_path}preprocessed.jpg")

plt.figure(figsize=(15, 4))

for i in range(len(images)):
    plt.subplot(1, 6, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
