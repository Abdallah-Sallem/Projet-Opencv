# Upload images
from google.colab import files
uploaded = files.upload()

# Check uploaded filenames
print("Uploaded files:", uploaded.keys())

# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# File paths (update these if the filenames are different)
img1_path = 'form.jpg'
img2_path = 'scanned-form.jpg'

# Read images
img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

# Check for loading errors
if img1 is None:
    raise ValueError(f"Could not load image: {img1_path}")
if img2 is None:
    raise ValueError(f"Could not load image: {img2_path}")

# Convert to RGB for visualization
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Display original images
plt.figure(figsize=[20, 10])
plt.subplot(121)
plt.axis('off')
plt.imshow(img1)
plt.title("Original Form")
plt.subplot(122)
plt.axis('off')
plt.imshow(img2)
plt.title("Scanned Form")
plt.show()

# Convert to grayscale for feature matching
img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

# ORB feature detection and description
orb = cv2.ORB_create(500)
keypoints1, descriptors1 = orb.detectAndCompute(img1_gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2_gray, None)

# Draw keypoints for visualization
img1_kp = cv2.drawKeypoints(img1, keypoints1, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_kp = cv2.drawKeypoints(img2, keypoints2, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Match features
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = matcher.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

# Keep only good matches
numGoodMatches = int(len(matches) * 0.1)
matches = matches[:numGoodMatches]

# Draw good matches
im_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)

# Show matches
plt.figure(figsize=[20, 10])
plt.imshow(im_matches)
plt.axis('off')
plt.title("Top ORB Matches")
plt.show()

# Extract matched points
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

# Compute homography
h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

if h is not None:
    height, width, channels = img1.shape
    img2_reg = cv2.warpPerspective(img2, h, (width, height))

    # Show aligned result
    plt.figure(figsize=[20, 10])
    plt.subplot(121)
    plt.imshow(img1)
    plt.axis("off")
    plt.title("Original Form")
    plt.subplot(122)
    plt.imshow(img2_reg)
    plt.axis("off")
    plt.title("Registered Scanned Form")
    plt.show()
else:
    print("Homography could not be computed.")
