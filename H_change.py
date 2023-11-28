# from Stitcher import Stitcher
import cv2
import numpy as np
import os
from PIL import Image

def change(filenameA, ratio=0.75, reprojThresh=4.0,):
    # 获取输入图片
    imageA_name = os.path.join('./static/images/',  "filenameA.jpg")
    imageB_name = os.path.join('./static/images/',  "board-1.jpg")
    imageA = cv2.imread(imageA_name)
    # imageA = cv2.resize(imageA, (500, 414))
    imageB = cv2.imread(imageB_name)
    # imageB = cv2.resize(imageB, (500, 414))
    imageA_for_sift = imageA.copy()
    # The SIFT key feature points of pictures A and B are detected, and the feature descriptors are calculated
    (kpsA, featuresA) = detectAndDescribe(imageA)
    (kpsB, featuresB) = detectAndDescribe(imageB)
    imageA_for_sift = cv2.drawKeypoints(imageA_for_sift, kpsA, imageA_for_sift)
    cv2.imwrite("./static/images/" + filenameA.split(".")[0] + '-sift.jpg', imageA_for_sift)
    # Match all the feature points of the two images and return the matching results
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])
    M = matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

    # If the result is empty and there are no matching points, we exit the algorithm
    if M is None:
        return None

    # Otherwise, the matching result is extracted
    # H is a 3x3 perspective transformation matrix
    (matches, H, status) = M
    # The Angle of view of picture A is transformed, and the result is the transformed image
    result_a = cv2.warpPerspective(imageA, H, (imageA.shape[1], imageA.shape[0]))
    cv2.imwrite("./static/images/" + filenameA.split(".")[0] + '-H.jpg', result_a)
    result_name=filenameA.split(".")[0]
    return result_name

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detectAndDescribe(image):
    # Convert color images to grayscale
    # Set up the SIFT generator
    descriptor = cv2.xfeatures2d.SIFT_create()
    #SIFT feature points are detected and descriptors are calculated
    (kps, features) = descriptor.detectAndCompute(image, None)


    # Converts the result to a NumPy array


    # Returns a set of feature points and their corresponding descriptive features
    return (kps, features)

def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
    # Build a violence matcher
    matcher = cv2.BFMatcher()

    # KNN was used to detect SIFT feature matching pairs from graphs A and B, K=2
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

    matches = []
    for m in rawMatches:
        # When the ratio of the nearest distance to the second closest distance is less than the ratio value, the matching pair is retained
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            # Store the index of two points in featuresA and featuresB
            matches.append((m[0].trainIdx, m[0].queryIdx))

    # When the filtered matching pairs are greater than 4, the perspective transformation matrix is calculated
    if len(matches) > 4:
        # Get the coordinates of the points that match the pair
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])

        # Calculate the perspective transformation matrix
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

        return (matches, H, status)

    # Returns None if the matching pair is less than 4
    return None

def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
    # Initialize the visual image and connect the left and right sides of the A and B images together
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    # Joint traversal, draw matching pairs
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # When the point pair is successfully matched, it is drawn on the visual diagram
        if s == 1:
            # Draw a match
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

    # Return visual results
    return vis



