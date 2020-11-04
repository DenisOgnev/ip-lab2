import cv2
import numpy as np
import random as r
import math
import time

def clamp(value, min, max):
    if (value < min):
        return min
    if (value > max):
        return max
    return value

def addGaussianNoise(image, sigma, mean):
    ga = np.random.normal(sigma, mean, image.shape)
    ga = np.uint8(ga)
    
    result = image + ga

    result[result < 0] = 0
    result[result > 255] = 255

    return result

def addSaltAndPepper(image, n):
    heigth = image.shape[0]
    width = image.shape[1]
    numberOfNoisyPixels = r.randrange(0, math.floor(heigth * width / 100)) * n
    counterOfAddedNoisyPixels = 0

    returnImage = np.array(image)

    for i in range(heigth * width):
        if counterOfAddedNoisyPixels < numberOfNoisyPixels:
            if r.random() > 0.7:
                returnImage[r.randrange(0, heigth)][r.randrange(0, width)] = 255
            else:
                returnImage[r.randrange(0, heigth)][r.randrange(0, width)] = 0
            counterOfAddedNoisyPixels += 1
        else:
            break
    
    return returnImage

def gaussianFilter(image, sigma):

    radius = 3 // 2
    resultImage = np.array(image)
    size = 2 * radius + 1
    kernel = np.zeros((size, size))
    norm = 0.0
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            kernel[i + radius][j + radius] = math.exp(-(i * i + j * j) / (2 * sigma * sigma))
            norm = norm + kernel[i + radius][j + radius]
    for i in range(size):
        for j in range(size):
            kernel[i][j] = kernel[i][j] / norm

    resultR = 0.0
    resultG = 0.0
    resultB = 0.0

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for l in range(-radius, radius + 1):
                for k in range(-radius, radius + 1):
                    idx = clamp(i + k, 0, image.shape[0] - 1)
                    idy = clamp(j + l, 0, image.shape[1] - 1)
                    cv2.waitKey(0)
                    resultR = resultR + resultImage[idx][idy][2] * kernel[k + radius][l + radius]
                    resultG = resultG + resultImage[idx][idy][1] * kernel[k + radius][l + radius]
                    resultB = resultB + resultImage[idx][idy][0] * kernel[k + radius][l + radius]

            resultImage[i][j][2] = int(resultR)
            resultImage[i][j][1] = int(resultG)
            resultImage[i][j][0] = int(resultB)
            resultR = 0.0
            resultG = 0.0
            resultB = 0.0

    return resultImage
                    
def medianFilter(image):
    resultImage = np.array(image)
    size = 3
    radius = size // 2
    R = np.zeros(size * size)
    G = np.zeros(size * size)
    B = np.zeros(size * size)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            resultR = 0.0
            resultG = 0.0
            resultB = 0.0
            index = 0
            for k in range(-radius, radius + 1):
                for l in range(-radius, radius + 1):
                    if i + radius == image.shape[0] or j + radius == image.shape[1]:
                        continue
                    R[index] = image[i + k][j + l][2]
                    G[index] = image[i + k][j + l][1]
                    B[index] = image[i + k][j + l][0]
                    index = index + 1
            R.sort()
            G.sort()
            B.sort()
            resultR = clamp(R[(size * size + 1) // 2], 0, 255)
            resultG = clamp(G[(size * size + 1) // 2], 0, 255)
            resultB = clamp(B[(size * size + 1) // 2], 0, 255)
            resultImage[i][j][2] = resultR
            resultImage[i][j][1] = resultG
            resultImage[i][j][0] = resultB
    return resultImage


def main():
    handle = open("result.txt", "w")
    start_time = 0
    end_time = 0
    image = cv2.imread("1.png")

    start_time = time.time()
    saltAndPepperImage = addSaltAndPepper(image, 10)
    end_time = time.time() - start_time
    handle.write("Изображение с шумом соль и перец сохранено в MySaltAndPepperImage.png, время работы: " + str(end_time) + " секунд")

    start_time = time.time()
    gaussianNoisedImage = addGaussianNoise(image, 0, 10)
    end_time = time.time() - start_time
    handle.write("\nИзображение с шумом гаусса сохранено в MyGaussianNoisedImage.png, время работы: " + str(end_time) + " секунд")

    start_time = time.time()
    medianOpenCV = cv2.medianBlur(gaussianNoisedImage, 3)
    end_time = time.time() - start_time
    handle.write("\nИзображение с OpenCV версией медианного фильтра изображения с шумом гаусса сохранено в MedianOpenCV.png, время работы: " + str(end_time) + " секунд")

    start_time = time.time()
    gaussianOpenCV = cv2.GaussianBlur(gaussianNoisedImage, (5, 5), 0)
    end_time = time.time() - start_time
    handle.write("\nИзображение с OpenCV версией фильтра гаусса изображения с шумом гаусса сохранено в GaussianOpenCV.png, время работы: " + str(end_time) + " секунд")

    start_time = time.time()
    medianOpenCVsalted = cv2.medianBlur(saltAndPepperImage, 3)
    end_time = time.time() - start_time
    handle.write("\nИзображение с OpenCV версией медианного фильтра для изображения с шумом соль и перец сохранено в MedianOpenCVsalted.png, время работы: " + str(end_time) + " секунд")

    start_time = time.time()
    gaussianOpenCVsalted = cv2.GaussianBlur(saltAndPepperImage, (5, 5), 0)
    end_time = time.time() - start_time
    handle.write("\nИзображение с OpenCV версией фильтра гаусса для изображения с шумом соль и перец сохранено в GaussianOpenCVsalted.png, время работы: " + str(end_time) + " секунд")
    
    start_time = time.time()
    myMedian = medianFilter(gaussianNoisedImage)
    end_time = time.time() - start_time
    handle.write("\nИзображение с моей версией медианного фильтра для изображения с шумом гаусса сохранено в MyMedian.png, время работы: " + str(end_time) + " секунд")

    start_time = time.time()
    myGaussian = gaussianFilter(gaussianNoisedImage, 2)
    end_time = time.time() - start_time
    handle.write("\nИзображение с моей версией фильтра гаусса для изображения с шумом гаусса сохранено в MyGaussian.png, время работы: " + str(end_time) + " секунд")

    start_time = time.time()
    myMedianSalted = medianFilter(saltAndPepperImage)
    end_time = time.time() - start_time
    handle.write("\nИзображение с моей версией медианного фильтра для изображения с шумом соль и перец сохранено в MyMedianSalted.png, время работы: " + str(end_time) + " секунд")

    start_time = time.time()
    myGaussianSalted = gaussianFilter(saltAndPepperImage, 2)
    end_time = time.time() - start_time
    handle.write("\nИзображение с моей версией фильтра гаусса для изображения с шумом гаусса сохранено в MyGaussianSalted.png, время работы: " + str(end_time) + " секунд")

    cv2.imwrite("MySaltAndPepperImage.png", saltAndPepperImage)
    cv2.imwrite("MyGaussianNoisedImage.png", gaussianNoisedImage)
    cv2.imwrite("MedianOpenCV.png", medianOpenCV)
    cv2.imwrite("GaussianOpenCV.png", gaussianOpenCV)
    cv2.imwrite("MyMedian.png", myMedian)
    cv2.imwrite("MyGaussian.png", myGaussian)
    cv2.imwrite("MyMedianSalted.png", myMedianSalted)
    cv2.imwrite("MyGaussianSlated.png", myGaussianSalted)
    cv2.imwrite("GaussianOpenCVsalted.png", gaussianOpenCVsalted)
    cv2.imwrite("MedianOpenCVsalted.png", medianOpenCVsalted)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()