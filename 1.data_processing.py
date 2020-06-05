from __future__ import absolute_import, division, print_function

from utils import *


def move_blurred(image_directory, threshold):
    """
    라플라시안 variance 구해서 thresholding threshold보다 낮으면 error라는 디렉토리로 move

    1) 현 경로에 "error 라는 폴더 생성"
    2) image_directory안의 모든 이미지를 가져와서 라플라시안 variance 추출
    3) threshold보다 낮으면 이미지를 "error" 폴더로 이동
    4) 이동한 파일명과 라플라시안 분산값 추출
    :param image_directory: 대상 디렉토리
    :param threshold: 라플라시안 분산 threshold 값
    """
    def variance_of_laplacian(image):
        # compute the Laplacian of the image and then return the focus
        return cv2.Laplacian(image, cv2.CV_64F).var()

    current_dir = os.getcwd()
    error_dir = "error"
    dir_make(os.path.join(current_dir, error_dir))

    for imagepath in paths.list_images(image_directory):
        image = cv2.imread(imagepath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)

        if fm < threshold:
            name = imagepath.split(os.sep)[-1]
            shutil.move(imagepath, error_dir)
            print(name, fm)


def crop_hist_equal(image_directory, chdir_name, coords):
    """
    좌표 지정 시 (coords) 좌표에 맞게 특정 디렉토리의 이미지를 자르고
    히스토그램 평활화 후 새로 생성한 디렉토리(chdir_name)에 저장 : sub directory 있을 시 사용

    1) 가공된 이미지가 저장될 디렉토리 생성
    2) image_directory 에서 이미지를 한 장씩 받아와 coords 에 맞춰 이미지 crop
    3) crop된 이미지를 히스토그램 평활화
    4) 새로운 디렉토리에 저장
    :param image_directory: 가공할 대상 이미지가 있는 디렉토리
    :param chdir_name: 가공된 이미지가 저장될 디렉토리
    :param coords: crop 좌표
    """
    dir_make(chdir_name)
    for imagepath in paths.list_images(image_directory):
        img = Image.open(imagepath)
        cropped_img = img.crop(coords)
        hist_img = ImageOps.equalize(cropped_img)
        name = chdir_name + os.sep + imagepath.split(os.sep)[-1]
        hist_img.save(name)


def just_crop(image_directory, chdir_name, coords,subdir):
    """
    sub directory 여부를 지정해서 좌표에 맞춰 자르고 저장

    1) 가공된 이미지가 저장될 디렉토리 생성
    2) image_directory 에서 이미지를 한 장씩 받아와 coords 에 맞춰 이미지 crop
    3) 하위 경로 그대로 이미지를 저장할지 여부를 boolean으로 받아서
        3-1) True일 경우 image_directory와 동일한 하위 디렉토리를 새 디렉토리에 생성하여 그 안에 저장
        3-1) False일 경우 image_directory는 하위 디렉토리가 없는 것

    :param image_directory: 가공할 대상 이미지가 있는 디렉토리
    :param chdir_name: 가공된 이미지가 저장될 디렉토리
    :param coords: crop 좌표
    :param subdir: image_directory 안에 하위 디렉토리가 있는지의 여부
    """
    dir_make(chdir_name)
    for imagepath in paths.list_images(image_directory):
        img = Image.open(imagepath)
        cropped_img = img.crop(coords)
        if subdir == True :
            dir_make(os.path.join(chdir_name, imagepath.split(os.sep)[-2]))
            name = os.path.join(chdir_name, imagepath.split(os.sep)[-2] , imagepath.split(os.sep)[-1])
        else:
            name = chdir_name + os.sep + imagepath.split(os.sep)[-1]
        cropped_img.save(name)


def crop_enhance(image_directory, chdir_name, coords):
    """
    히스토그램 평활화와 이미지 contrast 강화(1.5) 후 저장 : sub directory 있다고 가정

    1) 가공된 이미지가 저장될 디렉토리 생성
    2) image_directory 에서 이미지를 한 장씩 받아와 coords 에 맞춰 이미지 crop
    3) 히스토그램 평활화
    4) 이미지 대조 강화
    5) 새로운 디렉토리에 저장
    :param image_directory: 가공할 대상 이미지가 있는 디렉토리
    :param chdir_name: 가공된 이미지가 저장될 디렉토리
    :param coords: crop 좌표
    """
    dir_make(chdir_name)
    for imagepath in paths.list_images(image_directory):
        img = Image.open(imagepath)
        cropped_img = img.crop(coords)
        hist_img = ImageOps.equalize(cropped_img)
        contrast_img = ImageEnhance.Contrast(hist_img).enhance(1.5)
        name = chdir_name + os.sep + imagepath.split(os.sep)[-1]
        contrast_img.save(name)


def hist_equal(image_directory,new_directory):
    """
    히스토그램 평활화 후 저장 :sub directory 있다고 가정

    :param image_directory: 가공할 대상 이미지가 있는 디렉토리
    :param new_directory: 가공된 이미지가 저장될 디렉토리
    :return:
    """
    dir_make(new_directory)
    for imagepath in paths.list_images(image_directory):
        img = Image.open(imagepath)
        hist_img = ImageOps.equalize(img)
        name = new_directory + os.sep + imagepath.split(os.sep)[-1]
        hist_img.save(name)




