
import cv2
import numpy as np 
import os
import pydicom
from configs.graduate.config_training import config
import math

# jpg标记图像中，ct有黑边，而dicom没有黑边，需要将中间的圆提取出来然后resize到512
class profile_processing:
    def __init__(self,pic_path):
        self.pic_path = pic_path
    
    def extract_center(self):
        """
        在原始图像中找到大圆
        """

        # cv2.imdecode()函数从指定的内存缓存中读取数据，并把数据转换(解码)成图像格式;主要用于从网络传输数据中恢复出图像。
        # cv2.imencode()函数是将图片格式转换(编码)成流数据，赋值到内存缓存中;主要用于图像数据格式的压缩，方便网络传输。
        # https://blog.csdn.net/dcrmg/article/details/79155233
        ori = cv2.imdecode(np.fromfile(self.pic_path, dtype=np.uint8), cv2.IMREAD_ANYCOLOR)

        if ori.shape[0] != 512 or ori.shape[1] != 512:
            # interpolation 为使用的差值方法
            ori = cv2.resize(ori, (512, 512), interpolation=cv2.INTER_CUBIC)
        # 从BGR 转换为灰度图
        # 这个是为什么用BGR https://www.learnopencv.com/why-does-opencv-use-bgr-color-format/
        gray = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
        # 霍夫变换圆检测  参数解释 https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghcircles#houghcircles
        # 霍夫变换 TODO 这里先不更改
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=10, minRadius=150,
                                   maxRadius=250)
        for circle in circles[0]:  # TODO 可以判断找到的大圈是不是只有一个
            x = int(circle[0])
            y = int(circle[1])
            r = int(circle[2])

            xbool = x > 200 and x < 300
            ybool = y > 200 and y < 300
            if xbool and ybool:
                if x < r or y < r:
                    continue

                x_start = x - r
                x_end = x + r

                y_start = y - r
                y_end = y + r

                # 把大圈单拿出来 重新缩放到512*512
                red_only_image = cv2.imdecode(np.fromfile(self.pic_path, dtype=np.uint8), cv2.IMREAD_ANYCOLOR)
                red_only_image = red_only_image[y_start:y_end, x_start:x_end]
                red_only_image = cv2.resize(red_only_image, (512, 512), interpolation=cv2.INTER_CUBIC)
                return red_only_image

        return None


class sift_dicom_jpg:
    """
    @anthor: zmy
    """
    def __init__(self, tag_picture):
        self.tag_picture = tag_picture
        self.dicom_total_folder = "H:/medical_data_for_huodong/Images"  # 图像文件根目录
        self.patient_id = tag_picture.split("/")[-2]   # 病人id
        self.seq = int(tag_picture.split("/")[-1].split("-")[-1].split(".")[0])  # 对应的CT的序号

    def get_files(self, path):
        for root, dirs, files in os.walk(path):
            # print(root)  # 当前目录路径
            # print(files)  # 当前路径下所有非目录子文件
            # self.sub_dirs = dirs  # 当前路径下所有子目录
            return files

    def get_dirs(self, path):
        for root, dirs, files in os.walk(path):
            # print(root)  # 当前目录路径
            # print(files)  # 当前路径下所有非目录子文件
            # self.sub_dirs = dirs  # 当前路径下所有子目录
            return dirs


    def find_match_dicom(self):
        """
        找到对应的dicom
        """
        years_folder = self.get_dirs(self.dicom_total_folder)
        for year in years_folder:
            dates_folder = self.get_dirs(self.dicom_total_folder + "/" + year)
            for date in dates_folder:
                ids_folder = self.get_dirs(self.dicom_total_folder + "/" + year + "/" + date)
                for id in ids_folder:
                    if id != self.patient_id:
                        continue
                    # 病人id下的所有的dicom文件列表
                    dicom_list = self.get_files(self.dicom_total_folder + "/" + year + "/" + date + "/" + id)
                    for dicom_name in dicom_list:
                        # 首先读文件
                        dcm = pydicom.read_file(
                            self.dicom_total_folder + "/" + year + "/" + date + "/" + id + "/" + dicom_name)
                        if dcm.InstanceNumber == '':
                            continue
                        if int(dcm.InstanceNumber) != self.seq:
                            continue
                        dcm.image = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
                        # DCM.image = DCM.pixel_array
                        img = dcm.image
                        y = img.shape[0]
                        x = img.shape[1]
                        temp_img = np.zeros(shape=[y, x])
                        row = 0
                        while row < y:
                            col = 0
                            while col < x:
                                # TODO 这里对不同的阈值的取值范围可能不太对
                                if int(img[row][col]) <= -160:
                                    temp_img[row][col] = int(0)
                                elif int(img[row][col]) >= 240:
                                    temp_img[row][col] = int(255)
                                else:
                                    # TODO 进行归一化？
                                    temp_img[row][col] = int((float(img[row][col]) + 160) * 255 / 400)
                                col = col + 1
                            row = row + 1

                        cv2.imwrite("img.jpg", temp_img)
                        dicom_with_windows = cv2.imread("img.jpg", cv2.IMREAD_ANYCOLOR)
                        return dicom_with_windows
        return None

        
    def do_match(self):
        """
        将原来的图和新的特征图进行特征匹配
        """
        # img1 = cv2.imread("img1.jpg")
        # jpg 标签图片
        img1 = cv2.imdecode(np.fromfile(self.tag_picture, dtype=np.uint8), cv2.IMREAD_ANYCOLOR)
        # dicom  找到Dicom文件,读进来的时候就已经是灰度图了
        img2 = self.find_match_dicom()
        if (img2 is None) or (img1 is None):
            return None

        # jpg_gray 转换为灰度图
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # dicom
        img2_gray = img2

        # sift 介绍文档 https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html?highlight=sift#introduction-to-sift-scale-invariant-feature-transform
        sift = cv2.xfeatures2d.SIFT_create()

        # 特征，描述
        kp1, des1 = sift.detectAndCompute(img1_gray, None)
        kp2, des2 = sift.detectAndCompute(img2_gray, None)

        # BFmatcher with default parms
        # 粗暴匹配法 https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html?highlight=bfmatcher#feature-matching
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw first 10 matches.
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[0:10], None, flags=2)
        cv2.imshow('match', img3)
        cv2.waitKey()
        cv2.destroyAllWindows()

        p1, p2, kp_pairs = self.filter_matches(kp1, kp2, matches, ratio=0.5)

        if p1.shape[0] < 10 or p2.shape[0] < 10: # 匹配的特征小于10 就认为是没有匹配上
            print("淘汰：" + self.tag_picture)

            return None

        print(self.tag_picture)  # 匹配上了就进行输出

        # self.explore_match('matches', img1_gray, img2_gray, kp_pairs)
        # img3 = cv2.drawMatchesKnn(img1_gray,kp1,img2_gray,kp2,good[:10],flag=2)

        return self.get_right_jpg(img1, img2, kp_pairs)
    


    def get_right_jpg(self, jpg_img, dicom_img, kp_pairs):
        # temp_mat = jpg_img

        p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
        p2 = np.int32([kpp[1].pt for kpp in kp_pairs])

        number_of_point = p1.shape[0]

        # 产生不相重复的三组随机数
        index1 = np.random.randint(0, number_of_point)
        index2 = np.random.randint(0, number_of_point)
        while index2 == index1:
            index2 = np.random.randint(0, number_of_point)
        index3 = np.random.randint(0, number_of_point)
        while index3 == index1 or index3 == index2:
            index3 = np.random.randint(0, number_of_point)

        point1_1 = p1[index1]
        point2_1 = p1[index2]
        point3_1 = p1[index3]

        point1_2 = p2[index1]
        point2_2 = p2[index2]
        point3_2 = p2[index3]


        # TODO 这里没有看懂，但是大致上可以认为是进行误差矫正
        if point1_1[0] != point1_2[0]:
            slope1 = float(point1_2[1] - point1_1[1]) / (point1_2[0] - point1_1[0])
        else:
            slope1 = 100000000

        if point2_1[0] != point2_2[0]:
            slope2 = float(point2_2[1] - point2_1[1]) / (point2_2[0] - point2_1[0])
        else:
            slope2 = 100000000

        if point3_1[0] != point3_2[0]:
            slope3 = float(point3_2[1] - point3_1[1]) / (point3_2[0] - point3_1[0])
        else:
            slope3 = 100000000

        print(slope1)
        print(slope2)
        print(slope3)

        substract1_2 = np.abs(slope1 - slope2)
        substract1_3 = np.abs(slope1 - slope3)
        substract2_3 = np.abs(slope2 - slope3)

        min = float(np.min(np.array([substract1_2, substract1_3, substract2_3])))

        # point1_1 = p1[index1]
        # point2_1 = p1[index2]
        # point3_1 = p1[index3]
        #
        # point1_2 = p2[index1]
        # point2_2 = p2[index2]
        # point3_2 = p2[index3]

        if np.abs(min - substract1_2) <= 1e-6:
            temp_mat = self.get_four_cut_margin(point1_1, point2_1, point1_2, point2_2, jpg_img, dicom_img)
        elif np.abs(min - substract1_3) <= 1e-6:
            temp_mat = self.get_four_cut_margin(point1_1, point3_1, point1_2, point3_2, jpg_img, dicom_img)
        else:
            temp_mat = self.get_four_cut_margin(point2_1, point3_1, point2_2, point3_2, jpg_img, dicom_img)

        if temp_mat is None:
            return None

        return temp_mat


    def get_four_cut_margin(self, point1_1, point2_1, point1_2, point2_2, jpg_img, dicom_img):
        """
        简单理解就是切下四边的margin 
        """

        xd_total = dicom_img.shape[1]
        yd_total = dicom_img.shape[0]
        xj_total = jpg_img.shape[1]
        yj_total = jpg_img.shape[0]

        xd1 = point1_2[0]
        yd1 = point1_2[1]

        xd2 = point2_2[0]
        yd2 = point2_2[1]

        xj1 = point1_1[0]
        yj1 = point1_1[1]

        xj2 = point2_1[0]
        yj2 = point2_1[1]

        cut_up = self.compute(xd1, yd1, xd2, yd2, xj1, yj1, xj2, yj2)

        if math.isnan(cut_up):
            return None

        yd1 = point1_2[0]
        xd1 = point1_2[1]

        yd2 = point2_2[0]
        xd2 = point2_2[1]

        yj1 = point1_1[0]
        xj1 = point1_1[1]

        yj2 = point2_1[0]
        xj2 = point2_1[1]

        cut_left = self.compute(xd1, yd1, xd2, yd2, xj1, yj1, xj2, yj2)

        if math.isnan(cut_left):
            return None

        xd1 = xd_total - point1_2[0]
        yd1 = yd_total - point1_2[1]
        xd2 = xd_total - point2_2[0]
        yd2 = yd_total - point2_2[1]
        xj1 = xj_total - point1_1[0]
        yj1 = yj_total - point1_1[1]
        xj2 = xj_total - point2_1[0]
        yj2 = yj_total - point2_1[1]

        cut_right = self.compute(xd1, yd1, xd2, yd2, xj1, yj1, xj2, yj2)

        if math.isnan(cut_right):
            return None

        yd1 = xd_total - point1_2[0]
        xd1 = yd_total - point1_2[1]
        yd2 = xd_total - point2_2[0]
        xd2 = yd_total - point2_2[1]
        yj1 = xj_total - point1_1[0]
        xj1 = yj_total - point1_1[1]
        yj2 = xj_total - point2_1[0]
        xj2 = yj_total - point2_1[1]
        cut_down = self.compute(xd1, yd1, xd2, yd2, xj1, yj1, xj2, yj2)

        if math.isnan(cut_down):
            return None

        temp_mat = self.padding_resize(yj_total, xj_total, cut_right, cut_left, cut_up, cut_down, jpg_img)

        try:
            temp_mat = cv2.resize(temp_mat, (yd_total, xd_total))
        except:
            return None

        # cv2.imwrite("cut.jpg", temp_mat)
        return temp_mat



    def padding_resize(self, yj_total, xj_total, cut_right, cut_left, cut_up, cut_down, jpg_img):
        """
        重新进行padding
        """
        # temp_mat = jpg_img

        # cv2.imshow("jpg", jpg_img)
        # cv2.waitKey()

        if cut_up >= 0 and cut_down >= 0:
            temp_mat = jpg_img[cut_up:(yj_total - cut_down)]
        elif cut_up >= 0 and cut_down < 0:
            temp_mat = jpg_img[cut_up:yj_total]
            padding = np.zeros(shape=[np.abs(cut_down), xj_total, 3])
            temp_mat = np.concatenate((temp_mat, padding), axis=0)

        elif cut_up < 0 and cut_down >= 0:
            temp_mat = jpg_img[0:(yj_total - cut_down)]
            padding = np.zeros(shape=[np.abs(cut_up), xj_total, 3])
            temp_mat = np.concatenate((padding, temp_mat), axis=0)

        else:
            temp_mat = jpg_img
            padding_up = np.zeros(shape=[np.abs(cut_up), xj_total, 3])
            padding_down = np.zeros(shape=[np.abs(cut_down), xj_total, 3])
            temp_mat = np.concatenate((padding_up, temp_mat, padding_down), axis=0)

        yj_total = temp_mat.shape[0]

        # cv2.imshow("jpg", temp_mat)
        # cv2.waitKey()

        if cut_left >= 0 and cut_right >= 0:
            temp_mat = temp_mat[:, cut_left:(xj_total - cut_right)]
        elif cut_left >= 0 and cut_right < 0:
            temp_mat = temp_mat[:, cut_up:xj_total]
            padding = np.zeros(shape=[yj_total, np.abs(cut_right), 3])
            temp_mat = np.concatenate((temp_mat, padding), axis=1)

        elif cut_left < 0 and cut_right >= 0:
            temp_mat = temp_mat[:, 0:(xj_total - cut_right)]
            padding = np.zeros(shape=[yj_total, np.abs(cut_left), 3])
            temp_mat = np.concatenate((padding, temp_mat), axis=1)

        else:
            temp_mat = temp_mat
            padding_up = np.zeros(shape=[yj_total, np.abs(cut_left), 3])
            padding_down = np.zeros(shape=[yj_total, np.abs(cut_right), 3])
            temp_mat = np.concatenate((padding_up, temp_mat, padding_down), axis=1)

        cv2.imwrite("padding.jpg", temp_mat)
        temp_mat = cv2.imread("padding.jpg", cv2.IMREAD_ANYCOLOR)

        # cv2.imshow("padding", temp_mat)
        # cv2.waitKey()
        # print(cut_up)
        # print(cut_down)
        # print(cut_left)
        # print(cut_right)

        return temp_mat

    
    def compute(self, xd1, yd1, xd2, yd2, xj1, yj1, xj2, yj2):
        """
        TODO 暂时没有理解，可以先理解为计算偏移量
        """
        sum1 = xd2 * (yj1 * xd1 - yd1 * xj1)
        sum2 = xd1 * (yd2 * xj2 - yj2 * xd2)
        sum3 = yd2 * xd1 - yd1 * xd2
        if sum3 == 0:
            print("偏移量计算失败")
            return float('nan')
        # print(str(sum1) + ":" + str(sum2) + ":" + str(sum3))
        return int(float((sum1 + sum2)) / sum3)


    def filter_matches(self, kp1, kp2, matches, ratio=0.75):
        """
        TODO 滤波器匹配 涵义暂时不明
        """
        mkp1, mkp2 = [], []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                m = m[0]
                mkp1.append(kp1[m.queryIdx])
                mkp2.append(kp2[m.trainIdx])
        p1 = np.float32([kp.pt for kp in mkp1])
        p2 = np.float32([kp.pt for kp in mkp2])
        kp_pairs = list(zip(mkp1, mkp2))
        return p1, p2, kp_pairs

    def explore_match(self, win, img1, img2, kp_pairs, status=None, H=None):
        """
        探索性匹配
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
        vis[:h1, :w1] = img1
        vis[:h2, w1:w1 + w2] = img2
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        if H is not None:
            corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
            corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
            cv2.polylines(vis, [corners], True, (255, 255, 255))

        if status is None:
            status = np.ones(len(kp_pairs), np.bool)
        p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
        p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

        green = (0, 255, 0)

        index = 0
        for (x1, y1), (x2, y2), inlier in list(zip(p1, p2, status)):

            if index % 20 != 0:
                index = index + 1
                continue
            else:
                index = index + 1

            if inlier:
                cv2.line(vis, (x1, y1), (x2, y2), green)
                cv2.circle(vis, (x1, y1), 5, (0, 0, 255))
                cv2.circle(vis, (x2, y2), 5, (0, 0, 255))

        # cv2.imshow(win, vis)
        # cv2.imwrite("match.jpg", vis)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



class recognize_circle:
    def __init__(self, path):
        self.path = path
        self.isLung = True

    # 把原图中的红色提取出来，并转为灰度图
    def get_red(self):
        pp = profile_processing(self.path)
        img = pp.extract_center()
        # 写出图片
        # cv2.imwrite("save_img/extra_big_circle.jpg", img)
        # cv2.imshow('extra', img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        if img is None:
            sdj = sift_dicom_jpg(self.path)

            img = sdj.do_match()

            if img is None:
                return None, None

            else:
                self.isLung = False

        else:
            self.isLung = True

        res = img
        # cv2.imshow('hehe', res)
        #
        # cv2.waitKey()
        hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
        lower_red = np.array([170, 100, 100])
        upper_red = np.array([180, 255, 255])

        # 黑背景白圈
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # 黑背景红圈
        res = cv2.bitwise_and(res, res, mask=mask)
        cv2.imwrite("save_img/extra_red_circle.jpg", res)
        cv2.imshow('red', res)
        cv2.waitKey()
        cv2.destroyAllWindows()
        return res, img

    def get_circle(self):

        # 得到只剩红色的图
        red_only_image, img_after_cut = self.get_red()
        print('hahha')
        # 为了生成测试数据，之后注掉
        # if self.isLung == True:
        #     return None
        #
        # if red_only_image is None:
        #     return None

        # print(red_only_image.shape)
        # 灰度化
        gray = cv2.cvtColor(red_only_image, cv2.COLOR_BGR2GRAY)
        # 霍夫变换圆检测
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=10, minRadius=1, maxRadius=60)
        # 输出返回值，方便查看类型

        if circles is None:
            print("没有识别到圆")
            return None

        # print(circles)
        # 输出检测到圆的个数
        # print(len(circles[0]))

        print('处理识别到的圆')
        print(circles[0])
        # 根据检测到圆的信息，画出每一个圆
        for circle in circles[0]:
            # 圆的基本信息
            # print(circle[2])
            # 坐标行列
            x = int(circle[0])
            y = int(circle[1])
            r = int(circle[2])

            xbool = x > 120 and x < 400
            ybool = y > 60 and y < 430

            if xbool and ybool:
                print(circle)

                id = self.path.split("/")[-2]
                red_only_image = cv2.circle(red_only_image, (x, y), r, (0, 0, 255))
                cv2.imshow('write_red', red_only_image)
                cv2.waitKey()
                cv2.destroyAllWindows()
                # cv2.imwrite("circle_pics/" + id + ".jpg", red_only_image)

                # red_only_image = cv2.circle(red_only_image, (x, y), r, (0, 0, 255))

                # cv2.imshow("red", red_only_image)
                # cv2.waitKey()

                id = self.path.split("/")[-2]

                img_after_cut = img_after_cut[(y - r):(y + r), (x - r):(x + r)]

                # cv2.imwrite("jpg_ori/" + id + ".jpg", img_after_cut)

                return [x, y, r]

            # 半径

            # 在原图用指定颜色标记出圆的位置

        # with open("cut_of_id.txt", "a", encoding="UTF-8") as target:
        #     target.write(self.path.split("/")[-2] + "\n")

        return None








# if __name__ == "__main__":
#     tag_picture = "H:/medical_data_for_huodong/taged_picture/梁邦玉/13355/孙洪君-1-57.jpg"

#     sdj = sift_dicom_jpg(tag_picture)
#     sdj.do_match()

# if __name__ == "__main__":
#     pic_path = "D:/Mywork/医学图像标记代码/李奎权-1-106.jpg"

#     pp = profile_processing(pic_path)
#     # pp.find_out_circle()
#     pp.extract_center()


# if __name__ == "__main__":
#     tag_picture = "H:/medical_data_for_huodong/taged_picture/梁邦玉/13355/孙洪君-1-57.jpg"

#     sdj = sift_dicom_jpg(tag_picture)
#     sdj.do_match()