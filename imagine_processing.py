import glob
import random
import struct
import tkinter as tk
import win32ui
from PIL import Image, ImageTk
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import cv2 as cv
from tkinter import filedialog
import tkinter.simpledialog
import collections
from math import log

# 使图片能输入中文
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#创建一个窗口
root = tk.Tk()
#设置主窗口的背景色
root.configure(bg='#262626')
#图片文件的名字
filename = ''
# 图片几何变换时标签放置的初始位置
img_x = 0.5
img_y = 0.5
#全局使用的图片
img_open = ''
#备胎
img_open_backup = ''
#最初的图
img_open_init = ''
#图像合成的底图和文件夹
file_di = ''
file_dir = ''
angle = 0
note_alpha = 0.5


#图像放大
def image_magify(label_img):
    global img_open
    global img_open_backup
    img = np.asarray(img_open)
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    n = 0.9
    newH = int(height / n)
    newW = int(width / n)
    # 创建空白模板，其中np.uint8代表图片的数据类型0-255
    newimg = np.zeros((newH, newW, 3), np.uint8)
    # 对新的图像坐标进行重新计算，对矩阵进行行列遍历
    for i in range(0, newH):
        for j in range(0, newW):
            iNew = int(i * (height * 1.0 / newH))
            jNew = int(j * (width * 1.0 / newW))
            newimg[i, j] = img[iNew, jNew]
    img = Image.fromarray(newimg)
    img_open = img
    img_open_backup = img
    photo = ImageTk.PhotoImage(img)
    label_img.config(image=photo)
    label_img.image = photo


#图像缩小
def image_shrink(label_img):
    global img_open
    global img_open_backup
    img = np.asarray(img_open)
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    n = 1.1
    newH = int(height / n)
    newW = int(width / n)
    # 创建空白模板，其中np.uint8代表图片的数据类型0-255
    newimg = np.zeros((newH, newW, 3), np.uint8)
    # 对新的图像坐标进行重新计算，对矩阵进行行列遍历
    for i in range(0, newH):
        for j in range(0, newW):
            iNew = int(i * (height * 1.0 / newH))
            jNew = int(j * (width * 1.0 / newW))
            newimg[i, j] = img[iNew, jNew]
    img = Image.fromarray(newimg)
    img_open = img
    img_open_backup = img
    photo = ImageTk.PhotoImage(img)
    label_img.config(image=photo)
    label_img.image = photo


#图像上移
def image_upward(label_img):
    global img_x
    global img_y
    img_y -= 0.01
    label_img.place(relx=img_x, rely=img_y)


#图像下移
def image_down(label_img):
    global img_x
    global img_y
    img_y += 0.01
    label_img.place(relx=img_x, rely=img_y)


#图像左移
def image_turn_left(label_img):
    global img_x
    global img_y
    img_x -= 0.01
    label_img.place(relx=img_x, rely=img_y)


#图像右移
def image_turn_right(label_img):
    global img_x
    global img_y
    img_x += 0.01
    label_img.place(relx=img_x, rely=img_y)


#图像翻转
def image_flip(label_img):
    global img_open
    global img_open_backup
    img = np.asarray(img_open)
    imgInfo = img.shape
    high = imgInfo[0]
    width = imgInfo[1]
    # 创建空白模板，其中np.uint8代表图片的数据类型0-255
    newimg = np.zeros((high, width, 3), np.uint8)
    for i in range(0, high):
        for j in range(0, width):
            newimg[i, j] = img[high - i - 1, j]
    print(newimg)
    img = Image.fromarray(newimg)
    img_open = img
    img_open_backup = img
    # 显示图片
    photo = ImageTk.PhotoImage(img)
    label_img.config(image=photo)
    label_img.image = photo

#顺时针旋转90度
def image_revolt_clockwise(label_img):
    global img_open
    global img_open_backup
    global angle
    img = np.asarray(img_open_backup)
    imgInfo = img.shape
    high = imgInfo[0]
    width = imgInfo[1]
    angle = (90 + angle) % 360
    if (int(angle / 90) % 2 == 0):
        reshape_angle = angle % 90
    else:
        reshape_angle = 90 - (angle % 90)
    reshape_radian = math.radians(reshape_angle)  # 角度转弧度
    # 三角函数计算出来的结果会有小数，所以做了向上取整的操作。
    new_high = math.ceil(high * np.cos(reshape_radian) + width * np.sin(reshape_radian))
    new_width = math.ceil(width * np.cos(reshape_radian) + high * np.sin(reshape_radian))
    # 创建空白模板，其中np.uint8代表图片的数据类型0-255
    newimg = np.zeros((new_high, new_width, 3), np.uint8)
    radian = math.radians(angle)
    cos_radian = np.cos(radian)
    sin_radian = np.sin(radian)
    # 通过新图像的每个坐标点找到原始图像中对应的坐标点，再把像素赋值上去
    # x0=xcosθ+ysinθ-0.5w'cosθ-0.5h'sinθ+0.5w'
    # y0=-xsinθ+ycosθ+0.5w'sinθ-0.5h'cosθ+0.5h'
    dx_back = 0.5 * width - 0.5 * new_width * cos_radian - 0.5 * new_high * sin_radian
    dy_back = 0.5 * high + 0.5 * new_width * sin_radian - 0.5 * new_high * cos_radian
    for y in range(new_high):
        for x in range(new_width):
            x0 = x * cos_radian + y * sin_radian + dx_back
            y0 = y * cos_radian - x * sin_radian + dy_back
            # 计算结果是这一范围内的x0，y0才是原始图像的坐标。
            if 0 < int(x0) <= width and 0 < int(y0) <= high:
                # 因为计算的结果会有偏移，所以这里做减一操作。
                newimg[int(y), int(x)] = img[int(y0) - 1, int(x0) - 1]
    img = Image.fromarray(newimg)
    img_open = img
    img_open_backup = img
    # 显示图片
    photo = ImageTk.PhotoImage(img)
    label_img.config(image=photo)
    label_img.image = photo


# 逆时针旋转90度
def image_revolt_anticlockwise(label_img):
    global img_open
    global img_open_backup
    global angle
    img = np.asarray(img_open_backup)
    imgInfo = img.shape
    high = imgInfo[0]
    width = imgInfo[1]
    angle = (-90 + angle) % 360
    if (int(angle / 90) % 2 == 0):
        reshape_angle = angle % 90
    else:
        reshape_angle = 90 - (angle % 90)
    reshape_radian = math.radians(reshape_angle)  # 角度转弧度
    # 三角函数计算出来的结果会有小数，所以做了向上取整的操作。
    new_high = math.ceil(high * np.cos(reshape_radian) + width * np.sin(reshape_radian))
    new_width = math.ceil(width * np.cos(reshape_radian) + high * np.sin(reshape_radian))
    # 创建空白模板，其中np.uint8代表图片的数据类型0-255
    newimg = np.zeros((new_high, new_width, 3), np.uint8)
    radian = math.radians(angle)
    cos_radian = np.cos(radian)
    sin_radian = np.sin(radian)
    # 通过新图像的每个坐标点找到原始图像中对应的坐标点，再把像素赋值上去
    # x0=xcosθ+ysinθ-0.5w'cosθ-0.5h'sinθ+0.5w'
    # y0=-xsinθ+ycosθ+0.5w'sinθ-0.5h'cosθ+0.5h'
    dx_back = 0.5 * width - 0.5 * new_width * cos_radian - 0.5 * new_high * sin_radian
    dy_back = 0.5 * high + 0.5 * new_width * sin_radian - 0.5 * new_high * cos_radian
    for y in range(new_high):
        for x in range(new_width):
            x0 = x * cos_radian + y * sin_radian + dx_back
            y0 = y * cos_radian - x * sin_radian + dy_back
            # 计算结果是这一范围内的x0，y0才是原始图像的坐标。
            if 0 < int(x0) <= width and 0 < int(y0) <= high:
                # 因为计算的结果会有偏移，所以这里做减一操作。
                newimg[int(y), int(x)] = img[int(y0) - 1, int(x0) - 1]
    img = Image.fromarray(newimg)
    img_open = img
    img_open_backup = img
    # 显示图片
    photo = ImageTk.PhotoImage(img)
    label_img.config(image=photo)
    label_img.image = photo


#图像旋转
def image_revolt(label_img):
    global img_open
    global img_open_backup
    global angle
    img = np.asarray(img_open_backup)
    imgInfo = img.shape
    high = imgInfo[0]
    width = imgInfo[1]
    angle = (tk.simpledialog.askinteger(title='获取队列数', prompt='请输入调度队列数量：', initialvalue='3') + angle) % 360
    if (int(angle / 90) % 2 == 0):
        reshape_angle = angle % 90
    else:
        reshape_angle = 90 - (angle % 90)
    reshape_radian = math.radians(reshape_angle)  # 角度转弧度
    # 三角函数计算出来的结果会有小数，所以做了向上取整的操作。
    new_high = math.ceil(high * np.cos(reshape_radian) + width * np.sin(reshape_radian))
    new_width = math.ceil(width * np.cos(reshape_radian) + high * np.sin(reshape_radian))
    # 创建空白模板，其中np.uint8代表图片的数据类型0-255
    newimg = np.zeros((new_high, new_width, 3), np.uint8)
    radian = math.radians(angle)
    cos_radian = np.cos(radian)
    sin_radian = np.sin(radian)
    # 通过新图像的每个坐标点找到原始图像中对应的坐标点，再把像素赋值上去
    # x0=xcosθ+ysinθ-0.5w'cosθ-0.5h'sinθ+0.5w'
    # y0=-xsinθ+ycosθ+0.5w'sinθ-0.5h'cosθ+0.5h'
    dx_back = 0.5 * width - 0.5 * new_width * cos_radian - 0.5 * new_high * sin_radian
    dy_back = 0.5 * high + 0.5 * new_width * sin_radian - 0.5 * new_high * cos_radian
    for y in range(new_high):
        for x in range(new_width):
            x0 = x * cos_radian + y * sin_radian + dx_back
            y0 = y * cos_radian - x * sin_radian + dy_back
            # 计算结果是这一范围内的x0，y0才是原始图像的坐标。
            if 0 < int(x0) <= width and 0 < int(y0) <= high:
                # 因为计算的结果会有偏移，所以这里做减一操作。
                newimg[int(y), int(x)] = img[int(y0) - 1, int(x0) - 1]
    img = Image.fromarray(newimg)
    img_open = img
    img_open_backup = img
    # 显示图片
    photo = ImageTk.PhotoImage(img)
    label_img.config(image=photo)
    label_img.image = photo


#压缩文件的选择
def choose_file_to_compose(e):
    global file_di
    try:
        compose_di = filedialog.askopenfilename()
        file_di = compose_di
        e.delete(0, "end")
        e.insert(0, compose_di)
    except:
        pass


#选择要合成的文件夹
def choose_dir_to_compose(e):
    global file_dir
    try:
        compose_dir = filedialog.askdirectory()  # 1表示打开文件对话框
        file_dir = compose_dir
        e.delete(0, "end")
        e.insert(0, compose_dir)
    except:
        pass


#生成合成后的图片
def generate(label_img2, baseImgPath, imagesPath, alpha):
    global img_open
    global img_open_backup
    baseImg = Image.open(baseImgPath)
    baseImg = baseImg.convert('RGBA')
    files = glob.glob(imagesPath + '/*.*')  # 获取图片
    random.shuffle(files)
    num = len(files)
    # 模板图片大小
    x = baseImg.size[0]
    y = baseImg.size[1]
    # 每张图片数量 这个公式是为了xNum * yNum 的总图片数量<num又成比例的最大整数
    yNum = int((num / (y / x)) ** 0.5)
    if yNum == 0:
        yNum = 1
    xNum = int(num / yNum)
    # 图片大小 因为像素没有小数点 为防止黑边所以+1
    xSize = int(x / xNum) + 1
    ySize = int(y / yNum) + 1
    # 生成数量的随机列表 用于随机位置合成图片
    l = [n for n in range(0, xNum * yNum)]
    random.shuffle(l)
    toImage = Image.new('RGB', (x, y))
    num = 1
    for file in files:
        if num <= xNum * yNum:
            num = num + 1
        else:
            break
        fromImage = Image.open(file)

        temp = l.pop()
        i = int(temp % xNum)
        j = int(temp / xNum)
        out = fromImage.resize((xSize, ySize), Image.ANTIALIAS).convert('RGBA')
        toImage.paste(out, (i * xSize, j * ySize))
        toImage = toImage.convert('RGBA')
        #调节透明度
        img = Image.blend(baseImg, toImage, alpha)

        resize = img.resize((600, 600), Image.ANTIALIAS).convert('RGBA')
        img_open = resize
        img_open_backup = resize
        # 显示图片
        photo = ImageTk.PhotoImage(resize)
        label_img2.config(image=photo)
        label_img2.image = photo


#确认后关闭
def close(win, label_img2, v):
    global file_di
    global file_dir
    global note_alpha
    alpha = v.get()
    note_alpha = alpha
    win.destroy()
    generate(label_img2, file_di, file_dir, alpha)


#图像合成
def image_compose(label_img2):
    global file_di
    global file_dir
    global note_alpha
    win = tk.Toplevel()
    win.geometry("{}x{}".format(400, 200))  # 窗口的大小，中间为x
    #提示标签
    v = tk.DoubleVar()
    #设置滑动条默认值
    v.set(note_alpha)
    #透明度滑动条
    sa = tk.Scale(win, from_=0, to=1, orient=tk.HORIZONTAL, variable=v, resolution=0.01)
    l1 = tk.Label(win, text='请选择合成底图')
    l2 = tk.Label(win, text='请选择合成文件夹')
    l3 = tk.Label(win, text='调节透明度')
    e1 = tk.Entry(win, font=('Arial', 10), width=30)
    e2 = tk.Entry(win, font=('Arial', 10), width=30)
    e1.delete(0, "end")
    e1.insert(0, file_di)
    e2.delete(0, "end")
    e2.insert(0, file_dir)
    b1 = tk.Button(win, text='...', font=('Arial', 10), width=3, height=1, command=lambda: choose_file_to_compose(e1))
    # 选择合成文件夹按钮
    b2 = tk.Button(win, text='...', font=('Arial', 10), width=3, height=1, command=lambda: choose_dir_to_compose(e2))
    # 确认后关闭的按钮
    b3 = tk.Button(win, text='确定', font=('华文行楷', 13), width=10, height=1, command=lambda: close(win, label_img2, v))

    sa.place(relx=0.5, rely=0.6, anchor='c')
    l1.place(relx=0.3, rely=0.25, anchor='e')
    l2.place(relx=0.3, rely=0.45, anchor='e')
    l3.place(relx=0.35, rely=0.65, anchor='e')
    e1.place(relx=0.3, rely=0.2)
    e2.place(relx=0.3, rely=0.4)
    b1.place(relx=0.9, rely=0.25, anchor='c')
    b2.place(relx=0.9, rely=0.45, anchor='c')
    b3.place(relx=0.5, rely=0.85, anchor='center')

    win.mainloop()


#显示出反转函数
def show_reserve(c):
    # 显示出变换函数
    x = np.arange(0, c, 0.01)
    y = c - x
    plt.plot(x, y, 'r', linewidth=1)
    plt.title('反转变换函数')
    plt.xlim(0, 256)
    path = 'gray_reserve.jpg'
    plt.savefig(path)
    img = Image.open(path)
    win = tk.Toplevel()
    la = tk.Label(win)
    # 显示图片
    photo = ImageTk.PhotoImage(img)
    la.config(image=photo)
    la.image = photo
    la.pack()
    # 读取之后删除痕迹
    os.remove(path)
    win.mainloop()



#图像的灰度反转
def img_reserve(label_img):
    global img_open
    global img_open_backup
    img = np.asarray(img_open_backup)
    img = rgb2gray(img)
    rows, cols = img.shape
    max1 = img.max()
    emptyImage = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            emptyImage[i, j] = max1 - img[i, j]
    img = Image.fromarray(np.array(emptyImage))
    img_open = img
    # tk中打开图片
    img_png = ImageTk.PhotoImage(img)
    # 将标签的图片选项设为显示这张图
    label_img.config(image=img_png)
    label_img.image = img_png
    show_reserve(max1)


#显示出对数函数
def show_log(c):
    # 显示出变换函数
    x = np.arange(0.01, 256, 0.01)
    y = c * np.log(1 + x)
    plt.plot(x, y, 'r', linewidth=1)
    plt.title('对数变换函数')
    plt.xlim(0, 256)
    path = 'gray_log.jpg'
    plt.savefig(path)
    img = Image.open(path)
    win = tk.Toplevel()
    la = tk.Label(win)
    # 显示图片
    photo = ImageTk.PhotoImage(img)
    la.config(image=photo)
    la.image = photo
    la.pack()
    # 读取之后删除痕迹
    os.remove(path)
    win.mainloop()


#计算显示对数函数变换
def gray_log(label_img):
    global img_open
    global img_open_backup
    c = tkinter.simpledialog.askfloat('对数变换参数', '请输入对数变换的参数', initialvalue=1)
    img = np.asarray(img_open_backup)
    img = rgb2gray(img)
    rows, cols = img.shape
    emptyImage = np.zeros((rows, cols))
    max1 = img.max()
    for i in range(rows):
        for j in range(cols):
            r = img[i, j]
            # 重新量化
            emptyImage[i, j] = c * (math.log(1 + r))
    max2 = emptyImage.max()
    for i in range(rows):
        for j in range(cols):
            # 归一化再量化
            emptyImage[i, j] = emptyImage[i, j] / max2 * max1
    img = Image.fromarray(np.array(emptyImage))
    img_open = img
    # tk中打开图片
    img_png = ImageTk.PhotoImage(img)
    # 将标签的图片选项设为显示这张图
    label_img.config(image=img_png)
    label_img.image = img_png
    return c



#图像的对数灰度变换
def img_log(label_img):
    try:
        c = gray_log(label_img)
        show_log(c)
    except:
        pass


#展示gamma图像
def show_gamma(gamma):
    # 显示出变换函数
    x = np.arange(0.01, 256, 0.01)
    y = (x+1) ** gamma
    plt.plot(x, y, 'r', linewidth=1)
    plt.title('伽马变换函数')
    plt.xlim(0, 256)
    path = 'gamma_log.jpg'
    plt.savefig(path)
    img = Image.open(path)
    win = tk.Toplevel()
    la = tk.Label(win)
    # 显示图片
    photo = ImageTk.PhotoImage(img)
    la.config(image=photo)
    la.image = photo
    la.pack()
    # 读取之后删除痕迹
    os.remove(path)
    win.mainloop()


#图像的伽马变换
def img_gamma(label_img):
    try:
        global img_open
        global img_open_backup
        c = 1
        gamma = tkinter.simpledialog.askfloat('伽马变换参数', '请输入伽马系数', initialvalue=1)
        img = np.asarray(img_open_backup)
        img = rgb2gray(img)
        rows, cols = img.shape
        emptyImage = np.zeros((rows, cols))
        max1 = img.max()
        for i in range(rows):
            for j in range(cols):
                r = img[i, j]
                emptyImage[i, j] = math.pow((c + r) / max1, gamma) * max1
        img = Image.fromarray(np.array(emptyImage))
        img_open = img
        # tk中打开图片
        img_png = ImageTk.PhotoImage(img)
        # 将标签的图片选项设为显示这张图
        label_img.config(image=img_png)
        label_img.image = img_png
        show_gamma(gamma)
    except:
        pass


#得到灰度直方图的值
def get_gray_hist():
    global img_open
    img = np.asarray(img_open)
    img = rgb2gray(img)
    imgInfo = img.shape
    h = imgInfo[0]
    w = imgInfo[1]
    gray_hist = np.zeros([256], np.uint64)
    for i in range(h):
        for j in range(w):
            gray_hist[img[i][j]] += 1
    return gray_hist


#直方图计算
def hist_compute(label_img):
    #得到灰度直方的值
    gray_hist = get_gray_hist()
    plt.bar(range(256), gray_hist)
    #将直方图保存为图片以便读取成PIL对象
    plt.show()


#直方图均衡化
def hist_balanced(label_img):
    global img_open
    global img_open_backup
    img = np.asarray(img_open)
    img = rgb2gray(img)
    h, w = img.shape[:2]
    # 计算灰度直方图
    gray_hist = get_gray_hist()
    # 计算累加灰度直方图
    cumulative_hist = np.zeros([256], np.uint64)
    for p in range(256):
        if p == 0:
            cumulative_hist[p] = gray_hist[p]
        else:
            cumulative_hist[p] = cumulative_hist[p - 1] + gray_hist[p]
    # 根据累加灰度直方图得到输入灰度级和输出灰度级之间的映射关系
    relative_map = np.zeros([256], np.uint8)
    coff = 256.0 / (h * w)
    for p in range(256):
        q = coff * float(cumulative_hist[p]) - 1
        if q < 0:
            relative_map[p] = 0
        else:
            relative_map[p] = math.floor(q)

    img_result = np.zeros(img.shape, np.uint8)
    h, w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            img_result[i][j] = relative_map[img[i][j]]
    img = Image.fromarray(img_result)
    img_open = img
    img_open_backup = img
    photo = ImageTk.PhotoImage(img)
    label_img.config(image=photo)
    label_img.image = photo


#将RGB图像转为灰度图像
#采用的比例为MATLAB内置函数比例
def rgb2gray(rgb):
    if np.ndim(rgb) == 2:
        return rgb
    else:
        gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
        return gray.astype(np.int)


#均值滤波关闭窗口响应事件
def av_close(win, v, label_img):
    num = v.get()
    win.destroy()
    average_filter(num, label_img)


#均值滤波的弹出窗口
def img_av_win(label_img):
    win = tk.Toplevel()
    win.geometry("{}x{}".format(400, 200))  # 窗口的大小，中间为x
    #设置提示标签
    l1 = tk.Label(win, text='轻度')
    l1.place(relx=0.35, rely=0.5, anchor='e')
    l1 = tk.Label(win, text='深度')
    l1.place(relx=0.65, rely=0.5, anchor='w')
    # 设置单选框的数字
    v = tk.IntVar()
    sa = tk.Scale(win, from_=1, to=15, orient=tk.HORIZONTAL, variable=v)
    sa.place(relx=0.5, rely=0.45, anchor='c')
    b = tk.Button(win, text='确定', font=('华文行楷', 18), width=12, height=1, command=lambda: av_close(win, v, label_img))
    b.place(relx=0.5, rely=0.9, anchor='c')
    win.mainloop()


#图像的均值滤波
def average_filter(num, label_img):
    global img_open
    global img_open_backup
    img = np.asarray(img_open_backup)
    G = num
    imgInfo = img.shape
    high = imgInfo[0]
    width = imgInfo[1]
    C = imgInfo[2]
    # 创建空白模板，其中np.uint8代表图片的数据类型0-255
    newimg = np.zeros((high, width, 3), np.uint8)
    newH = int(high / G)
    newW = int(width / G)
    for y in range(newH):
        for x in range(newW):
            for c in range(C):
                newimg[G * y:G * (y + 1), G * x:G * (x + 1), c] = np.mean(
                    img[G * y:G * (y + 1), G * x:G * (x + 1), c]).astype(np.int)
    #将矩阵转为图像
    img = Image.fromarray(newimg)
    img_open = img
    # 显示图片
    photo = ImageTk.PhotoImage(img)
    label_img.config(image=photo)
    label_img.image = photo


#中值滤波关闭窗口响应事件
def me_close(win, v, label_img):
    num = v.get()
    win.destroy()
    median_filter(num, label_img)


#中值滤波的弹出窗口
def img_me_win(label_img):
    win = tk.Toplevel()
    win.geometry("{}x{}".format(400, 200))  # 窗口的大小，中间为x
    #设置提示标签
    l1 = tk.Label(win, text='轻度')
    l1.place(relx=0.35, rely=0.5, anchor='e')
    l1 = tk.Label(win, text='深度')
    l1.place(relx=0.65, rely=0.5, anchor='w')
    # 设置单选框的数字
    v = tk.IntVar()
    sa = tk.Scale(win, from_=1, to=15, orient=tk.HORIZONTAL, variable=v)
    sa.place(relx=0.5, rely=0.45, anchor='c')
    b = tk.Button(win, text='确定', font=('华文行楷', 18), width=12, height=1, command=lambda: me_close(win, v, label_img))
    b.place(relx=0.5, rely=0.9, anchor='c')
    win.mainloop()



#图像的中值滤波
def median_filter(num, label_img):
    global img_open
    global img_open_backup
    img = np.asarray(img_open_backup)
    winSize = num
    imgInfo = img.shape
    high = imgInfo[0]
    width = imgInfo[1]
    C = imgInfo[2]
    pad = winSize // 2  # 运算符//取整除 - 返回商的整数部分（向下取整）
    newimg = np.zeros((high + pad * 2, width + pad * 2, C), dtype=np.float)
    newimg[pad:pad + high, pad:pad + width] = img.copy().astype(np.float)
    tmp = newimg.copy()
    for y in range(high):
        for x in range(width):
            for c in range(C):
                newimg[pad + y, pad + x, c] = np.median(tmp[y:y + winSize, x:x + winSize, c])
    newimg = newimg[pad:pad + high, pad:pad + width].astype(np.uint8)
    # 将矩阵转为图像
    img = Image.fromarray(newimg)
    img_open = img
    # 显示图片
    photo = ImageTk.PhotoImage(img)
    label_img.config(image=photo)
    label_img.image = photo


#空域模糊关闭窗口响应事件
def km_close(win, v, label_img):
    num = v.get()
    win.destroy()
    #因为传入的参数必须为奇数，迫不得已
    num = num * 2 - 1
    airspace_vague(num, label_img)


#空域模糊的弹出窗口
def img_km_win(label_img):
    win = tk.Toplevel()
    win.geometry("{}x{}".format(400, 200))  # 窗口的大小，中间为x
    #设置提示标签
    l1 = tk.Label(win, text='轻度')
    l1.place(relx=0.35, rely=0.5, anchor='e')
    l1 = tk.Label(win, text='深度')
    l1.place(relx=0.65, rely=0.5, anchor='w')
    # 设置单选框的数字
    v = tk.IntVar()
    sa = tk.Scale(win, from_=1, to=15, orient=tk.HORIZONTAL, variable=v)
    sa.place(relx=0.5, rely=0.45, anchor='c')
    b = tk.Button(win, text='确定', font=('华文行楷', 18), width=12, height=1, command=lambda: km_close(win, v, label_img))
    b.place(relx=0.5, rely=0.9, anchor='c')
    win.mainloop()



#图像的空域滤波模糊
def airspace_vague(num, label_img):
    # kernel_size为奇数。sigmax为0时是程序根据公式计算得方差，不为0时代表指定标准差为sigmax
    global img_open
    global img_open_backup
    img = np.asarray(img_open_backup)
    img = rgb2gray(img)
    kernel_size = num
    sigmax = 0
    row, col = img.shape[:2]  # 获得未添加边界前的大小信息
    # 下面产生卷积核
    kernel = np.zeros([kernel_size, kernel_size])
    center = kernel_size // 2  # 整除
    # 计算标准差
    if sigmax == 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
    else:
        sigma = sigmax
    s = 2 * (sigma ** 2)
    sum_val = 0
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            x = i - center
            y = j - center  # center-j也无所谓，反正权重是按到圆心距离算的，而且距离带平方了，正负无所谓，x**2+y**2的值代表了权重。
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / s)
            sum_val += kernel[i, j]
    sum_val = 1 / sum_val
    # 对卷积核归一化，确保卷积核之和为1
    kernel = kernel * sum_val  # 对于np.array来说是对应位置相乘。这里不要用除，最好是用乘以分之一的形式
    # 以上是产生卷积核
    # 计算图像边界需要添加的范围，扩充图像边界使得能遍历到原图像的每一个像素
    addLine = int((kernel_size - 1) / 2)  # 虽然addLine理应是整数，但存储类型是浮点，要转换类型
    img = cv.copyMakeBorder(img, addLine, addLine, addLine, addLine, borderType=cv.BORDER_REPLICATE)
    # 定位未扩充之前图像左上角元素在新图像中的下标索引,这个索引将用来遍历原图像的每一个像素点，相当于指针
    source_x = addLine  # 定义起始位置，即未扩充之前图像左上角元素在新图像中的下标索引
    source_y = addLine  # 定义起始位置，即未扩充之前图像左上角元素在新图像中的下标索引
    # addLine的值是单边添加边界的大小（行数，也是列数），一共四个边添加边界
    # 在添加了边界后的图像中遍历未添加边界时的原像素点，进行滤波
    # 外层控制行，内层控制列
    for delta_x in range(0, row):
        for delta_y in range(0, col):
            img[source_x, source_y] = np.sum(
                img[source_x - addLine:source_x + addLine + 1, source_y - addLine:source_y + addLine + 1] * kernel)
            source_y = source_y + 1
        source_x = source_x + 1  # 行加1，准备进行下行的所有列的遍历
        source_y = addLine  # 把列归位到原始的列起点准备下轮列遍历
    # 经过上面的循环后，图像已经滤波完成了
    # 剥除四边添加的边界，然后返回滤波后的图片
    img = img[addLine:row + addLine, addLine:col + addLine]
    img = Image.fromarray(img)
    img_open = img
    # 显示图片
    photo = ImageTk.PhotoImage(img)
    label_img.config(image=photo)
    label_img.image = photo



#图像的空域滤波锐化
def airspase_sharpen(label_img):
    global img_open
    global img_open_backup
    img = np.asarray(img_open_backup)
    img = rgb2gray(img)
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    newimg = np.zeros((height, width), np.uint8)
    for i in range(2, height - 1):
        for j in range(2, width - 1):
            newimg[i, j] = abs(img[i + 1, j + 1] - img[i, j]) + abs(img[i + 1, j] - img[i, j + 1])
    #    plt.imshow(newimg, cmap=plt.get_cmap('gray'))
    img = Image.fromarray(newimg)
    img_open = img
    # 显示图片
    photo = ImageTk.PhotoImage(img)
    label_img.config(image=photo)
    label_img.image = photo


#频域锐化关闭窗口响应事件
def pr_close(win, v, label_img):
    num = v.get()
    win.destroy()
    frequency_sharpen(num, label_img)


#频域锐化的弹出窗口
def img_pr_win(label_img):
    win = tk.Toplevel()
    win.geometry("{}x{}".format(400, 200))  # 窗口的大小，中间为x
    #设置提示标签
    l1 = tk.Label(win, text='轻度')
    l1.place(relx=0.35, rely=0.5, anchor='e')
    l1 = tk.Label(win, text='深度')
    l1.place(relx=0.65, rely=0.5, anchor='w')
    # 设置单选框的数字
    v = tk.IntVar()
    sa = tk.Scale(win, from_=1, to=15, orient=tk.HORIZONTAL, variable=v)
    sa.place(relx=0.5, rely=0.45, anchor='c')
    b = tk.Button(win, text='确定', font=('华文行楷', 18), width=12, height=1, command=lambda: pr_close(win, v, label_img))
    b.place(relx=0.5, rely=0.9, anchor='c')
    win.mainloop()


#得到图像的频域滤波锐化矩阵
def pinyurh(num):
    global img_open_backup
    D = num
    img = np.asarray(img_open_backup)
    img = rgb2gray(img)
    # numpy 中的傅里叶变换
    f1 = np.fft.fft2(img)
    f1_shift = np.fft.fftshift(f1)
    rows, cols = img.shape[:2]
    # 计算频谱中心
    crow, ccol = int(rows / 2), int(cols / 2)
    # 生成rows，cols列的矩阵，数据格式为uint8
    mask = np.zeros((rows, cols), np.uint8)
    # 将距离频谱中心距离小于D的低通信息部分设置为1，属于低通滤波
    for i in range(rows):
        for j in range(cols):
            if np.sqrt(i * i + j * j) <= D:
                mask[crow - D:crow + D, ccol - D:ccol + D] = 1
    mask = 1 - mask
    f1_shift = f1_shift * mask
    # 实现理想高通滤波器
    # 傅里叶逆变换
    f_ishift = np.fft.ifftshift(f1_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = (img_back - np.amin(img_back)) / (np.amax(img_back) - np.amin(img_back))
    return img_back


#频域模糊关闭窗口响应事件
def pm_close(win, v, label_img):
    num = v.get()
    win.destroy()
    frequency_vague(num, label_img)


#频域模糊的弹出窗口
def img_pm_win(label_img):
    win = tk.Toplevel()
    win.geometry("{}x{}".format(400, 200))  # 窗口的大小，中间为x
    #设置提示标签
    l1 = tk.Label(win, text='轻度')
    l1.place(relx=0.35, rely=0.5, anchor='e')
    l1 = tk.Label(win, text='深度')
    l1.place(relx=0.65, rely=0.5, anchor='w')
    # 设置单选框的数字
    v = tk.IntVar()
    sa = tk.Scale(win, from_=1, to=15, orient=tk.HORIZONTAL, variable=v)
    sa.place(relx=0.5, rely=0.45, anchor='c')
    b = tk.Button(win, text='确定', font=('华文行楷', 18), width=12, height=1, command=lambda: pm_close(win, v, label_img))
    b.place(relx=0.5, rely=0.9, anchor='c')
    win.mainloop()


#图像的频域滤波模糊
def frequency_vague(num, label_img):
    D = num
    global img_open
    global img_open_backup
    img = np.asarray(img_open_backup)
    new_img = rgb2gray(img)
    # 傅里叶变换
    f1 = np.fft.fft2(new_img)
    # 使用np.fft.fftshift()函数实现平移，让直流分量输出图像的重心
    f1_shift = np.fft.fftshift(f1)
    # 实现理想低通滤波器
    rows, cols = new_img.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 计算频谱中心
    mask = np.zeros((rows, cols), dtype='uint8')  # 生成rows行，从cols列的矩阵，数据格式为uint8
    # 将距离频谱中心距离小于D的低通信息部分设置为1，属于低通滤波
    for i in range(rows):
        for j in range(cols):
            if np.sqrt(i * i + j * j) <= D:
                mask[crow - D:crow + D, ccol - D:ccol + D] = 1
    f1_shift = f1_shift * mask
    # 傅里叶逆变换
    f_ishift = np.fft.ifftshift(f1_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = (img_back - np.amin(img_back)) / (np.amax(img_back) - np.amin(img_back))
    img_back = img_back * 255
    img = Image.fromarray(img_back)
    img_open = img
    # 显示图片
    photo = ImageTk.PhotoImage(img)
    label_img.config(image=photo)
    label_img.image = photo


#图像的频域滤波锐化
def frequency_sharpen(num, label_img):
    global img_open
    img_back = pinyurh(num)
    img_back = img_back * 255
    img = Image.fromarray(img_back)
    img_open = img
    # 显示图片
    photo = ImageTk.PhotoImage(img)
    label_img.config(image=photo)
    label_img.image = photo


#图像的点检测
def point_detection(label_img):
    global img_open
    global img_open_backup
    img = pinyurh(3)
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    newimg = np.zeros((height, width), np.uint8)
    #    256白
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if (img[i, j].all() != img[i + 1, j].all() and img[i, j].all() != img[i + 1, j + 1].all()
                    and img[i, j].all() != img[i, j + 1].all() and img[i, j].all() != img[i - 1, j].all()
                    and img[i, j].all() != img[i - 1, j - 1].all() and img[i, j].all() != img[i, j - 1].all()
                    and img[i, j].all() != img[i - 1, j + 1].all() and img[i, j].all() != img[i + 1, j - 1].all()):
                newimg[i, j] = 255
    img = Image.fromarray(newimg)
    img_open = img
    # 显示图片
    photo = ImageTk.PhotoImage(img)
    label_img.config(image=photo)
    label_img.image = photo


#滤波函数
def imfilter(gray, w):
    size_gray = gray.shape
    size = w.shape
    #返回的数组,存储滤波后的矩阵
    s = np.zeros(size_gray, dtype=float)
    #找到滤波模板的中心
    center_y = size[0] // 2
    center_x = size[1] // 2
    #模2等于0就从上一个开始
    if size[0] % 2 == 0:
        center_y -= 1
    if size[1] % 2 == 0:
        center_x -= 1
    for i in range(size_gray[0]):
        for j in range(size_gray[1]):
            # 对矩阵中每一个元素重合滤波矩阵中心进行滤波
            a = 0
            for m in range(size[0]):
                for n in range(size[1]):
                    #判断当前重合元素在矩阵中是否存在
                    if (0 <= (i + m - center_y) < size_gray[0]) and (0 <= (j + n - center_x) < size_gray[1]):
                        a = a + gray[i + m - center_y][j + n - center_x] * w[m][n]
            s[i][j] = a
    return s


#图像的线检测
def line_detection(label_img):
    global img_open
    global img_open_backup
    im = np.asarray(img_open_backup)
    gray = rgb2gray(im)
    # 创建水平线检测模板
    w = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
    # 用水平模板进行滤波
    g = imfilter(gray, w)
    #矩阵转图像
    img = Image.fromarray(g)
    img_open = img
    # 显示图片
    photo = ImageTk.PhotoImage(img)
    label_img.config(image=photo)
    label_img.image = photo


#Sobel检测关闭窗口响应事件
def sobel_close(win, v, label_img):
    num = v.get()
    win.destroy()
    sobel_detection(num, label_img)


#Sobel检测的弹出窗口
def img_sobel_win(label_img):
    win = tk.Toplevel()
    win.geometry("{}x{}".format(400, 200))  # 窗口的大小，中间为x
    #设置提示标签
    l1 = tk.Label(win, text='轻度')
    l1.place(relx=0.35, rely=0.5, anchor='e')
    l1 = tk.Label(win, text='深度')
    l1.place(relx=0.65, rely=0.5, anchor='w')
    l3 = tk.Label(win, text='调节Sobel算子的阈值', font=('华文行楷', 20))
    l3.place(relx=0.5, rely=0.2, anchor='c')
    # 设置单选框的数字
    v = tk.DoubleVar()
    sa = tk.Scale(win, from_=0, to=1, orient=tk.HORIZONTAL, variable=v, resolution=0.01)
    sa.place(relx=0.5, rely=0.45, anchor='c')
    b = tk.Button(win, text='确定', font=('华文行楷', 18), width=12, height=1, command=lambda: sobel_close(win, v, label_img))
    b.place(relx=0.5, rely=0.9, anchor='c')
    win.mainloop()


#图像的Sobel边缘检测
def sobel_detection(num, label_img):
    global img_open
    global img_open_backup
    image = np.asarray(img_open_backup)
    image = rgb2gray(image)
    #得到图像的长宽
    rows = image.shape[0]
    cols = image.shape[1]
    image1 = np.zeros(image.shape)
    image2 = np.zeros(image.shape)
    image3 = np.zeros(image.shape)
    # 创建两个滤波的Sobel模板
    m1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    m2 = np.array([[-1, -0, 1], [-2, 0, 2], [-1, 0, 1]])
    # 将图片对两模板分别进行滤波
    image1 = imfilter(image, m1)
    image2 = imfilter(image, m2)
    # 求两滤波后的各像素平方
    image1 = np.square(image1)
    image2 = np.square(image2)
    # 将两个模板进行滤波后的结果相加再取平方根
    image3 = image2 + image1
    image3 = np.sqrt(image3)
    #归255化
    max = image3.max()
    for i in range(rows):
        for j in range(cols):
            image3[i][j] = image3[i][j] / max * 255
    image3 = im2bw(image3, num * 255)
    # 矩阵转图像
    img = Image.fromarray(image3)
    img_open = img
    # 显示图片
    photo = ImageTk.PhotoImage(img)
    label_img.config(image=photo)
    label_img.image = photo


#prewitt检测关闭窗口响应事件
def prewitt_close(win, v, label_img):
    num = v.get()
    win.destroy()
    prewitt_detection(num, label_img)


#prewitt检测的弹出窗口
def img_prewitt_win(label_img):
    win = tk.Toplevel()
    win.geometry("{}x{}".format(400, 200))  # 窗口的大小，中间为x
    #设置提示标签
    l1 = tk.Label(win, text='轻度')
    l1.place(relx=0.35, rely=0.5, anchor='e')
    l1 = tk.Label(win, text='深度')
    l1.place(relx=0.65, rely=0.5, anchor='w')
    l3 = tk.Label(win, text='调节Prewitt算子的阈值', font=('华文行楷', 20))
    l3.place(relx=0.5, rely=0.2, anchor='c')
    # 设置单选框的数字
    v = tk.DoubleVar()
    sa = tk.Scale(win, from_=0, to=1, orient=tk.HORIZONTAL, variable=v, resolution=0.01)
    sa.place(relx=0.5, rely=0.45, anchor='c')
    b = tk.Button(win, text='确定', font=('华文行楷', 18), width=12, height=1, command=lambda: prewitt_close(win, v, label_img))
    b.place(relx=0.5, rely=0.9, anchor='c')
    win.mainloop()


#图像的Prewitt边缘检测
def prewitt_detection(num, label_img):
    global img_open
    global img_open_backup
    image = np.asarray(img_open_backup)
    image = rgb2gray(image)
    #得到图像的长宽
    rows = image.shape[0]
    cols = image.shape[1]
    image1 = np.zeros(image.shape)
    image2 = np.zeros(image.shape)
    image3 = np.zeros(image.shape)
    # 创建两个滤波的Sobel模板
    m1 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    m2 = np.array([[-1, -0, 1], [-1, 0, 1], [-1, 0, 1]])
    # 将图片对两模板分别进行滤波
    image1 = imfilter(image, m1)
    image2 = imfilter(image, m2)
    # 求两滤波后的各像素平方
    image1 = np.square(image1)
    image2 = np.square(image2)
    # 将两个模板进行滤波后的结果相加再取平方根
    image3 = image2 + image1
    image3 = np.sqrt(image3)
    #归255化
    max = image3.max()
    for i in range(rows):
        for j in range(cols):
            image3[i][j] = image3[i][j] / max * 255
    image3 = im2bw(image3, num * 255)
    # 矩阵转图像
    img = Image.fromarray(image3)
    img_open = img
    # 显示图片
    photo = ImageTk.PhotoImage(img)
    label_img.config(image=photo)
    label_img.image = photo



#Roberts检测关闭窗口响应事件
def roberts_close(win, v, label_img):
    num = v.get()
    win.destroy()
    roberts_detection(num, label_img)


#Roberts检测的弹出窗口
def img_roberts_win(label_img):
    win = tk.Toplevel()
    win.geometry("{}x{}".format(400, 200))  # 窗口的大小，中间为x
    #设置提示标签
    l1 = tk.Label(win, text='轻度')
    l1.place(relx=0.35, rely=0.5, anchor='e')
    l1 = tk.Label(win, text='深度')
    l1.place(relx=0.65, rely=0.5, anchor='w')
    l3 = tk.Label(win, text='调节Prewitt算子的阈值', font=('华文行楷', 20))
    l3.place(relx=0.5, rely=0.2, anchor='c')
    # 设置单选框的数字
    v = tk.DoubleVar()
    sa = tk.Scale(win, from_=0, to=1, orient=tk.HORIZONTAL, variable=v, resolution=0.01)
    sa.place(relx=0.5, rely=0.45, anchor='c')
    b = tk.Button(win, text='确定', font=('华文行楷', 18), width=12, height=1, command=lambda: roberts_close(win, v, label_img))
    b.place(relx=0.5, rely=0.9, anchor='c')
    win.mainloop()


#图像的Roberts边缘检测
def roberts_detection(num, label_img):
    global img_open
    global img_open_backup
    image = np.asarray(img_open_backup)
    image = rgb2gray(image)
    #得到图像的长宽
    rows = image.shape[0]
    cols = image.shape[1]
    image1 = np.zeros(image.shape)
    image2 = np.zeros(image.shape)
    image3 = np.zeros(image.shape)
    # 创建两个滤波的Sobel模板
    m1 = np.array([[-1, 0], [0, 1]])
    m2 = np.array([[0, -1], [1, 0]])
    # 将图片对两模板分别进行滤波
    image1 = imfilter(image, m1)
    image2 = imfilter(image, m2)
    # 求两滤波后的各像素平方
    image1 = np.square(image1)
    image2 = np.square(image2)
    # 将两个模板进行滤波后的结果相加再取平方根
    image3 = image2 + image1
    image3 = np.sqrt(image3)
    #归255化
    max = image3.max()
    for i in range(rows):
        for j in range(cols):
            image3[i][j] = image3[i][j] / max * 255
    image3 = im2bw(image3, num * 255)
    # 矩阵转图像
    img = Image.fromarray(image3)
    img_open = img
    # 显示图片
    photo = ImageTk.PhotoImage(img)
    label_img.config(image=photo)
    label_img.image = photo



#LoG检测关闭窗口响应事件
def log_close(win, v, label_img):
    num = v.get()
    win.destroy()
    log_detection(num, label_img)


#LoG检测的弹出窗口
def img_log_win(label_img):
    win = tk.Toplevel()
    win.geometry("{}x{}".format(400, 200))  # 窗口的大小，中间为x
    #设置提示标签
    l1 = tk.Label(win, text='轻度')
    l1.place(relx=0.35, rely=0.5, anchor='e')
    l1 = tk.Label(win, text='深度')
    l1.place(relx=0.65, rely=0.5, anchor='w')
    l3 = tk.Label(win, text='调节LoG算子的阈值', font=('华文行楷', 20))
    l3.place(relx=0.5, rely=0.2, anchor='c')
    # 设置单选框的数字
    v = tk.DoubleVar()
    sa = tk.Scale(win, from_=0, to=1, orient=tk.HORIZONTAL, variable=v, resolution=0.01)
    sa.place(relx=0.5, rely=0.45, anchor='c')
    b = tk.Button(win, text='确定', font=('华文行楷', 18), width=12, height=1, command=lambda: log_close(win, v, label_img))
    b.place(relx=0.5, rely=0.9, anchor='c')
    win.mainloop()


#图像的LoG边缘检测
def log_detection(num, label_img):
    global img_open
    global img_open_backup
    image = np.asarray(img_open_backup)
    image = rgb2gray(image)
    rows = image.shape[0]
    cols = image.shape[1]
    temp = 0
    image1 = np.zeros(image.shape)
    m1 = np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])
    for i in range(2, rows - 2):
        for j in range(2, cols - 2):
            temp = np.abs(
                (np.dot(np.array([1, 1, 1, 1, 1]), (m1 * image[i - 2:i + 3, j - 2:j + 3])))
                    .dot(np.array([[1], [1], [1], [1], [1]])))

            image1[i, j] = int(temp)
    max = image1.max()
    for i in range(rows):
        for j in range(cols):
            image1[i][j] = image1[i][j] / max * 255
    image1 = im2bw(image1, num * 255)
    # 矩阵转图像
    img = Image.fromarray(image1)
    img_open = img
    # 显示图片
    photo = ImageTk.PhotoImage(img)
    label_img.config(image=photo)
    label_img.image = photo



#获取基本阈值处理的全局阈值，传入一个矩阵
def GetBasisThreshold(f):
    t = np.mean(f)
    c = np.zeros(f.shape)
    size = f.shape
    m = size[0]
    n = size[1]
    done = False
    while ~done:
        g = f > t
        count = 0
        #求其中非0元的平均值
        for i in range(m):
            for j in range(n):
                c[i, j] = f[i, j] * g[i, j]
                if c[i, j] != 0:
                    count += 1
        meanfg = np.sum(c) / count
        count = 0
        for i in range(m):
            for j in range(n):
                c[i, j] = f[i, j] * (~g[i, j])
                if c[i, j] != 0:
                    count += 1
        meanf_g = np.sum(c) / count
        Tnext = 0.5 * (meanfg + meanf_g)
        done = abs(t - Tnext) < 0.5
        t = Tnext
    return t


#阈值变换将灰度图变为二值图,传入图像矩阵f与阈值t
def im2bw(f, t):
    c = np.zeros(f.shape)
    size = f.shape
    m = size[0]
    n = size[1]
    #进行阈值分割，大于阈值的为1，小于阈值的为0
    for i in range(m):
        for j in range(n):
            if f[i, j] > t:
                c[i, j] = 255
            else:
                c[i, j] = 0
    return c


#基本阈值分割
def basic_threshold(label_img):
    global img_open
    global img_open_backup
    # 读取图像矩阵
    f = np.asarray(img_open_backup)
    #转为灰度图
    f = rgb2gray(f)
    # 计算基本全局阈值
    t = GetBasisThreshold(f)
    # 阈值变换将灰度图变为二值图
    binarr = im2bw(f, t)
    # 矩阵转图像
    img = Image.fromarray(binarr)
    img_open = img
    # 显示图片
    photo = ImageTk.PhotoImage(img)
    label_img.config(image=photo)
    label_img.image = photo


# 计算Otsu全局阈值
def GetOtsuThreshold(gray):
    #获取图像的大小
    size = gray.shape
    #获取图像高
    m = size[0]
    #获取图像宽
    n = size[1]
    #统计每个像素值的多少
    dic = {}
    for i in range(m):
        for j in range(n):
            if gray[i,j] in dic:
                dic[gray[i,j]] += 1
            else:
                dic[gray[i, j]] = 1
    sum = 0
    for i in range(256):
        if i in dic:
            sum += (dic[i] * i)
    print(sum / (m * n))
    return sum / (m * n)


#Otsu全局分割方法
def otsu_threshold(label_img):
    global img_open
    global img_open_backup
    # 读取图像矩阵
    f = np.asarray(img_open_backup)
    # 转为灰度图
    f = rgb2gray(f)
    # 计算Otsu全局阈值
    t = GetOtsuThreshold(f)
    binarr = im2bw(f, t)
    # 矩阵转图像
    img = Image.fromarray(binarr)
    img_open = img
    # 显示图片
    photo = ImageTk.PhotoImage(img)
    label_img.config(image=photo)
    label_img.image = photo


#获取要压缩的文件名
def choose_file_to_compression():
    try:
        # 1表示打开文件对话框
        dlg = win32ui.CreateFileDialog(1)
        # 设置打开文件对话框中的初始显示目录
        dlg.SetOFNInitialDir('E:/Python')
        dlg.DoModal()
        # 获取选择的文件名称
        filechoose = dlg.GetPathName()
        return filechoose
    except:
        pass


#一个类，保存霍夫曼树的结点
class Node:
    def __init__(self, name, weight):
        # 节点名
        self.name = name
        # 节点权重
        self.weight = weight
        # 节点左孩子
        self.left = None
        # 节点右孩子
        self.right = None
        # 节点父节点
        self.father = None
    #判断是否是左孩子
    def is_left_child(self):
        return self.father.left == self


#创建最初的叶子节点
def create_prim_nodes(data_set, labels):
    if(len(data_set) != len(labels)):
        raise Exception('数据和标签不匹配!')
    nodes = []
    for i in range(len(labels)):
        nodes.append( Node(labels[i],data_set[i]) )
    return nodes


# 创建huffman树
def create_HF_tree(nodes):
    #此处注意，copy()属于浅拷贝，只拷贝最外层元素，内层嵌套元素则通过引用，而不是独立分配内存
    tree_nodes = nodes.copy()
    while len(tree_nodes) > 1: #只剩根节点时，退出循环
        tree_nodes.sort(key=lambda node: node.weight)#升序排列
        new_left = tree_nodes.pop(0)
        new_right = tree_nodes.pop(0)
        new_node = Node(None, (new_left.weight + new_right.weight))
        new_node.left = new_left
        new_node.right = new_right
        new_left.father = new_right.father = new_node
        tree_nodes.append(new_node)
    tree_nodes[0].father = None #根节点父亲为None
    return tree_nodes[0] #返回根节点


#获取huffman编码
def get_huffman_code(nodes):
    codes = {}
    for node in nodes:
        code=''
        name = node.name
        while node.father != None:
            if node.is_left_child():
                code = '0' + code
            else:
                code = '1' + code
            node = node.father
        codes[name] = code
    return codes


#得到哈夫曼编码的字典
def GetHuffmanDict(file):
    # 二进制读取文件
    f = open(file, "rb")
    dict = {}
    while True:
        # 读取一个字节
        a1 = f.read(1)
        if a1:
            # 字节解码为整型
            a = struct.unpack('B', a1)[0]
            if a in dict:
                dict[a] += 1
            else:
                dict[a] = 1
        else:
            break
    f.close()
    return dict


#获取哈夫曼编码并将其写入文件
def GetHuffmanCode(labels, data, file):
    #获取选择文件的文件夹路径
    path = os.path.dirname(file)
    #获取选择文件的文件名
    name = file[len(path)+1:]
    #得到路径和前缀名
    prefile = path + '\\' + name.replace('.', '_')
    # 创建一个文件存储哈夫曼字典
    f = open(prefile + "__dict.txt", "w")
    # 创建初始叶子节点
    nodes = create_prim_nodes(data, labels)
    # 创建huffman树
    tree_root = create_HF_tree(nodes)
    # 获取huffman编码
    codes = get_huffman_code(nodes)
    for i in range(len(data)):
        s = str(labels[i]) + ":" + str(data[i]) + '\n'
        f.write(s)
    f.close()
    return codes, tree_root


#将文件压缩为哈夫曼编码并储存
def TurnToHuffman(code, file):
    # 获取选择文件的文件夹路径
    path = os.path.dirname(file)
    # 获取选择文件的文件名
    name = file[len(path) + 1:]
    # 得到路径和前缀名
    prefile = path + '\\' + name.replace('.', '_')
    # 二进制读取文件
    f = open(file, "rb")
    f1 = open(prefile + '__compress.txt', 'wb')
    s = ''
    remain = 0
    while True:
        # 读取一个字节
        a1 = f.read(1)
        if a1:
            # 字节解码为整型
            a = struct.unpack('B', a1)[0]
            s = s + code[a]
            #当前字符串长度大于8就进行压缩
            if len(s) >= 8:
                while len(s) >= 8:
                    #每次压缩时取出前面8位压缩，同时字符串长度-8
                    a = s[0:8]
                    s = s[8:]
                    #将二进制字符串转为10进制整数
                    b = int(a, 2)
                    #将十进制整数转化为bytes类型
                    c = b.to_bytes(1, byteorder='little', signed=False)
                    f1.write(c)
            #remain记录最后剩下的字符长度
            remain = len(s)
        else:
            break
    f.close()
    #如果最后还剩下小于8的二进制
    if remain > 0:
        a = s
        #将位数左移，同时保留remain
        for i in range(len(a), 8):
            a = a + '0'
        b = int(a, 2)
        c = b.to_bytes(1, byteorder='little', signed=False)
        f1.write(c)
    f1.close()
    #在字典的末尾追加写上最后剩余多少个字符串
    f = open(prefile + "__dict.txt", "a")
    f.write(str(remain))
    f.close()


#哈夫曼编码，传入一个待压缩文件
def huffmanenco(file):
    #获取文件的字节字典
    dict = GetHuffmanDict(file)
    #将字典的键、值转为列表
    labels = list(dict.keys())
    data = list(dict.values())
    #获取哈夫曼编码字典及根结点并将其写进文件
    code, tree_root = GetHuffmanCode(labels, data, file)
    # 将文件压缩为哈夫曼编码并储存,返回剩下的落单二进制
    TurnToHuffman(code, file)


#文件压缩
def file_compression():
    #获取要压缩的文件名
    file_to_compression = choose_file_to_compression()
    # 对一个文件哈夫曼压缩
    huffmanenco(file_to_compression)


#霍夫曼文件的解压
def SecureCompress(remain, dict, tree_root, file):
    re_dic = {}
    # 获取选择文件的文件夹路径
    path = os.path.dirname(file)
    # 获取选择文件的文件名
    name = file[len(path) + 1:]
    #获取压缩前的文件名
    sname = name.split('__')[0]
    sname = sname.split('_')
    s_path = path + '\\' + sname[0] + '_decompression.' + sname[1]

    #生成反向字典,即编码对数的索引
    for key, value in dict.items():
        re_dic[value] = key
    fya = open(file, 'rb')
    f = open(s_path, 'w')
    list_key = list(re_dic.keys())
    #记录比较的字符串
    com = ''
    while True:
        #读入一个字节
        a = fya.read(1)
        if a:
            #字节转整数
            b = int.from_bytes(a, byteorder='big', signed=False)
            #整数转二进制
            c = bin(b)[2:]
            #前面用0填充转为8位二进制
            if len(c) < 8:
                for i in range(len(c),8):
                    c = '0' + c
            com = com + c
            i = 0
            temp = ''
            start = tree_root
            #设定长度大于16才能进入解码是为了保证出循环得到的com中含有最后落单的几位
            while  len(com) > 16 and i < len(com):
                if com[i] == '0':
                    temp += '0'
                    start = start.left
                else:
                    temp += '1'
                    start = start.right
                i += 1
                #如果是叶子结点就说明找到了一个编码，即进行解码
                if start.right is None and start.left is None:
                    com = com[i:]
                    i = 0
                    start = tree_root
                    #字节转为字符串写入
                    f.write(re_dic[temp].to_bytes(1, byteorder='little', signed=False).decode())
                    temp = ''
        else:
            break
    #将最后的八位分割开来，得到落单的串再合并
    if remain > 0:
        a = com[-8:]
        com = com[:-8]
        a = a[:remain]
        com = com + a
    #对最后的串解析
    while True:
        i = 0
        temp = ''
        start = tree_root
        while i < len(com):
            if com[i] == '0':
                temp += '0'
                start = start.left
            else:
                temp += '1'
                start = start.right
            i += 1
            if start.right is None and start.left is None:
                com = com[i:]
                i = 0
                start = tree_root
                f.write(re_dic[temp].to_bytes(1, byteorder='little', signed=False).decode())
                temp = ''
        if len(com) == 0:
            break
    fya.close()
    f.close()
    s =  path + '\\' + name.split('__')[0] + '__compression.' + 'txt'
    os.remove(path + '\\' + name.split('__')[0] + '__compress.' + 'txt')
    os.remove(path + '\\' + name.split('__')[0] + '__dict.' + 'txt')


#得到文件的霍夫曼字典
def get_file_code(file):
    #压缩的字典
    dict = {}
    # 获取选择文件的文件夹路径
    path = os.path.dirname(file)
    # 获取选择文件的文件名
    name = file[len(path) + 1:]
    remain = 0
    s_name = name.split('__')[0]
    # 得到路径和前缀名
    dict_file = path + '\\' + s_name + '__dict.txt'
    f = open(dict_file, "r")
    for line in f:
        if len(line) == 1:
            remain = int(line)
        else:
            s = line.split(':')
            dict[int(s[0])] = int(s[1])
    return dict, remain


#文件解压
def file_decompression():
    # 获取要解压的文件名
    file_to_compression = choose_file_to_compression()
    #获取欲解压文件的code
    dic, remain = get_file_code(file_to_compression)
    # 将字典的键、值转为列表
    labels = list(dic.keys())
    data = list(dic.values())
    # 创建初始叶子节点
    nodes = create_prim_nodes(data, labels)
    # 创建huffman树
    tree_root = create_HF_tree(nodes)
    # 获取huffman编码
    code = get_huffman_code(nodes)
    # 将压缩文件解压
    SecureCompress(remain, code, tree_root, file_to_compression)


#灰度级量化压缩关闭窗口响应事件
def igs_close(win, v, label_img):
    num = v.get()
    win.destroy()
    igs(num, label_img)


#灰度级量化压缩的弹出窗口
def img_igs_win(label_img):
    win = tk.Toplevel()
    win.geometry("{}x{}".format(400, 200))  # 窗口的大小，中间为x
    #设置提示标签
    l1 = tk.Label(win, text='轻度')
    l1.place(relx=0.35, rely=0.5, anchor='e')
    l1 = tk.Label(win, text='深度')
    l1.place(relx=0.65, rely=0.5, anchor='w')
    # 设置单选框的数字
    v = tk.IntVar()
    sa = tk.Scale(win, from_=8, to=0, orient=tk.HORIZONTAL, variable=v)
    sa.place(relx=0.5, rely=0.45, anchor='c')
    b = tk.Button(win, text='确定', font=('华文行楷', 18), width=12, height=1, command=lambda: igs_close(win, v, label_img))
    b.place(relx=0.5, rely=0.9, anchor='c')
    win.mainloop()


#计算客观保真度,传入一个原图像矩阵，一个压缩图像矩阵,传入为numpy.arraty类型
def ObSave(origin,compress):
    #计算均方根误差
    #将俩矩阵转为double类型，防止超出类型大小
    origin = np.double(origin)
    compress = np.double(compress)
    size = origin.shape
    c = compress - origin
    #求矩阵各个元素平方
    c = np.square(c)
    #求各维度的乘积
    demision = 1
    for i in size:
        demision *= i
    #求均方根误差
    e = math.sqrt((1 / demision) * np.sum(c))

    #计算均方信噪比
    if np.sum(c) == 0:
        SNR = 0
    else:
        vb = np.sum(compress)
        b = np.square(compress)
        m = np.sum(np.square(compress))
        v = np.sum(c)
        SNR = (np.sum(np.square(compress)) / np.sum(c))
    return e,SNR


#IGS量化压缩
def igs(b, label_img):
    global img_open
    global img_open_backup
    # IGS扰动方法：对于任何高位不是hi的像素，将其加上前一列相邻像素的对应lo低位的值
    # lo有8-b位1在低位，其余位为0
    #设置压缩比特位为4
    lo = 2 ** (8 - b) - 1
    lo = np.uint8(lo)
    # hi有b位1在高位，其余位为0
    hi = 2 ** (8) - lo - 1
    hi = np.uint8(hi)
    # 读取图像矩阵
    f = np.asarray(img_open_backup)
    # 转为灰度图
    gray = rgb2gray(f)
    # 获得对应于x中高位不是hi的那些元素的集合
    size = gray.shape
    m = size[0]
    n = size[1]
    hitest = np.zeros(gray.shape)
    for i in range(m):
        for j in range(n):
            hitest[i, j] = ((gray[i, j] & hi) != hi)
    y = np.zeros(gray.shape)
    s = np.zeros([m, 1])
    tt = np.zeros([m, 1])
    s = np.uint8(s)
    tt = np.uint8(tt)
    y = np.uint8(y)
    gray = np.uint8(gray)
    hitest = np.uint8(hitest)
    nn = np.zeros([m, 1])
    nn = np.uint8(nn)
    for i in range(n):
        # tt为经IGS扰动后的X中前一列像素的对应于lo低位的值
        tt = s & lo
        # 经扰动后的当前列的像素值
        for j in range(m):
            nn[j] = hitest[j, i] * tt[j]
        for j in range(m):
            s[j] = gray[j, i] + nn[j]
        # 保留当前列像素灰度的高位值
        for j in range(m):
            y[j, i] = s[j] & hi
    #计算客观保真度
    e, SNR = ObSave(gray, y)
    tkinter.messagebox.showinfo('客观保真度', '此次灰度量化压缩信息如下\n均方根误差为{:.2f}\n均方信噪比为{:.2f}'.format(e, SNR))
    # 矩阵转图像
    img = Image.fromarray(y)
    img_open = img
    # 显示图片
    photo = ImageTk.PhotoImage(img)
    label_img.config(image=photo)
    label_img.image = photo


#图像的默认模式
def defaulted(label_img):
    global img_open_backup
    global img_open
    img_open = img_open_backup
    # tk中打开图片
    img_png = ImageTk.PhotoImage(img_open)
    # 将标签的图片选项设为显示这张图
    label_img.config(image=img_png)
    label_img.image = img_png


#图像的黑白照
def black_white(label_img):
    global img_open_backup
    global img_open
    im = np.asarray(img_open_backup.convert('RGB'))
    trans = np.array([[0.299, 0.587, 0.114], [0.299, 0.587, 0.114], [0.299, 0.587, 0.114]]).transpose()
    im = np.dot(im, trans)
    img = Image.fromarray(np.array(im).astype('uint8'))
    img_open = img
    # tk中打开图片
    img_png = ImageTk.PhotoImage(img)
    # 将标签的图片选项设为显示这张图
    label_img.config(image=img_png)
    label_img.image = img_png


#图像的流年模式
def liunian(label_img):
    global img_open_backup
    global img_open
    im = np.asarray(img_open_backup.convert('RGB'))
    im1 = np.sqrt(im * [1.0, 0.0, 0.0]) * 12
    im2 = im * [0.0, 1.0, 1.0]
    im = im1 + im2
    img = Image.fromarray(np.array(im).astype('uint8'))

    img_open = img
    # tk中打开图片
    img_png = ImageTk.PhotoImage(img)
    # 将标签的图片选项设为显示这张图
    label_img.config(image=img_png)
    label_img.image = img_png


#图像的旧电影模式
def old_movie(label_img):
    global img_open_backup
    global img_open
    im = np.asarray(img_open_backup.convert('RGB'))
    # r=r*0.393+g*0.769+b*0.189 g=r*0.349+g*0.686+b*0.168 b=r*0.272+g*0.534b*0.131
    trans = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]).transpose()
    # clip 超过255的颜色置为255
    im = np.dot(im, trans).clip(max=255)
    img = Image.fromarray(np.array(im).astype('uint8'))

    img_open = img
    # tk中打开图片
    img_png = ImageTk.PhotoImage(img)
    # 将标签的图片选项设为显示这张图
    label_img.config(image=img_png)
    label_img.image = img_png


#图像的反色模式
def reverse_color(label_img):
    global img_open_backup
    global img_open
    im = 255 - np.asarray(img_open_backup.convert('RGB'))
    img = Image.fromarray(np.array(im).astype('uint8'))

    img_open = img
    # tk中打开图片
    img_png = ImageTk.PhotoImage(img)
    # 将标签的图片选项设为显示这张图
    label_img.config(image=img_png)
    label_img.image = img_png


#预测编码，传入一个图像矩阵
def LPCencode(gray):
    size = gray.shape
    for i in range(size[0]):
        for j in range(size[1] - 1, 0 , -1):
            gray[i, j] = gray[i, j] - gray[i, j - 1]
    return gray


#预测解码，传入有一个解码矩阵
def LPCdecode(pre, path):
    size = pre.shape
    for i in range(size[0]):
        for j in range(1, size[1]):
            pre[i, j] = pre[i, j-1] + pre[i, j]
    # 写入解码后文件
    f = open(path + '\\' + '解码后矩阵.txt', 'w')
    m, n = pre.shape
    f.write(str(m) + " " + str(n) + '\n')
    for i in range(m):
        for j in range(n):
            f.write(str(pre[i][j]) + " ")
        f.write('\n')
    f.close()


#获取lpc存储的路径
def choose_file_lpc():
    try:
        return filedialog.askdirectory()  # 1表示打开文件对话框
    except:
        pass


#前值预测编码
def LPCbian():
    global img_open
    im = np.asarray(img_open)
    # 转化为灰度图
    gray = rgb2gray(im)
    #获取来lpc的存储路径
    file_path = choose_file_lpc()
    #将编码前的矩阵写进
    f = open(file_path + '\\' + '编码前矩阵.txt','w')
    m, n = gray.shape
    f.write(str(m) + " " + str(n) + '\n')
    for i in range(m):
        for j in range(n):
            f.write(str(gray[i][j]) + " ")
        f.write('\n')
    f.close()
    # 前值预测编码
    pre = LPCencode(gray)
    #写入编码后的矩阵
    f = open(file_path + '\\' + '编码后矩阵.txt', 'w')
    m, n = pre.shape
    f.write(str(m) + " " + str(n) + '\n')
    for i in range(m):
        for j in range(n):
            f.write(str(pre[i][j]) + " ")
        f.write('\n')
    f.close()


def Gojie():
    file_path = filedialog.askopenfilename()
    f = open(file_path, 'r')
    l = f.readline()
    l = l.split()
    s = np.zeros([int(l[0]), int(l[1])],dtype=int)
    i = 0
    j = 0
    while True:
        l = f.readline()
        j = 0
        if l == '':
            break
        line = l.split()
        for c in line:
            s[i][j] = int(c)
            j += 1
        i += 1
    f.close()
    path = os.path.dirname(file_path)
    return s, path


#前值预测解码
def LPCjie():
    #读取要解码文件
    pre, path = Gojie()
    # 前值预测解码
    LPCdecode(pre, path)


#图片信息量的计算
def InformationNum():
    global img_open
    im = np.asarray(img_open)
    # 转化为灰度图
    gray = rgb2gray(im)
    # 将读取出来的数组转化为一维列表方便循环遍历
    gray1 = list(gray.ravel())
    size = gray.shape
    #总的像素
    zong = (size[0] + 1) * (size[1] + 1)
    #统计图中的灰度级别
    obj = dict(collections.Counter(gray1))
    #计算信息量H
    H = 0
    #统计各个灰度级别的概率
    for i in obj.keys():
        obj[i] = obj[i] / zong
        #计算信息量H
        H += -1 * obj[i] * log(obj[i], 2)
    tkinter.messagebox.showinfo('图片信息量', '该图片信息量为{:.2f}'.format(H))


#图片保存模块,传入一张图片
def image_save():
    global img_open
    try:
        # 1表示打开文件对话框
        dlg = win32ui.CreateFileDialog(0)
        # 设置打开文件对话框中的初始显示目录
        dlg.SetOFNInitialDir('E:/Python')
        dlg.DoModal()
        # 获取选择的文件名称
        path = dlg.GetPathName()
        path = path + '.png'
        im = np.asarray(img_open)
        if im.ndim == 2:
            img_open.convert('L').save(path)
        else:
            img_open.save(path)
    except:
        pass


#文件选择，获取文件模块
def choose_file(label_img):   #文件选择框
    global filename
    global img_open_init
    global img_open
    global img_open_backup
    try:
        # 1表示打开文件对话框
        dlg = win32ui.CreateFileDialog(1)
        # 设置打开文件对话框中的初始显示目录
        dlg.SetOFNInitialDir('E:/Python')
        dlg.DoModal()
        # 获取选择的文件名称
        filename = dlg.GetPathName()
        #打开图片
        img_open = Image.open(filename)
        #将图片按比例缩放统一最大为700
        img_open.thumbnail((1000, 700))
        # 获取初始的图片
        img_open_init = img_open
        img_open_backup = img_open
        #tk中打开图片
        img_png = ImageTk.PhotoImage(img_open)
        #将标签的图片选项设为显示这张图
        label_img.config(image=img_png)
        label_img.image = img_png
    except:
        pass


#界面
def Interface():
    global img_x
    global img_y
    w, h = root.maxsize()
    root.geometry("{}x{}".format(w, h))  # 窗口的大小，中间为x
    #设置单选框的数字
    model_number = tk.IntVar()
    # 设置显示图片的标签
    label_img = tk.Label(root)
    label_img.place(relx=img_x, rely=img_y, anchor='c')
    # 创建一个顶级菜单
    menubar = tk.Menu(root)
    # 创建一个下拉菜单“文件”，然后将它添加到顶级菜单中
    filemenu = tk.Menu(menubar, tearoff=False)
    # 文件的打开
    filemenu.add_command(label="   打开      ", command=lambda: choose_file(label_img))
    # 图片的保存
    filemenu.add_command(label="   保存      ", command=image_save)
    # 将菜单文件加入顶级菜单中
    menubar.add_cascade(label="  文件  ", menu=filemenu)

    # 创建一个子菜单“几何变换”，然后将它添加到顶级菜单中
    img_jihe = tk.Menu(menubar, tearoff=False)
    # 图片的上移
    img_jihe.add_command(label="向上平移", command=lambda: image_upward(label_img))
    # 图片的下移
    img_jihe.add_command(label="向下平移", command=lambda: image_down(label_img))
    # 图片的左移
    img_jihe.add_command(label="向左平移", command=lambda: image_turn_left(label_img))
    # 图片的右移
    img_jihe.add_command(label="向右平移", command=lambda: image_turn_right(label_img))
    # 图片的放大
    img_jihe.add_command(label="图片放大", command=lambda: image_magify(label_img))
    # 图片的缩小
    img_jihe.add_command(label="图片缩小", command=lambda: image_shrink(label_img))
    # 图片的旋转
    img_xz = tk.Menu(img_jihe, tearoff=False)
    #顺时针旋转90度
    img_xz.add_command(label="顺时针旋转90度", command=lambda: image_revolt_clockwise(label_img))
    # 逆时针旋转90度
    img_xz.add_command(label="逆时针旋转90度", command=lambda: image_revolt_anticlockwise(label_img))
    # 旋转任意角度
    img_xz.add_command(label="旋转任意角度", command=lambda: image_revolt(label_img))
    img_jihe.add_cascade(label="旋转", menu=img_xz)
    # 图片的翻转
    img_jihe.add_command(label="图片翻转", command=lambda: image_flip(label_img))
    #将其加入主菜单
    menubar.add_cascade(label="  几何变换    ", menu=img_jihe)

    # 创建一个子菜单“像素变换”，然后将它添加到顶级菜单中
    img_xs = tk.Menu(menubar, tearoff=False)
    # 图片的合成
    img_xs.add_command(label="图片合成", command=lambda: image_compose(label_img))

    # 图片的灰度变换
    img_hd = tk.Menu(img_xs, tearoff=False)
    # 顺时针旋转90度
    img_hd.add_command(label="灰度反转", command=lambda: img_reserve(label_img))
    # 逆时针旋转90度
    img_hd.add_command(label="对数变换", command=lambda: img_log(label_img))
    # 旋转任意角度
    img_hd.add_command(label="伽马变换", command=lambda: img_gamma(label_img))
    img_xs.add_cascade(label="灰度变换", menu=img_hd)

    # 图片的直方图显示
    img_xs.add_command(label="直方图显示", command=lambda: hist_compute(label_img))
    # 图片的直方图均衡化
    img_xs.add_command(label="直方图均衡化", command=lambda: hist_balanced(label_img))
    # 将其加入主菜单
    menubar.add_cascade(label="  像素变换    ", menu=img_xs)

    # 创建一个子菜单“图像去噪”，然后将它添加到顶级菜单中
    img_qz = tk.Menu(menubar, tearoff=False)
    # 均值滤波去噪
    img_qz.add_command(label="均值滤波", command=lambda: img_av_win(label_img))
    # 中值滤波去噪
    img_qz.add_command(label="中值滤波", command=lambda: img_me_win(label_img))
    # 将其加入主菜单
    menubar.add_cascade(label="  图像去噪    ", menu=img_qz)

    # 创建一个子菜单“模糊锐化”，然后将它添加到顶级菜单中
    img_mhrh = tk.Menu(menubar, tearoff=False)
    # 图片的空域滤波模糊
    img_mhrh.add_command(label="空域模糊", command=lambda: img_km_win(label_img))
    # 图片空域滤波锐化
    img_mhrh.add_command(label="空域锐化", command=lambda: airspase_sharpen(label_img))
    # 图片的频域滤波模糊
    img_mhrh.add_command(label="频域模糊", command=lambda: img_pm_win(label_img))
    # 图片的频域滤波锐化
    img_mhrh.add_command(label="频域锐化", command=lambda: img_pr_win(label_img))
    # 将其加入主菜单
    menubar.add_cascade(label="  模糊锐化    ", menu=img_mhrh)

    # 创建一个子菜单“边缘检测”，然后将它添加到顶级菜单中
    img_byjc = tk.Menu(menubar, tearoff=False)
    # 图片的点检测
    img_byjc.add_command(label="点检测", command=lambda: point_detection(label_img))
    # 图片的线检测
    img_byjc.add_command(label="线检测", command=lambda: line_detection(label_img))

    # 图片的边缘检测
    img_wo = tk.Menu(img_byjc, tearoff=False)
    # LoG算子检测
    img_wo.add_command(label="LoG边缘检测器", command=lambda: img_log_win(label_img))
    # Sobel边缘检测器
    img_wo.add_command(label="Sobel边缘检测器", command=lambda: img_sobel_win(label_img))
    # Prewitt边缘检测器
    img_wo.add_command(label="Prewitt边缘检测器", command=lambda: img_prewitt_win(label_img))
    # Roberts边缘检测器
    img_wo.add_command(label="Roberts边缘检测器", command=lambda: img_roberts_win(label_img))
    img_byjc.add_cascade(label="边缘检测", menu=img_wo)

    # 将其加入主菜单
    menubar.add_cascade(label="  边缘检测    ", menu=img_byjc)

    # 创建一个子菜单“图像分割”，然后将它添加到顶级菜单中
    img_fg = tk.Menu(menubar, tearoff=False)
    # 基本阈值分割
    img_fg.add_command(label="基本阈值分割", command=lambda: basic_threshold(label_img))
    # Otsu阈值分割
    img_fg.add_command(label="Otsu阈值分割", command=lambda: otsu_threshold(label_img))
    # 将其加入主菜单
    menubar.add_cascade(label="  图像分割    ", menu=img_fg)

    # 创建一个子菜单“图像压缩”，然后将它添加到顶级菜单中
    img_ys = tk.Menu(menubar, tearoff=False)
    # 文件压缩
    img_ys.add_command(label="文件压缩", command=file_compression)
    # 文件解压
    img_ys.add_command(label="文件解压", command=file_decompression)
    # 灰度级量化压缩
    img_ys.add_command(label="灰度级量化压缩", command=lambda: img_igs_win(label_img))
    # 无损的前值预测编码
    img_ys.add_command(label="前值预测编码", command=LPCbian)
    # 无损预测解码
    img_ys.add_command(label="前值预测解码", command=LPCjie)
    # 图像信息量的计算
    img_ys.add_command(label="图像信息量计算", command=InformationNum)
    # 将其加入主菜单
    menubar.add_cascade(label="  图像压缩    ", menu=img_ys)

    #设置显示的模式子菜单
    model = tk.Menu(menubar, tearoff=False)
    # 设置默认显示单选框
    model_number.set(1)
    # 默认模式，即图像原模式
    model.add_radiobutton(label="默认", variable=model_number, value=1, command=lambda: defaulted(label_img))
    # 图像的黑白模式
    model.add_radiobutton(label="黑白", variable=model_number, value=2, command=lambda: black_white(label_img))
    # 图像的流年样式
    model.add_radiobutton(label="流年", variable=model_number, value=3, command=lambda: liunian(label_img))
    # 图像的旧电影样式
    model.add_radiobutton(label="旧电影", variable=model_number, value=4, command=lambda: old_movie(label_img))
    # 图像的反色样式
    model.add_radiobutton(label="反色", variable=model_number, value=5, command=lambda: reverse_color(label_img))
    #加入主菜单
    menubar.add_cascade(label='  样式  ', menu=model)

    # 显示菜单
    root.config(menu=menubar)
    root.mainloop()


if __name__ == "__main__":
    Interface()
