
from queue import Empty
import cv2
import numpy as np
import imutils
from imutils import contours as cont
from collections import defaultdict
import pytesseract
from PIL import ImageFont, ImageDraw, Image, ImageEnhance
import math
import re
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'


from pickle import load
# import tensorflow.compat.v1 as tf  #(it will import tensorflow 1.0 version in your system)
import tensorflow as tf
# from tensorflow.keras.models import Model
from keras.models import load_model
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras import backend as K
#from keras.backend.tensorflow_backend import set_session
from tensorflow.compat.v1.keras.backend import set_session
import numpy as np
import re
import tensorflow
from jiwer import wer

# class Line():
#     def __init__(self, startx, starty, endx, endy):
#         self.startx = startx
#         self.starty = starty
#         self.endx = endx
#         self.endy = endy
        
#     def __str__(self):
#         return 'Line:{},{},{},{}'.format(self.startx, self.starty, self.endx, self.endy)
#     def lenx(self):
#         return abs(self.startx - self.endx)
    
#     def leny(self):
#         return abs(self.starty - self.endy)
    
#     def toArray(self):
#         return [self.startx, self.starty, self.endx, self.endy]

# def reDrawLine(img, aleft, aright, same_len=True):
#     w, h = img.shape[0], img.shape[1]
#     for r in range(w-1):
#         pixel_white = 0
#         start = 0
#         end = 0
#         for c in range(h-1):
#             if img[r,c] == 255:
#                 pixel_white += 1
#             if img[r, c] == 0 and img[r,c+1] == 255:
#                 start = c
#             if img[r, c] == 255 and img[r,c+1] == 0:
#                 end = c
#         if pixel_white > 20:
#             if same_len:
#                 img[r,aleft:aright] = 255
#             else:
#                 img[r,start:end] = 255
#     return img

# def findMinMaxRow(v_img):
#     aleft, aright = 0, 0
#     list_col = []
#     w, h = v_img.shape[0], v_img.shape[1]
#     #print (w,h)
#     for r in range(w-1):
#         pixel_white = 0
#         for c in range(h-1):
#             if v_img[r,c] == 255:
#                 pixel_white += 1
#         if pixel_white > 20:
#             list_col.append(r)
#     #print("list_col: ", list_col)
#     if (list_col == []):
#         return 0, 0
#     aleft, aright = min(list_col), max(list_col)
#     return aleft, aright

# def getLines(img):
#     lines = []
#     w, h = img.shape[0], img.shape[1]
#     for r in range(w-1):
#         pixel_white = 0
#         startx, starty, endx, endy = 0,0,0,0
#         for c in range(h-1):
#             if img[r,c] == 0 and img[r,c+1] == 255:
#                 startx = c
#                 starty = r
#             if img[r,c] == 255 and img[r,c+1] == 0:
#                 endx = c
#                 endy = r
#             if img[r,c] == 255:
#                 pixel_white += 1
#         if pixel_white > 20:
#             lines.append(Line(startx,starty,endx,endy))
#             #print(Line(startx,starty,endx,endy).toArray())
#     return lines

# def findTable(arr):
#     table = defaultdict(list)
#     for i,b in enumerate(arr):
#         if b[2] < b[3]/2:
#             continue
#         table[str(b[1])].append(b)
#     #print(table)
#     table = [i[1] for i in table.items()]# if len(i[1]) > 1]
#     #print(([len(x) for x in table]))
#     num_cols = max([len(x) for x in table])
#     #print("num_cols:",num_cols)
#     table = [i for i in table if len(i) == num_cols]
#     #print("table rows=", len(table))
#     #print("table cols=",num_cols)
#     print("table size:{}x{}".format(len(table), num_cols))
#     return table

# def getTable(src_img, y_start=0, min_w=3, min_h=3):
#     if y_start != 0:
#         src_img = src_img[y_start:,:]
#     if len(src_img.shape) == 2:
#         gray_img = src_img
#     elif len(src_img.shape) ==3:
#         gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

#     thresh_img = cv2.adaptiveThreshold(~gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, -3)
#     h_img = thresh_img.copy()
#     v_img = thresh_img.copy()
#     scale = 15

#     h_size = int(h_img.shape[1]/scale)
#     h_structure = cv2.getStructuringElement(cv2.MORPH_RECT,(h_size,1))

#     h_erode_img = cv2.erode(h_img,h_structure,1)
#     h_dilate_img = cv2.dilate(h_erode_img,h_structure,1)

#     v_size = int(v_img.shape[0] / scale)
#     v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
#     v_erode_img = cv2.erode(v_img, v_structure, 1)
#     v_dilate_img = cv2.dilate(v_erode_img, v_structure, 1)

    
#     aleft, aright = findMinMaxRow(v_dilate_img.T)
#     aleft2, aright2 = findMinMaxRow(h_dilate_img)

#     h_dilate_img = reDrawLine(h_dilate_img, aleft, aright, True)
#     #v_dilate_img = reDrawLine(v_dilate_img.T, aleft, aright, False).T
#     #cv2.imshow('h_dilate_img',h_dilate_img)
#     #cv2.imshow('h_dilate_img',v_dilate_img)
#     #cv2.waitKey()
#     #list_hlines = getLines(h_dilate_img)
#     #list_vlines = getLines(v_dilate_img.T)
#     #print(len(list_hlines))
#     #print(len(list_vlines))
#     #for i,_ in list_hlines:
#     #    for j,_ in list_hlines
#     #exit()
#     #v_dilate_img = reDrawLine(v_dilate_img.T, aleft2, aright2, True).T
#     v_dilate_img.T[aleft,aleft2:aright2] = 255
#     v_dilate_img.T[aright,aleft2:aright2] = 255
    
#     edges = cv2.Canny(h_dilate_img,50,150,apertureSize = 3) 
#     #cv2.imshow("edge", edges)
#     #print("len edges: ", len(edges))

#     # This returns an array of r and theta values 
#     lines = cv2.HoughLines(edges, 1, np.pi/180, 200) 
#     #print(len(lines))
#     #cv2.waitKey()
#     mask_img = h_dilate_img + v_dilate_img
#     joints_img = cv2.bitwise_and(h_dilate_img, v_dilate_img)
#     #mask_img = 255 - mask_img
#     #mask_img = unsharp_mask(mask_img)
#     convolution_kernel = np.array(
#                                 [[0, 1, 0], 
#                                 [1, 2, 1], 
#                                 [0, 1, 0]]
#                                 )

#     #mask_img = cv2.filter2D(mask_img, -1, convolution_kernel)
#     #mask_img = 255- mask_img
#     #cv2.imshow('mask', mask_img)
#     #cv2.imshow('joints_img', joints_img)
#     #cv2.waitKey()
#     # cv2.imshow('join', joints_img)
#     # cv2.waitKey()
#     # fig, ax = plt.subplots(2,2)
#     # fig.suptitle("table detect")
#     # ax[0,0].imshow(h_dilate_img)
#     # ax[0,1].imshow(v_dilate_img)
#     # ax[1,0].imshow(mask_img)
#     # ax[1,1].imshow(joints_img)
#     # plt.show()cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
#     contours, _ = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#     (contours, boundingBoxes) = cont.sort_contours(contours, method="left-to-right")
#     (contours, boundingBoxes) = cont.sort_contours(contours, method="top-to-bottom")

#     table = findTable([cv2.boundingRect(x) for x in contours])
    
#     # for r in table:
#     #     for c in r:

#     #         cv2.rectangle(src_img,(c[0], c[1]),(c[0] + c[2], c[1] + c[3]),(0, 0, 255), 1)
#     #         cv2.putText(src_img, , (c[0] + c[2]//2,c[1] + c[3]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 2)
#     # for c in contours:
#     #     x, y, w, h = cv2.boundingRect(c)
#     #     if (w >= min_w and h >= min_h):
#     #         #count += 1
#     #         if count != 0:
#     #             cv2.rectangle(src_img,(x, y),(x + w, y + h),(0, 0, 255), 1)
#     #             list_cells.append([x,y,w,h])
#     #             cv2.putText(src_img, str(count), (x+w//2,y+h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
#     #         count += 1
#     #cv2.waitKey()
#     #cv2.imwrite('a.jpg', src_img)
#     return table#mask_img, joints_img

# code minh tu day tro di nha

def getTextOfBox(img):
    return pytesseract.image_to_string(img, config='-l vie+en --oem 1 --psm 6').strip()#.lower()

def putTextUTF8(img, text, point, fsize=10):
    fontpath = "Roboto-Regular.ttf"
    # fontpath = "TNKeyUni-Arial.ttf"
    font = ImageFont.truetype(fontpath, fsize)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(point, text, font = font, fill = ((0,0,0)))
    img = np.array(img_pil)
    return img

def getTableValue(table, img, img_ocr, fsize):
    #img_ocr = img.copy()
    #img_ocr = cv2.cvtColor(img_ocr,cv2.COLOR_BGR2GRAY)
    #print("fsize: ", fsize)
    data = []
    header = []
    for i,row in enumerate(table):
        data_row = []
        for cell in row:
            crop = img_ocr[cell[1]+3:cell[1]+cell[3]-3, cell[0]+3:cell[0]+cell[2]-3]
            #cv2.imwrite(str(i)+".png",crop)
            cell_text = getTextOfBox(crop)
            #print("row "+str(i)+": ", cell_text)
            if i == 0:
                header.append(cell_text)
                cv2.rectangle(img, (cell[0], cell[1]), (cell[0] + cell[2], cell[1] + cell[3]), (0,255,255), -1)
            else:
                cv2.rectangle(img, (cell[0], cell[1]), (cell[0] + cell[2], cell[1] + cell[3]), (0,255,255), -1)
                data_row.append(cell_text)
            img = putTextUTF8(img, cell_text, (cell[0],cell[1]), fsize)
        if i == 0:
            data.append(header)
        else:
            data.append(data_row)
    return data, img

def preprocess(img, factor: int): # preprocess to make image more contrast
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Sharpness(img).enhance(factor)
    if gray.std() < 30:
        enhancer = ImageEnhance.Contrast(enhancer).enhance(factor)
    return np.array(enhancer)

def findHorizontalLine(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #thresh, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh, img_bin = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_bin = 255-img_bin


    kernel_len = gray.shape[1]//120
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    image_horizontal = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_horizontal, hor_kernel, iterations=3)

    h_lines = cv2.HoughLinesP(
    horizontal_lines, 1, np.pi/180, 30, maxLineGap=250)

def group_h_lines(h_lines, thin_thresh):
    new_h_lines = []
    if h_lines is None:
        return new_h_lines
    while len(h_lines) > 0:
        thresh = sorted(h_lines, key=lambda x: x[0][1])[0][0]
        lines = [line for line in h_lines if thresh[1] -
                 thin_thresh <= line[0][1] <= thresh[1] + thin_thresh]
        h_lines = [line for line in h_lines if thresh[1] - thin_thresh >
                   line[0][1] or line[0][1] > thresh[1] + thin_thresh]
        x = []
        for line in lines:
            x.append(line[0][0])
            x.append(line[0][2])
        x_min, x_max = min(x) - int(5*thin_thresh), max(x) + int(5*thin_thresh)
        new_h_lines.append([x_min, thresh[1], x_max, thresh[1]])
    return new_h_lines

def group_v_lines(v_lines, thin_thresh):
    new_v_lines = []
    if v_lines is None:
        return new_v_lines
    while len(v_lines) > 0:
        thresh = sorted(v_lines, key=lambda x: x[0][0])[0][0]
        lines = [line for line in v_lines if thresh[0] -
                 thin_thresh <= line[0][0] <= thresh[0] + thin_thresh]
        v_lines = [line for line in v_lines if thresh[0] - thin_thresh >
                   line[0][0] or line[0][0] > thresh[0] + thin_thresh]
        y = []
        for line in lines:
            y.append(line[0][1])
            y.append(line[0][3])
        y_min, y_max = min(y) - int(4*thin_thresh), max(y) + int(4*thin_thresh)
        new_v_lines.append([thresh[0], y_min, thresh[0], y_max])
    return new_v_lines
    
def seg_intersect(line1: list, line2: list):
    a1, a2 = line1
    b1, b2 = line2
    da = a2-a1
    db = b2-b1
    dp = a1-b1

    def perp(a):
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b

    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float))*db + b1

def get_bottom_right(right_points, bottom_points, points): #get the neareast  right bottom point to create a rectangle
    for right in right_points:
        for bottom in bottom_points:
            if [right[0], bottom[1]] in points:
                return right[0], bottom[1]
    return None, None

def get_vertical_line(table_image): # lay 1 duong thang dau tien sau do xoay cho thang dua vao duong thang do
	# convert both the input image and template to grayscale
    table_image = preprocess(table_image, 2)
    gray = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)
    
    thresh,img_bin = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #thresh, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    table_image = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2RGB)
    
    img_bin = 255-img_bin
    kernel_len = gray.shape[1]//120
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    image_horizontal = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_horizontal, hor_kernel, iterations=3) #cac duong ke nam ngang
    h_lines = cv2.HoughLinesP(
        horizontal_lines, 1, np.pi/180, 30, maxLineGap=250)
    new_horizontal_lines = group_h_lines(h_lines, kernel_len) #nhom cac canh nam ngang voi nhau
    
    kernel_len = gray.shape[1]//120
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    image_vertical = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_vertical, ver_kernel, iterations=3)

    v_lines = cv2.HoughLinesP(vertical_lines, 1, np.pi/180, 30, maxLineGap=250)
    new_vertical_lines = group_v_lines(v_lines, kernel_len) # nhom cac canh nam doc voi nhau
    #print(new_vertical_lines)
    points = []
    for hline in new_horizontal_lines:
        x1A, y1A, x2A, y2A = hline
        for vline in new_vertical_lines:
            x1B, y1B, x2B, y2B = vline
            #print(vline)
            line1 = [np.array([x1A, y1A]), np.array([x2A, y2A])]
            line2 = [np.array([x1B, y1B]), np.array([x2B, y2B])]
            
            x, y = seg_intersect(line1, line2)
            if x1A <= x <= x2A and y1B <= y <= y2B:
                points.append([int(x), int(y)])
    if points == []:
        return None
    return v_lines[0]

def rotate_img(image, vline): # rotate image based on vertical line
    if vline == (None,None):
        return image
    #calculate angle
    rows = image.shape[0]
    cols = image.shape[1]
    #print(vline)
    x1, y1, x2, y2 = vline
    img_center = (cols / 2, rows / 2)
    # print("x1, y1, x2, y2: ", x1, y1, x2, y2)
    # print("goc tao thanh voi truc oy: ", np.rad2deg(np.arctan((x1-x2)/(y1-y2))))
    # print("goc tao thanh voi truc ox: ", np.rad2deg(np.arctan((y2-y1)/(x2-x1))))
    rotate_matrix = cv2.getRotationMatrix2D(center= img_center, angle= -np.rad2deg(np.arctan((x1-x2)/(y1-y2))), scale=1)
    rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(cols, rows), 
                                   borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
    # cv2.imwrite('./output/image_output/tuan4/rotated_img' + str(i) + '.jpg', rotated_image)
    # cv2.imshow("rotated_image: ", rotated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return rotated_image

def get_none_table_image(table_image, cells): #get the image with no table
    if cells == []: 
        cv2.imwrite('./output/image_output/non_table/non_table_image' + str(i) + '.jpg', table_image)
        return table_image, 0, 0
    #print("first line: ", cells[0][1], "; last line: ",  cells[-1][3])
    
    top = cells[0][1]
    bottom = cells[-1][3]
    right = cols = table_image.shape[1]
    
    cv2.rectangle(table_image, (0, top-1), (right, bottom+5), (255,255,255), -1)
    cv2.imwrite('./output/image_output/tuan4/non_table_image' + str(i) + '.jpg', table_image)
    # cv2.imshow("none_table_image: ", table_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return table_image, top, bottom

def OCR_all_image(image, i):  #OCR on non table image
    # outputDir = 'output/image_output/non_table/'
    text = getTextOfBox(image)
    # print(text)
    # imgPath = 'non_table_result' + str(i) + '.jpg'
    # with open(outputDir + imgPath[imgPath.rfind('/') + 1:-3] + 'txt', 'a', encoding='utf-8') as f:
    #     f.write(text)
    return text

inputDir = 'input/'
outputDir = 'output/image_output/tuan5/'


####################################
#######Start Correction model#######
####################################
Filename = "bana_raw_dataset.txt"
dict = {}
threshold = 5
# load doc into memory
def load_data(filename):
	# open the file as read only
	file = open(filename, 'r', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
two_char = ['c̆', 'ĕ', 'ĭ', 'ŏ', 'ơ̆', 'ŭ', 'ư̆', 'C̆', 'Ĕ', 'Ĭ', 'Ŏ', 'Ơ̆', 'Ŭ', 'Ư̆']
three_char = ['ĕ̂', 'ŏ̂', 'Ĕ̂', 'Ŏ̂']
def split_word(text):
    result = re.split('\s', text)
    return result
def split_character(text):
    result = []
    n = len(text)
    i = 0
    while (i < n):
        check = False
        for j in three_char:
            if i+2 < len(text) and text[i] + text[i+1] + text[i+2] == j:
                result.append(j)
                i = i+3
                check = True
                break
        for j in two_char:
            
            if i+1 < len(text) and text[i] + text[i+1] == j:
                # print(j)
                result.append(j)
                i = i+2
                check = True
                break
        if not check:
            result.append(text[i])
            i = i+1
    return result

def text_cleaner(text):
    result = re.sub('[,_“”()]',"", text) # TH thay thế là dấu , ' , "", ( )
    result = re.split('\s', result)
    for i in result:
        if len(i) <= 1:
            result.remove(i)
            continue
        if re.search("^ƀ", i):
            clone = 'b' + i[1:]
            result.append(clone)
    # print(result)
    for i in result:
        if i in dict.keys():
                dict[i] += 1
        else: 
            dict[i] = 1
        # for j in range(0, len(i)-1):
        #     temp = i[j:j+2]
        #     if temp in dict.keys():
        #         dict[temp] += 1
        #     else: 
        #         dict[temp] = 1
        # for j in range(0, len(i)-2):
        #     temp = i[j:j+3]
        #     if temp in dict.keys():
        #         dict[temp] += 1
        #     else: 
        #         dict[temp] = 1
        # for j in range(0, len(i)-3):
        #     temp = i[j:j+4]
        #     if temp in dict.keys():
        #         dict[temp] += 1
        #     else: 
        #         dict[temp] = 1
        # for j in range(0, len(i)-4):
        #     temp = i[j:j+5]
        #     if temp in dict.keys():
        #         dict[temp] += 1
        #     else: 
        #         dict[temp] = 1
CHAR_REPLACE = "a b c d e f g h i j k l m n o p q r s t u v w y z A B C D E F G H I J K L M N O P Q R S T U V ạ ả ã à á â ậ ầ ấ ẩ ẫ ă ắ ằ ặ ẳ ẵ ó ò ọ õ ỏ ô ộ ổ ỗ ồ ố ơ ờ ớ ợ ở ỡ é è ẻ ẹ ẽ ê ế ề ệ ể ễ ú ù ụ ủ ũ ư ự ữ ử ừ ứ í ì ị ỉ ĩ ý ỳ ỷ ỵ ỹ đ Ạ Ả Ã À Á Â Ậ Ầ Ấ Ẩ Ẫ Ă Ắ Ằ Ặ Ẳ Ẵ Ó Ò Ọ Õ Ỏ Ô Ộ Ổ Ỗ Ồ Ố Ơ Ờ Ớ Ợ Ở Ỡ É È Ẻ Ẹ Ẽ Ê Ế Ề Ệ Ể Ễ Ú Ù Ụ Ủ Ũ Ư Ự Ữ Ử Ừ Ứ Í Ì Ị Ỉ Ĩ Ý Ỳ Ỷ Ỵ Ỹ Đ a ă â b ƀ c̆ d đ e ĕ ê ĕ̂ g h i ĭ j k l m n ñ o ŏ ô ŏ̂ ơ ơ̆ p r s t u ŭ ư ư̆ w y f q v z A Ă Â B Ƀ C̆ D Đ E Ĕ Ê Ĕ̂ G H I Ĭ J K L M N Ñ O Ŏ Ô Ŏ̂ Ơ Ơ̆ P R S T U Ŭ Ư Ư̆ W Y F Q V Z"
char = CHAR_REPLACE.split(' ')
char_skip = ['"', "'", '“', '”', '(', ')', '_', ',', ';', ':', '.']
char_delete = ['@','#','$', '%', '|', '&', '*']
def findreplacement(text):
    result = 0
    current = None
    char_list = split_character(text)
    for i in range(len(char_list)): # duyet tren substring bi sai
        for j in char: # j la kí tự để thay thế
            pre_substr = ""
            after_substr = ""
            for l in range(0,i):
                pre_substr += char_list[l]
            for l in range(i+1,len(char_list)):
                after_substr += char_list[l]
            temp = pre_substr + j + after_substr
            if temp in dict and dict[temp] >= threshold and dict[temp] >= result:
                # print(i, temp, dict[temp])
                result = dict[temp]
                current = temp
    
    return current # current là từ có xác xuất đúng cao nhất               
def correction(text): # phiên bản hiện tại correct được cho cả câu.
    char_list_raw = split_character(text)
    char_list = []
    char_list_position = []
    for i in char_list_raw:
        if i in char_delete:
            char_list_raw.remove(i)
    for i in range(len(char_list_raw)):
        if char_list_raw[i] == '[':
            char_list_raw[i] = 'l'
        if char_list_raw[i] not in char_skip:
            char_list.append(char_list_raw[i])
            char_list_position.append(i)
    length = len(char_list_raw)
    # print(char_list)
    for i in range(0,1):
    # for i in range(len(char_list)):
        for j in reversed(range(2, 7)):
            if i+j > len(char_list):
                continue
            substr = ""
            for k in range(0, j):
                substr = substr + char_list[i+k]
            # print(i, j, substr)
            if (substr not in dict or dict[substr] < threshold): 
                # print(substr)
                sub_result = findreplacement(substr)
                if(not sub_result):
                    continue
                pre_substr = ""
                after_substr = ""
                for l in range(0,i):
                    char_list_raw[char_list_position[l]] = char_list[l] 
                    
                for m in range(0, char_list_position[i]):
                    pre_substr += char_list_raw[m]    
                
                for l in range(i+j,len(char_list)):
                    char_list_raw[char_list_position[l]] = char_list[l] 
                    
                for m in range(char_list_position[i+j-1]+1, length):
                    after_substr += char_list_raw[m]
                
                result = pre_substr + sub_result + after_substr
                # print("Start: ", i, " ; End: ", i+j-1)
                return result
            else:
                break
    return text
def correction_sentence(text):
    final_result = ""
    word_list = split_word(text)
    for word in word_list: # tách từ từ câu
        if word == "" or word in char_delete:
            word_list.remove(word)
    for i in word_list:
        if i in dict and dict[i] >= threshold:
            if final_result == "":
                final_result = i
                continue
            else:
                final_result = final_result + " " + i
                continue
        if final_result == "":
            final_result = correction(i)
        else:
            final_result = final_result + " " + correction(i) 
    return final_result
raw_text = load_data(Filename)
text_cleaner(raw_text)

# test = "kơkăš"
test1 = "“pở”"
# test2 = "Cách \n'phót'"
# test3 = "Pore \n(Cách 'phót' âm)"
# # # print(test[3:5])

# # print(correction_sentence(test))
print(correction_sentence(test1))
# print(correction_sentence(test2))
# print(correction_sentence(test3))
####################
####################
####################


for i in range(3, 4):
    ############### code tuan 6 #################3
    print("Image ", i)
    table_image  = cv2.imread('./input/crop_page'+ str(i) +'.jpg') # get image
    
    if get_vertical_line(table_image) is not None: # check if there are vertical line then rotate using the first one
        table_image = rotate_img(table_image, get_vertical_line(table_image)[0])
    
    table_image_clone = table_image
       
    table_image = preprocess(table_image, 2)
    
    gray = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)
    
    thresh,img_bin = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #thresh, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    table_image_bin = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2RGB)
    
    # erode and dialate to find horizontal lines and vertical lines then group them to find points
    img_bin = 255-img_bin
    kernel_len = gray.shape[1]//120
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    image_horizontal = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_horizontal, hor_kernel, iterations=3) #cac duong ke nam ngang
    h_lines = cv2.HoughLinesP(
        horizontal_lines, 1, np.pi/180, 30, maxLineGap=250)
    new_horizontal_lines = group_h_lines(h_lines, kernel_len) #nhom cac canh nam ngang voi nhau
    
    kernel_len = gray.shape[1]//120
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    image_vertical = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_vertical, ver_kernel, iterations=3)

    v_lines = cv2.HoughLinesP(vertical_lines, 1, np.pi/180, 30, maxLineGap=250)
    new_vertical_lines = group_v_lines(v_lines, kernel_len) # nhom cac canh nam doc voi nhau
    #print(new_vertical_lines)
    points = []
    for hline in new_horizontal_lines:
        x1A, y1A, x2A, y2A = hline
        for vline in new_vertical_lines:
            x1B, y1B, x2B, y2B = vline
            #print(vline)
            line1 = [np.array([x1A, y1A]), np.array([x2A, y2A])]
            line2 = [np.array([x1B, y1B]), np.array([x2B, y2B])]
            
            x, y = seg_intersect(line1, line2)
            if x1A <= x <= x2A and y1B <= y <= y2B:
                points.append([int(x), int(y)])
                
    cells = []
    j = 0
    imgPath = 'table_result' + str(i) + '.jpg'
    table_ocr_result = ""
    with open(outputDir + imgPath[imgPath.rfind('/') + 1:-3] + 'txt', 'w', encoding='utf-8') as f:
        
        for point in points: # loop in points to find cells
            j = j + 1
            
            left, top = point
            right_points = sorted(
                [p for p in points if p[0] > left and p[1] == top], key=lambda x: x[0])
            bottom_points = sorted(
                [p for p in points if p[1] > top and p[0] == left], key=lambda x: x[1])

            right, bottom = get_bottom_right(right_points, bottom_points, points)
            if right and bottom:
                
                if top + 7 > bottom -2 or left+7 > right -1:
                    print("loi")
                    continue
                
                crop = table_image_bin[top+7:bottom-2, left+8:right-1]
                # cv2.imwrite('./output/image_output/tuan3/'+str(j)+".jpg",crop)
                cell_text = getTextOfBox(crop)
                # print("Raw text: ",cell_text)
                # cell_text = correction(cell_text)
                try:
                    print("Pre: ", cell_text, " ; After: ", correction_sentence(cell_text))
                    cell_text = correction_sentence(cell_text)
                except Exception as e:
                    print(e)
                    pass 
                    
                cv2.rectangle(table_image_clone, (left, top), (right, bottom), (0, 0, 255), 2) # draw rectangle
                cv2.rectangle(table_image_clone, (left+1, top+1), (right-1, bottom-1), (0,255,255), -1) # fill in the rectangle yellow collor
                if cell_text != "":
                    
                    table_image_clone = putTextUTF8(table_image_clone, cell_text, (left+3, top+5), 30) # put text inside the rectangle
                if cell_text == "":
                    continue
                cell_text = re.sub("\n", " ", cell_text)
                cells.append([left, top, right, bottom, cell_text]) # left, top, right, bottom lan luot la x1, y1, x2, y2
            
                # print("cell text: ", point, right, bottom , cell_text)
                
                                
        print("so luong cell: ", len(cells))        
        curr = 0
        for j in range(0, len(cells)):
            if cells[j][3] != curr:
                if curr != 0:
                    table_ocr_result += "\n"
                curr = cells[j][3]
            
            cell_text = cells[j][4]
            if j != len(cells)-1 and cells[j][3] != cells[j+1][3]:
                table_ocr_result += cell_text
            else:
                table_ocr_result += cell_text + " | "
                                 
        
        none_table_image, top, bottom = get_none_table_image(table_image_bin, cells)
        # get result of nontable 
        cell_text= OCR_all_image(none_table_image[0:top-1,], i)
        if top != 0 and cell_text != "" :
            # print("Pre: ", cell_text, "\nAfter: ", "")
            cell_text = correction_sentence(cell_text)
            # try:
            #     cell_text = self_correction(OCR_all_image(none_table_image[0:top-1,], i))
            # except Exception as e:
            #     print("head: ", e)
            table_ocr_result = cell_text + "\n" + table_ocr_result
        cell_text = OCR_all_image(none_table_image[bottom+5:, ], i)
        if bottom + 5 < none_table_image.shape[0] and cell_text != "" :
            cell_text = correction_sentence(cell_text)
            # try:
            #     cell_text = self_correction(cell_text)
            # except Exception as e:
            #     print("tail: ", e)
            #     pass
            table_ocr_result += "\n" + cell_text 
        
        f.write(table_ocr_result)
    # cv2.imwrite('./output/image_output/tuan5/result_page' + str(i) + '.jpg', table_image_clone) # save image
    
    # cv2.imshow("table_image: ", table_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    ############# end code tuan 3 ##############
    
    
    ############# code tuan 2 ##################
    # img  = cv2.imread('./input/crop_page'+ str(i) +'.jpg')
    # img = preprocess(img, 3)
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print("anh thu ",i,": ", img.shape)
    # blur = cv2.GaussianBlur(image,(5,5),0)
    # #thresh, th3 = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # # cv2.imshow("anh fix", th3)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    # img = cv2.cvtColor(th3, cv2.COLOR_GRAY2RGB)
    # crop_img = img
    # img2 = crop_img.copy()
    # imgPath = 'crop_page' + str(i) + '.jpg'
    
    # table = getTable(crop_img)
    # data, img = getTableValue(table, crop_img, img2, 30)
    # cv2.imwrite('./output/image_output/result_page' + str(i) + '.jpg', img)
    # print("data: ", data)
    
    # custom_oem_psm_config = '-l vie+en --oem 1 --psm 6'
    # with open(outputDir + imgPath[imgPath.rfind('/') + 1:-3] + 'txt', 'w', encoding='utf-8') as f:
    #     for line in data:
    #         f.write(" | ".join(line) + "\n")