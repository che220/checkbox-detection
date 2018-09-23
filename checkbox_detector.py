import os, sys, numpy as np, pandas as pd, re, datetime as dt, subprocess
from utils import display_cv2_image
import cv2

pd.set_option('display.width', 5000)
pd.set_option('max_columns', 600)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

np.set_printoptions(threshold=np.nan) # print out all values, regardless length

class Checkbox:
    def __init__(self, box, text_box, text):
        self.box = box
        self.text_box = text_box
        self.text = text

def to_dataframe(text_loc):
    '''
    Turn tesseract TSV output into pandas Dataframe, and remove lines without texts

    :param text_loc:
    :return:
    '''
    lines = re.split('\n', text_loc)
    headers = re.split('\t', lines[0])
    print('Headers:', headers)
    text_idx = headers.index('text')
    n_cols = len(headers)

    rs = []
    for line in lines[1:]:
        flds = np.array(re.split('\t', line))
        if len(flds) != n_cols:
            continue

        if len(flds) <= text_idx:
            # these lines have no text. Discard them.
            continue
        flds[text_idx] = flds[text_idx].strip()
        text = flds[text_idx]
        if len(text) == 0:
            # these lines have empty strings as text. Discard them.
            continue
        if re.match('[0-9a-zA-Z]', text) is None:
            # the line must contain at one alphanumeric char
            continue
        tmp = text.upper()
        if len(tmp) == 1 and tmp in ('X', 'J', '1', 'I'):
            # X, J, 1, or I can be checkbox status
            continue

        rs = np.concatenate((rs, flds))
    rs = np.reshape(rs, (-1, len(headers)))
    df = pd.DataFrame(rs, columns=headers)
    for col in headers:
        if col == 'text':
            continue
        df[col] = df[col].astype('int')
    df['bottom'] = df.top + df.height
    df['right'] = df.left + df.width
    print(df)
    return df

def form_lines(extracted_df):
    """
    Adjacent words form lines. Every element of returned list is [left_top, bottom_right, text]

    :param extracted_df:
    :return: list
    """
    # if a row's word_num is less than or equal to the word_num of the row above, change it to 1
    # it is possible for you to see word_num starts from 2, not 1!!
    col_idx = list(extracted_df.columns).index('word_num')
    for i, row in extracted_df.iterrows():
        if i == 0:
            extracted_df.iloc[0, col_idx] = 1 # sometimes the first line starts with 2!
            continue

        word_num = row['word_num']
        if word_num == 1:
            continue

        last_word_num = extracted_df.iloc[i-1, :]['word_num']
        if word_num <= last_word_num:
            #print('change row:', row)
            extracted_df.iloc[i, col_idx] = 1

    word_1_df = extracted_df[extracted_df.word_num == 1]
    rs = []
    for i, row in enumerate(word_1_df.index):
        end_row = extracted_df.shape[0]
        if i < word_1_df.index.shape[0] - 1:
            end_row = word_1_df.index[i + 1]

        a_df = extracted_df.iloc[row:end_row].copy().reset_index(drop=True)
        a_df['last_right'] = [-10000] + list(a_df.right[0:-1])
        a_df['gap'] = a_df.left - a_df.last_right
        #print(a_df)

        # use the longest word to figure out the width of 2 letters. That will be max gap allowed between words
        # Otherwise break up line into multiple lines
        max_text_row = a_df[a_df.width == a_df.width.max()].iloc[0, :]
        two_letter_width = max_text_row['width'] // len(max_text_row['text']) * 2
        a_df.loc[a_df.gap > two_letter_width, 'word_num'] = 1 # set multiple 1s to indicate multiple lines
        #print(a_df)
        #print('max allowed gap:', max_word_gap)

        if a_df[a_df.word_num == 1].shape[0] > 1:
            sub_rs = form_lines(a_df)
            rs += sub_rs
        else:
            # Box: top, left, bottom, right
            left_top = (a_df.left.min(), a_df.top.min())
            right_bott = (a_df.right.max(), a_df.bottom.max())
            text = ' '.join(a_df.text)
            rs.append([left_top, right_bott, text])
        #print('texts:', rs)
    return rs

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img, max_area, min_area):
    img = cv2.GaussianBlur(img, (5, 5), 0) # no difference if we do not blur
    squares = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 60): # step 26 and 60 gave the same results
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            bin, contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            #print('thrs=', thrs, ' num contours=', len(contours))
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if not len(cnt) == 4:
                    continue
                if not cv2.isContourConvex(cnt):
                    continue

                area = cv2.contourArea(cnt)
                if area >= min_area and area <= max_area:
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return img, squares

def find_squares_gray_first(img, max_area):
    '''
    By graying, the image lost some info.

    :param img:
    :return:
    '''
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    ret3, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    squares = []
    for thrs in range(0, 255, 26):
        if thrs == 0:
            bin = cv2.Canny(img, 0, 50, apertureSize=5)
            bin = cv2.dilate(bin, None)
        else:
            _retval, bin = cv2.threshold(img, thrs, 255, cv2.THRESH_BINARY)
        bin, contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        print('thrs=', thrs, ' num contours=', len(contours))
        for cnt in contours:
            cnt_len = cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
            area = cv2.contourArea(cnt)
            if len(cnt) == 4 and area > 100 and area <= max_area and cv2.isContourConvex(cnt):
                cnt = cnt.reshape(-1, 2)
                max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                if max_cos < 0.1:
                    squares.append(cnt)
    return img, squares

def rectangle_center(rect):
    '''
    :param rect: list of 2 points: [left, top], [right, bottom]
    :return:
    '''
    x = (rect[0][0] + rect[1][0]) // 2
    y = (rect[0][1] + rect[1][1]) // 2
    return x, y

def rectangle_distance(rect1, rect2):
    '''
    distance between gravity centers

    :param rect1: list of 2 points: [left, top], [right, bottom]
    :param rect2: list of 2 points: [left, top], [right, bottom]
    :return:
    '''
    x1, y1 = rectangle_center(rect1)
    x2, y2 = rectangle_center(rect2)
    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    angle = np.arcsin((y2-y1)/dist) * 180/np.pi # in degrees
    return dist, angle

def rectangle_overlap_area(rect1, rect2):
    return max(0, min(rect1[1][0], rect2[1][0]) - max(rect1[0][0], rect2[0][0])) \
           * max(0, min(rect1[1][1], rect2[1][1]) - max(rect1[0][1], rect2[0][1]))

def get_squares(img, text_boxes):
    # figure out the range of checkbox areas
    max_area, min_area = 0, 100000000
    for box in text_boxes:
        height = box[1][1] - box[0][1]
        if height >= 50:
            continue
        print(box[2], '\t', height)
        area = height ** 2
        max_area = max(area, max_area)
        min_area = min(area, min_area)
    max_area *= 1.5
    min_area *= 0.5
    print('max_area:', max_area, 'min_area:', min_area)

    _, squares = find_squares(img, max_area, min_area)
    # cv2.drawContours(img, squares, -1, (0, 255, 0), 3)
    sqs = []
    for sq in squares:
        left_top = tuple(sq.min(axis=0))
        right_bott = tuple(sq.max(axis=0))
        # cv2.rectangle(img, left_top, right_bott, (0, 255, 0), 2)
        sqs.append([left_top, right_bott])
    return  sqs

def link_squares_with_texts(boxes, text_boxes):
    # link checkbox squares with texts
    checkboxes = []
    for sq in boxes:
        closest_text, closest_dist = None, 100000000
        for text_box in text_boxes:
            if rectangle_overlap_area(sq, text_box) > 0:
                continue
            dist, angle = rectangle_distance(sq, text_box)
            if dist < closest_dist:
                closest_text, closest_dist = text_box, dist

        if closest_text is not None:
            checkboxes.append(Checkbox(sq, closest_text[0:2], closest_text[2]))
    return checkboxes

def display_checkboxes(img, checkboxes):
    feature_img = np.copy(img)
    for sq in sqs:
        cv2.rectangle(feature_img, sq[0], sq[1], (0, 255, 255), box_thickness)
    for cb in checkboxes:
        cv2.rectangle(feature_img, cb.box[0], cb.box[1], (0, 255, 0), box_thickness)
        cv2.rectangle(feature_img, cb.text_box[0], cb.text_box[1], (0, 0, 255), box_thickness)

        box_center = rectangle_center(cb.box)
        text_center = rectangle_center(cb.text_box)
        cv2.line(feature_img, box_center, text_center, (255, 0, 0), box_thickness)
    display_cv2_image(feature_img, 'Text & Checkbox')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage:', sys.argv[0], '<image file>')
        exit(1)

    img_file = sys.argv[1]
    cmd = ['/usr/local/bin/tesseract', img_file, 'stdout', '--oem', '1', '-l', 'eng', '--psm',  '3', 'tsv']
    t0 = dt.datetime.now()
    text_loc = subprocess.check_output(cmd).decode('utf-8')
    t1 = dt.datetime.now()
    print('OCR:', t1 - t0)
    df = to_dataframe(text_loc)
    lines = form_lines(df)
    t1 = dt.datetime.now()
    print('OCR+post processing:', t1 - t0)

    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    display_cv2_image(img, 'Original')

    box_thickness = 3
    feature_img = np.copy(img)
    for sq in lines:
        cv2.rectangle(feature_img, sq[0], sq[1], (0, 255, 0), box_thickness)
    display_cv2_image(feature_img, 'Text')

    t0 = dt.datetime.now()
    sqs = get_squares(img, lines)
    t1 = dt.datetime.now()
    print('find squares+post processing:', t1 - t0)

    feature_img = np.copy(img)
    for sq in sqs:
        cv2.rectangle(feature_img, sq[0], sq[1], (0, 255, 0), box_thickness)
    display_cv2_image(feature_img, 'Squares')

    # link checkbox squares with texts
    checkboxes = link_squares_with_texts(sqs, lines)
    display_checkboxes(img, checkboxes)

    print('displayed')
    cv2.waitKey(0) & 0xFF # for 64-bit machine
    cv2.destroyAllWindows()
