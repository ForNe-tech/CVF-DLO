import os
import shutil

pic_dir = '../../../TEMP/pic'
json_dir = '../../../../TEMP/json'

def split_samples(beg, end, title):
    new_dir = '../../../TEMP/' + title
    os.makedirs(new_dir, exist_ok=True)
    for i in range(beg, end + 1):
        pic_path = os.path.join(pic_dir, 'temp{}.jpg'.format(i))
        json_path = os.path.join(json_dir, 'temp{}.json'.format(i))
        shutil.move(pic_path, new_dir)
        shutil.move(json_path, new_dir)

if __name__ == "__main__":
    # split_samples(0, 63, 'front')
    split_samples(71, 150, 'left1')
    split_samples(157, 223, 'left2')
    split_samples(234, 282, 'left3')
    split_samples(291, 349, 'left4')
    split_samples(389, 432, 'right1')
    split_samples(446, 478, 'right2')
    split_samples(497, 513, 'right4')
    split_samples(579, 601, 'front_left2')
    split_samples(616, 628, 'front_right2')
    split_samples(637, 656, 'front_right1')
    split_samples(668, 702, 'front_left1')
