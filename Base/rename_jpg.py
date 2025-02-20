import os

dir = '../../../../TEMP/0129/000010'
jpg_paths = os.listdir(dir)
jpg_paths.sort()
for i, jpg_path in enumerate(jpg_paths):
    new_jpg_path = '200' + jpg_path[4:-4].zfill(3) + '.jpg'
    # new_jpg_path = 'temp{}.json'.format(i+1)
    os.rename(os.path.join(dir, jpg_path), os.path.join(dir, new_jpg_path))