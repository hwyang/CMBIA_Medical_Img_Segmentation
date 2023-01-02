import pydicom
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

LOG_LENGTH = 100
LOAD_SLICES_NUM = 25
FOLDER_NAME = 'CT_chest_scans/0a0c32c9e08cc2ea76a71649de56be6d/'
DICOM_NAME = FOLDER_NAME + '0a67f9edb4915467ac16a565955898d3.dcm'

def loggingTitle(title, length):
    num_of_dash = length - (len(title) + 2)
    left_num_of_dash = num_of_dash // 2
    right_num_of_dash = num_of_dash - left_num_of_dash
    print('\n\n' + '-' * left_num_of_dash + " " + title + " " + '-' * right_num_of_dash)

def getDicomDataFields(file_path, log=False):
    ds = pydicom.read_file(file_path)
    if log:
        print(ds)
    return ds

def trans2Housefield(ds):
    for row in ds.pixel_array:
        for col in row:
            if col < ds.PixelPaddingValue:
                col = 0
    housefield_values = ds.RescaleSlope * ds.pixel_array + ds.RescaleIntercept
    return housefield_values

def getStatistics(data, data_type='raw_data', log=False):
    data = data.reshape(-1)
    min_data = np.min(data)
    max_data = np.max(data)
    mean_data = np.mean(data)
    std_data = np.std(data)
    if log:
        print(f'{data_type:15} {min_data:<15} {max_data:<15} {mean_data:<15.3f} {std_data:<15.3f}')
    return

def loadSlices(dir_path, get_slices_num, log=False):
    def normalize(data):
        norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        return norm_data
    if log:
        loggingTitle('part 2: Read slices', LOG_LENGTH)
    all_slices = [pydicom.read_file(dir_path + '/' + file_name) for file_name in os.listdir(dir_path)]

    if log:
        loggingTitle('part 2: Sort slices', LOG_LENGTH)
    all_slices.sort(key=lambda x: x.ImagePositionPatient[2])

    if log:
        loggingTitle('part 2: Transform to HouseField Unit and Normalize Slices', LOG_LENGTH)
    slices_data = list()
    for counts, slice in list(enumerate(all_slices, start=1)):
        slice_data = trans2Housefield(slice)
        slice_data = normalize(slice_data)
        slices_data.append(slice_data)
        if counts == get_slices_num:
            break
    return np.array(slices_data, dtype=np.float32)

def plotSlices(slices, save_path):
    col_num = 5
    row_num = slices.shape[0] // col_num
    if slices.shape[0] % col_num != 0:
        row_num += 1
    f, plots = plt.subplots(row_num, col_num, figsize=(20, 20))
    for i in range(row_num):
        for j in range(col_num):
            idx = i * col_num + j
            plots[i, j].axis('off')
            plots[i, j].imshow(slices[idx], cmap='binary_r')
            if idx == slices.shape[0]-1:
                plt.savefig(save_path)
                print(f'saving figure to "{save_path}"')
                plt.close()
                return

def getSegementSlice(slice, thres_method='mean'):
    if thres_method == 'mean' or thres_method == 'median':
        block_size = 21
        thresholds = threshold_local(slice, block_size, method=thres_method)
        seg_slice = slice > thresholds
        threshold = np.mean(thresholds.reshape(-1))
    else:
        raise ValueError('thres_method must be "mean" or "median"')
    return seg_slice, threshold

def plotSegmentSlice(thres_method='mean'):
    slice = loadSlices(FOLDER_NAME, 1)[0]
    seg_slice, threshold = getSegementSlice(slice=slice, thres_method=thres_method)
    flat_slice = slice.flatten()

    f, plots = plt.subplots(3, 1, figsize=(70, 70))
    font_size = 100

    plots[0].set_title('histogram', fontsize=font_size)
    plots[0].hist(flat_slice, bins=100, color='steelblue')
    plots[0].axvline(threshold, min(flat_slice), max(flat_slice), lw=25, color='red')
    plt.sca(plots[0])
    plt.xticks(fontsize=font_size-20)
    plt.yticks(fontsize=font_size-20)

    plots[1].set_title('raw pixels', fontsize=font_size)
    plots[1].axis('off')
    plots[1].imshow(slice, cmap='binary_r')

    plots[2].set_title('threshold pixels', fontsize=font_size)
    plots[2].axis('off')
    plots[2].imshow(seg_slice, cmap='binary_r')

    save_path = 'part3_' + thres_method + '.png'
    plt.savefig(save_path)
    plt.close()
    print(f'saving figure to "{save_path}"')

def part1():
    loggingTitle('Part 1: Read and Print Dicom DataFields', LOG_LENGTH)
    ds = getDicomDataFields(DICOM_NAME, log=True)

    loggingTitle('Part 1: Print Statistics', LOG_LENGTH)
    raw_data = ds.pixel_array
    housefield_data = trans2Housefield(ds)
    print('{:15s} {:15s} {:15s} {:15s} {:15s}'.format('Data Type', 'Min Data', 'Max Data', 'Mean Data', 'Std Data'))
    getStatistics(raw_data, data_type='Raw Data', log=True)
    getStatistics(housefield_data, data_type='Housefield Data', log=True)

def part2():
    slices = loadSlices(FOLDER_NAME, LOAD_SLICES_NUM, log=True)
    loggingTitle('Part 2: Plot Slices', LOG_LENGTH)
    plotSlices(slices, 'part2.png')

def part3():
    loggingTitle('Part 3: Plot Segment Slice - Mean', LOG_LENGTH)
    plotSegmentSlice(thres_method='mean')

    loggingTitle('Part 3: Plot Segment Slice - Median', LOG_LENGTH)
    plotSegmentSlice(thres_method='median')

def plot_3d(image, threshold):
    p = image.transpose(2, 1, 0)
    verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.savefig('bonus.png')
    #plt.show()

def bonus():
    scan = loadSlices(FOLDER_NAME, None)
    threshold = np.mean(np.array([getSegementSlice(slice, thres_method='mean')[1] for slice in scan]))
    # threshold = None
    loggingTitle('bonus part[ plot scan ]', LOG_LENGTH)
    plot_3d(scan, threshold)

def main():
    part1()
    part2()
    part3()
    bonus()


if __name__ == '__main__':
    main()