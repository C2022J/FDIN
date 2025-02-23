import os, shutil

source_dir = '/Storage/data/Stereo_waterdrop_removal/test_mynt/'
save_dir = '/Storage2/leo/StereoDerain/data/StereoWaterdrop/test_mynt/'
for dir_ in os.listdir(source_dir):
    for file_name in os.listdir(os.path.join(source_dir, dir_)):
        file_path = os.path.join(source_dir, dir_, file_name)
        save_name = dir_ + '_' + file_name
        if not file_name in ['000_0.png', '000_1.png']:
            if file_name.endswith('0.png'):
                save_path = os.path.join(save_dir, 'input', 'left', save_name[:-6]+'.png' )
                save_gt_path = os.path.join(save_dir, 'gt', 'left', save_name[:-6]+'.png' )
                shutil.copy(file_path, save_path)
                gt_path = os.path.join(source_dir,
                                       dir_, '000_0.png')
                shutil.copy(gt_path, save_gt_path)

            else:
                save_path = os.path.join(save_dir, 'input', 'right', save_name[:-6]+'.png' )
                save_gt_path = os.path.join(save_dir, 'gt', 'right', save_name[:-6]+'.png' )

                shutil.copy(file_path, save_path)
                gt_path = os.path.join(source_dir, dir_, '000_1.png')
                shutil.copy(gt_path, save_gt_path)
            # save_path
