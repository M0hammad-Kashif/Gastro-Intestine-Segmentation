from imports import *

path_df = pd.DataFrame(glob('/kaggle/input/uwmgi-25d-stride2-dataset/images/images/*'), columns=['image_path'])
path_df['mask_path'] = path_df.image_path.str.replace('image', 'mask')
path_df['id'] = path_df.image_path.map(lambda x: x.split('/')[-1].replace('.npy', ''))
