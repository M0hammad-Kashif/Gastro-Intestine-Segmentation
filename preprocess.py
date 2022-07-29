from imports import *

path_df = pd.DataFrame(glob('images/images/*'), columns=['image_path'])
path_df['mask_path'] = path_df.image_path.str.replace('image', 'mask')
path_df['id'] = path_df.image_path.map(lambda x: x.split('/')[-1].replace('.npy', ''))

df = pd.read_csv('train.csv')
df['segmentation'] = df.segmentation.fillna('')
df['rle_len'] = df.segmentation.map(len)

df2 = df.groupby(['id'])['segmentation'].agg(list).to_frame().reset_index()
df2 = df2.merge(df.groupby(['id'])['rle_len'].agg(sum).to_frame().reset_index())

df = df.drop(columns=['segmentation', 'class', 'rle_len'])
df = df.groupby(['id']).head(1).reset_index(drop=True)
df = df.merge(df2, on=['id'])
df['empty'] = (df.rle_len == 0)  # empty masks

df = df.drop(columns=['image_path', 'mask_path'])
df = df.merge(path_df, on=['id'])
